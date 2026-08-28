"""
Scene handling: a large image split into 1 km2 work tiles.

A 100 km2 scene at 0.3 m GSD is ~333k x 333k px (~110 gigapixels) and never
fits in RAM, so we read windows on demand. Two backends:

  * RasterScene  - a real GeoTIFF read through rasterio windowed reads. Pixel
                   <-> map coordinate transforms are available for GeoJSON out.
  * ArrayScene   - an in-memory numpy image, for the synthetic demo / tests.

Both expose the same interface: size, a tile grid, read_window(), and a
downsampled overview() for navigation. Embedded feature grids for visited tiles
are cached with a simple LRU so re-training on the work tile is instant.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TileSpec:
    """A tile's placement in the full scene, in scene-pixel coordinates."""
    col: int
    row: int
    x: int
    y: int
    w: int
    h: int

    @property
    def id(self) -> str:
        return f"{self.col}_{self.row}"


class Scene:
    """Base scene interface."""

    width: int
    height: int
    gsd_m: float          # ground sample distance, metres per pixel
    tile_px: int          # side length of a work tile in pixels (~1 km2)

    def read_window(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        raise NotImplementedError

    def overview(self, max_side: int = 1024) -> np.ndarray:
        raise NotImplementedError

    def pixel_to_lonlat(self, x: float, y: float) -> tuple[float, float] | None:
        return None

    # -- tiling ------------------------------------------------------------- #
    def tiles(self) -> list[TileSpec]:
        specs = []
        cols = (self.width + self.tile_px - 1) // self.tile_px
        rows = (self.height + self.tile_px - 1) // self.tile_px
        for r in range(rows):
            for c in range(cols):
                x = c * self.tile_px
                y = r * self.tile_px
                w = min(self.tile_px, self.width - x)
                h = min(self.tile_px, self.height - y)
                specs.append(TileSpec(c, r, x, y, w, h))
        return specs

    def read_tile(self, spec: TileSpec) -> np.ndarray:
        return self.read_window(spec.x, spec.y, spec.w, spec.h)


class ArrayScene(Scene):
    def __init__(self, img: np.ndarray, gsd_m: float = 0.3, tile_km: float = 1.0):
        assert img.ndim == 3 and img.shape[2] == 3, "expect HxWx3 uint8"
        self.img = img
        self.height, self.width = img.shape[:2]
        self.gsd_m = gsd_m
        self.tile_px = max(1, int(round((tile_km * 1000.0) / gsd_m)))

    def read_window(self, x, y, w, h):
        return self.img[y:y + h, x:x + w].copy()

    def overview(self, max_side=1024):
        scale = max(self.width, self.height) / max_side
        if scale <= 1:
            return self.img.copy()
        step = int(np.ceil(scale))
        return self.img[::step, ::step].copy()


class RasterScene(Scene):
    def __init__(self, path: str, gsd_m: float | None = None, tile_km: float = 1.0):
        import rasterio  # lazy
        self._rasterio = rasterio
        self.path = path
        self.ds = rasterio.open(path)
        self.width = self.ds.width
        self.height = self.ds.height
        # infer GSD from the transform if not given (assumes projected CRS in m)
        if gsd_m is None:
            gsd_m = float(abs(self.ds.transform.a))
        self.gsd_m = gsd_m
        self.tile_px = max(1, int(round((tile_km * 1000.0) / gsd_m)))

    def _read_bands(self, window) -> np.ndarray:
        count = min(3, self.ds.count)
        arr = self.ds.read(list(range(1, count + 1)), window=window)  # [C,H,W]
        arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        if arr.dtype != np.uint8:
            # simple percentile stretch to 8-bit for display/backbone input
            lo, hi = np.percentile(arr, (2, 98))
            arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255
            arr = arr.astype(np.uint8)
        return arr

    def read_window(self, x, y, w, h):
        from rasterio.windows import Window
        return self._read_bands(Window(x, y, w, h))

    def overview(self, max_side=1024):
        from rasterio.windows import Window
        scale = max(self.width, self.height) / max_side
        out_w = int(self.width / max(scale, 1))
        out_h = int(self.height / max(scale, 1))
        count = min(3, self.ds.count)
        arr = self.ds.read(
            list(range(1, count + 1)),
            out_shape=(count, out_h, out_w),
            window=Window(0, 0, self.width, self.height),
        )
        arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        if arr.dtype != np.uint8:
            lo, hi = np.percentile(arr, (2, 98))
            arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255
            arr = arr.astype(np.uint8)
        return arr

    def pixel_to_lonlat(self, x, y):
        try:
            from rasterio.warp import transform as warp_transform
            xs, ys = self.ds.xy(y, x)
            lon, lat = warp_transform(self.ds.crs, "EPSG:4326", [xs], [ys])
            return float(lon[0]), float(lat[0])
        except Exception:
            return None


class FeatureCache:
    """Bounded LRU cache of embedded feature grids, keyed by tile id."""

    def __init__(self, max_items: int = 8):
        self.max_items = max_items
        self._store: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, key: str) -> np.ndarray | None:
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def put(self, key: str, value: np.ndarray) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.max_items:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()
