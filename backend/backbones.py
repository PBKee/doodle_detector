"""
Frozen feature extractors for the doodle detector.

The whole "train on the fly" trick depends on a FROZEN backbone that turns an
image tile into a dense grid of feature vectors. We compute that grid once per
tile, then train a tiny linear head on doodled labels in feature space. Nothing
here is fine-tuned at interaction time.

Three implementations are provided behind one interface:

  * DinoV2Extractor  - real ViT features via torch.hub (good default for a POC).
  * MaeExtractor     - hook for your in-house S2/DS-EO MAE ViT checkpoint.
  * MockExtractor    - numpy-only, no weights, so the app runs anywhere and the
                       full doodle -> train -> detect loop is demoable/testable.

All extractors return features as float32 [Hf, Wf, C], already L2-normalized so
the downstream linear head behaves like a cosine-similarity classifier. That
normalization is what lets a head trained on one 1 km2 tile transfer to the
other tiles of the scene.
"""

from __future__ import annotations

import numpy as np


def l2_normalize(feats: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """L2-normalize the channel dimension of an [Hf, Wf, C] feature grid."""
    norm = np.linalg.norm(feats, axis=-1, keepdims=True)
    return (feats / (norm + eps)).astype(np.float32)


class FeatureExtractor:
    """Interface. Implementations turn a HxWx3 uint8 tile into an [Hf,Wf,C] grid."""

    #: ground pixels per feature cell along one axis (the effective feature stride)
    patch_stride: int = 14
    name: str = "base"

    def embed(self, tile_rgb: np.ndarray) -> np.ndarray:
        """tile_rgb: [H, W, 3] uint8 -> features [Hf, Wf, C] float32, L2-normed."""
        raise NotImplementedError

    def embed_dense(self, tile_rgb: np.ndarray, density: int = 1) -> np.ndarray:
        """
        Feature-density dial (Option A: overlapping windows, no model change).

        density=1  -> exactly self.embed(): cells are patch_stride px apart.
        density=d  -> run self.embed() at d x d sub-patch offsets and interleave
                      the grids into one that is d x finer (effective cell size
                      = patch_stride / d). The backbone, its weights and patch
                      size are untouched; we just sample it at more offsets, so
                      a sub-patch object (e.g. a truck ~1 cell) becomes a d x d
                      cluster the head can localise. Cost ~ d^2 forward passes.

        Works for any extractor because it only calls self.embed(); per-cell
        vectors (normalised or not) are preserved by the interleave.
        """
        density = max(1, int(density))
        if density == 1:
            return self.embed(tile_rgb)
        p = self.patch_stride
        offsets = [int(round(p * k / density)) for k in range(density)]
        grids: dict = {}
        for iy, oy in enumerate(offsets):
            for ix, ox in enumerate(offsets):
                grids[(iy, ix)] = self.embed(tile_rgb[oy:, ox:, :])
        Hf = min(g.shape[0] for g in grids.values())
        Wf = min(g.shape[1] for g in grids.values())
        C = next(iter(grids.values())).shape[2]
        merged = np.zeros((Hf * density, Wf * density, C), dtype=np.float32)
        for (iy, ix), g in grids.items():
            merged[iy::density, ix::density, :] = g[:Hf, :Wf, :]
        return merged

    def feature_grid_shape(self, h: int, w: int) -> tuple[int, int]:
        return h // self.patch_stride, w // self.patch_stride


# --------------------------------------------------------------------------- #
# Mock backbone (no torch, no weights) - for testing and offline demos.
# --------------------------------------------------------------------------- #
class MockExtractor(FeatureExtractor):
    """
    Deterministic hand-crafted features so the pipeline is fully exercisable
    without downloading model weights. Produces a small feature vector per patch
    from simple statistics (mean color, local contrast, edge energy) plus a few
    fixed random projections. It is NOT a good detector - it exists so the app,
    the training loop, and the UI can be run and tested end to end.
    """

    def __init__(self, patch_stride: int = 14, **_ignored):
        self.patch_stride = patch_stride
        self.name = "mock"

    def embed(self, tile_rgb: np.ndarray) -> np.ndarray:
        # NOTE: unlike the real backbones this does NOT L2-normalize. The mock's
        # discriminative signal lives in feature *magnitude* (brightness /
        # whiteness / saturation), so we keep raw magnitudes. Real ViT features
        # carry their signal in *direction*, which is why they are normalized
        # (that is what makes their head transfer across tiles).
        h, w = tile_rgb.shape[:2]
        s = self.patch_stride
        Hf, Wf = h // s, w // s
        img = tile_rgb[: Hf * s, : Wf * s].astype(np.float32) / 255.0
        patches = img.reshape(Hf, s, Wf, s, 3).transpose(0, 2, 1, 3, 4)
        flat = patches.reshape(Hf, Wf, s * s, 3)      # [Hf,Wf,s*s,3]
        mean = flat.mean(axis=2)                       # per-channel mean [Hf,Wf,3]
        mn = flat.min(axis=3).mean(axis=2)             # whiteness (min channel)
        mx = flat.max(axis=3).mean(axis=2)             # brightest channel
        bright = mean.mean(axis=2)                     # overall brightness
        sat = mx - mn                                  # saturation
        contrast = flat.mean(axis=3).std(axis=2)       # local contrast
        feats = np.stack(
            [mean[..., 0], mean[..., 1], mean[..., 2],
             mn, bright, sat, contrast], axis=-1
        ).astype(np.float32)                            # [Hf,Wf,7]
        return feats


# --------------------------------------------------------------------------- #
# DINOv2 backbone (real ViT features). Lazy torch import so the module loads
# even in a torch-free environment.
# --------------------------------------------------------------------------- #
class DinoV2Extractor(FeatureExtractor):
    """
    Dense DINOv2 patch features. The backbone is frozen; we slide a fixed-size
    window over the tile and stitch the per-window patch grids into one dense
    feature grid for the whole tile.

    variant: one of dinov2_vits14 / vitb14 / vitl14 / vitg14.
    window:  side length in pixels of each forward pass (multiple of 14).
    """

    def __init__(self, variant: str = "dinov2_vits14", window: int = 518,
                 device: str | None = None):
        import torch  # lazy

        assert window % 14 == 0, "DINOv2 window must be a multiple of 14"
        self.patch_stride = 14
        self.window = window
        self.name = variant
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        self.model = torch.hub.load("facebookresearch/dinov2", variant)
        self.model.eval().to(self.device)
        # ImageNet normalization
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @property
    def embed_dim(self) -> int:
        return self.model.embed_dim

    def _window_features(self, batch: "np.ndarray") -> np.ndarray:
        """batch: [N, win, win, 3] uint8 -> [N, win//14, win//14, C] float32."""
        torch = self._torch
        with torch.no_grad():
            x = torch.from_numpy(batch).to(self.device).float().permute(0, 3, 1, 2) / 255.0
            x = (x - self._mean) / self._std
            out = self.model.forward_features(x)
            tok = out["x_norm_patchtokens"]           # [N, win/14 * win/14, C]
            g = self.window // 14
            tok = tok.reshape(tok.shape[0], g, g, tok.shape[-1])
            return tok.float().cpu().numpy()

    def embed(self, tile_rgb: np.ndarray) -> np.ndarray:
        h, w = tile_rgb.shape[:2]
        win, g = self.window, self.window // 14
        Hf, Wf = (h // 14), (w // 14)
        C = self.embed_dim
        feats = np.zeros((Hf, Wf, C), dtype=np.float32)
        # tile the image into non-overlapping windows (pad the last row/col)
        ys = list(range(0, h, win))
        xs = list(range(0, w, win))
        for y in ys:
            for x in xs:
                crop = tile_rgb[y:y + win, x:x + win]
                ph, pw = crop.shape[:2]
                if ph != win or pw != win:
                    pad = np.zeros((win, win, 3), dtype=tile_rgb.dtype)
                    pad[:ph, :pw] = crop
                    crop = pad
                wf = self._window_features(crop[None])[0]        # [g,g,C]
                fy, fx = y // 14, x // 14
                fh = min(g, Hf - fy)
                fw = min(g, Wf - fx)
                feats[fy:fy + fh, fx:fx + fw] = wf[:fh, :fw]
        return l2_normalize(feats)


# --------------------------------------------------------------------------- #
# MAE backbone hook (your in-house S2/DS-EO MAE ViT).
# --------------------------------------------------------------------------- #
class MaeExtractor(FeatureExtractor):
    """
    Skeleton for plugging in your own MAE ViT checkpoint (the backbone the
    real doodling tool uses). Fill in `_load` and `_window_features` for your
    checkpoint's API. The tiling/stitching logic mirrors DinoV2Extractor.

    NOTE from the handoff: that MAE is likely GSD-mismatched for sub-meter
    imagery, so treat results as triage-quality until a sub-meter backbone is
    swapped in behind this same interface.
    """

    def __init__(self, checkpoint_path: str, window: int = 224,
                 patch_stride: int = 16, device: str | None = None):
        import torch  # lazy
        self._torch = torch
        self.patch_stride = patch_stride
        self.window = window
        self.name = "mae"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load(checkpoint_path)

    def _load(self, checkpoint_path: str):
        raise NotImplementedError(
            "Wire this to your MAE. Typical steps: build the ViT, "
            "load_state_dict from the checkpoint, model.eval().to(device). "
            "Return an object whose forward gives patch tokens."
        )

    def _window_features(self, batch: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Return [N, window//patch_stride, window//patch_stride, C] float32 "
            "for a [N, window, window, 3] uint8 batch, matching your MAE's "
            "preprocessing (its own mean/std, band order, etc.)."
        )

    embed = DinoV2Extractor.embed  # reuse the identical tiling/stitching loop


def build_extractor(kind: str = "mock", **kw) -> FeatureExtractor:
    kind = kind.lower()
    if kind == "mock":
        return MockExtractor(**kw)
    if kind in ("dino", "dinov2"):
        return DinoV2Extractor(**kw)
    if kind == "mae":
        return MaeExtractor(**kw)
    raise ValueError(f"unknown backbone kind: {kind}")
