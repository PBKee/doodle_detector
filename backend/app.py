"""
FastAPI backend for the doodle detector.

Flow the frontend drives:
  1. GET  /api/scene                 -> scene meta + overview PNG + tile grid
  2. GET  /api/tile/{id}.png         -> the RGB pixels of a work tile
  3. POST /api/embed  {tile_id}      -> run frozen backbone, cache feature grid
  4. POST /api/train  {tile_id,doodles,params}
                                     -> fit the linear head; return heatmap+boxes
  5. POST /api/infer_tile {tile_id}  -> re-run current head on a tile
  6. POST /api/infer_scene {params}  -> embed+classify every tile, stream progress

Run:  python -m backend.app --backbone mock --demo
      python -m backend.app --backbone dinov2 --image /path/to/scene.tif
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from typing import Optional

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

from . import backbones, head as head_mod, scene as scene_mod

# --------------------------------------------------------------------------- #
# Global app state (single-user POC).
#
# A "detector" = one object class (its own linear head + accumulated samples +
# scorecard + colour). Several can live in memory at once; the ACTIVE one is the
# one your doodles train. You can run any subset of them together.
# --------------------------------------------------------------------------- #
DETECTOR_COLORS = ["#f5a623", "#38bdf8", "#e879f9", "#4ade80",
                   "#fb7185", "#a78bfa", "#22d3ee", "#f97316"]


class Detector:
    def __init__(self, name: str, color: str):
        self.name = name
        self.color = color
        self.head = head_mod.DoodleHead()
        self.samples: dict = {}     # "imageId::tileId" -> {pos,neg,conf} arrays
        self.scores: dict = {}      # "imageId::tileId" -> {tp,fp,fn}
        self.conf_weight: float = 3.0


class State:
    scene: Optional[scene_mod.Scene] = None
    extractor: Optional[backbones.FeatureExtractor] = None
    cache: scene_mod.FeatureCache = scene_mod.FeatureCache(max_items=8)
    tiles_by_id: dict = {}
    image_id: str = ""               # id of the currently loaded image
    gsd: float = 0.3
    tile_km: float = 0.5
    images: dict = {}                # {id: path} loaded this session
    detectors: dict = {}             # {name: Detector}
    active: str = ""                 # name of the active detector
    density: int = 1                 # feature-density dial (1x = patch_stride)


STATE = State()
app = FastAPI(title="Doodle Detector")


def _new_detector(name: str) -> "Detector":
    name = (name or "untitled").strip() or "untitled"
    if name in STATE.detectors:
        return STATE.detectors[name]
    color = DETECTOR_COLORS[len(STATE.detectors) % len(DETECTOR_COLORS)]
    d = Detector(name, color)
    STATE.detectors[name] = d
    return d


def active_det() -> "Detector":
    if STATE.active not in STATE.detectors:
        _new_detector("untitled"); STATE.active = "untitled"
    return STATE.detectors[STATE.active]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def png_bytes(arr: np.ndarray) -> bytes:
    if arr.ndim == 2:
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
    elif arr.shape[2] == 4:
        img = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
    else:
        img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def heatmap_rgba(prob_grid: np.ndarray, out_h: int, out_w: int,
                 threshold: float) -> np.ndarray:
    """Build a translucent overlay: transparent below threshold, warm above."""
    up = head_mod.upsample_grid(prob_grid, out_h, out_w)
    rgba = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    # color ramp from amber (low-ish) to red (high)
    t = np.clip((up - threshold) / max(1 - threshold, 1e-6), 0, 1)
    rgba[..., 0] = 255
    rgba[..., 1] = (200 * (1 - t)).astype(np.uint8)
    rgba[..., 2] = 40
    rgba[..., 3] = np.where(up >= threshold, (90 + 150 * t).astype(np.uint8), 0)
    return rgba


def get_tile_spec(tile_id: str) -> scene_mod.TileSpec:
    return STATE.tiles_by_id[tile_id]


def eff_stride() -> float:
    """Effective feature-cell size in pixels (patch_stride shrunk by density)."""
    return STATE.extractor.patch_stride / max(1, STATE.density)


def embed_tile(tile_id: str) -> np.ndarray:
    key = f"{tile_id}@{STATE.density}"          # features differ per density level
    cached = STATE.cache.get(key)
    if cached is not None:
        return cached
    spec = get_tile_spec(tile_id)
    rgb = STATE.scene.read_tile(spec)
    feats = STATE.extractor.embed_dense(rgb, STATE.density)
    STATE.cache.put(key, feats)
    return feats


def sample_features_at(feats: np.ndarray, points: list[dict], stride: int) -> np.ndarray:
    """points: [{x, y, r?}] in tile-pixel coords -> feature vectors [N, C].

    Each point may carry a brush radius `r` (tile px). All feature cells whose
    centre falls within that radius are sampled, so a fatter brush and the shape
    of the stroke genuinely change which patch features are fed to the head --
    the doodle acts like hand-painting a region of feature space, not a single
    point. Cells are de-duplicated across overlapping brush stamps.
    """
    Hf, Wf, C = feats.shape
    cells: set = set()
    for p in points:
        cx, cy = float(p["x"]), float(p["y"])
        r = float(p.get("r", 0) or 0)
        if r <= stride * 0.5:                       # point-sized: one cell
            cells.add((int(cy / stride), int(cx / stride)))
        else:                                       # disc of cells under the brush
            rc = int(r / stride)
            fcx, fcy = int(cx / stride), int(cy / stride)
            for dy in range(-rc, rc + 1):
                for dx in range(-rc, rc + 1):
                    if dx * dx + dy * dy <= rc * rc + 1e-6:
                        cells.add((fcy + dy, fcx + dx))
    pts = [(fy, fx) for (fy, fx) in cells if 0 <= fy < Hf and 0 <= fx < Wf]
    if not pts:
        return np.zeros((0, C), dtype=np.float32)
    return np.stack([feats[fy, fx] for fy, fx in pts]).astype(np.float32)


def _accumulated_samples(det: "Detector") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate every tile's / image's stored samples into one training set."""
    P, N, Cf = [], [], []
    for v in det.samples.values():
        if len(v["pos"]):  P.append(v["pos"])
        if len(v["neg"]):  N.append(v["neg"])
        if len(v["conf"]): Cf.append(v["conf"])
    dim = None
    for lst in (P, N, Cf):
        if lst:
            dim = lst[0].shape[1]; break
    cat = lambda lst: (np.concatenate(lst, 0).astype(np.float32)
                       if lst else np.zeros((0, dim or 1), np.float32))
    return cat(P), cat(N), cat(Cf)


def _refit(det: "Detector") -> bool:
    """Refit a detector's linear head on its whole accumulated set. Cheap (ms)."""
    pos, neg, conf = _accumulated_samples(det)
    if len(pos) == 0 or (len(neg) + len(conf)) == 0:
        return False
    det.head.fit(pos, neg, conf=conf, conf_weight=det.conf_weight)
    return True


def _totals(det: "Detector") -> dict:
    tp = tn = tc = 0
    imgs = set()
    tiles = set()
    detail: dict = {}
    for k, v in det.samples.items():
        parts = k.split("::")
        img = parts[0]
        tile = parts[1] if len(parts) > 1 else ""
        review = len(parts) > 2 and parts[2] == "rev"
        imgs.add(img)
        base = f"{img}::{tile}"
        tiles.add(base)
        d = detail.setdefault(base, {"image": img, "tile": tile,
                                     "pos": 0, "neg": 0, "conf": 0, "review": False})
        d["pos"] += len(v["pos"]); d["neg"] += len(v["neg"]); d["conf"] += len(v["conf"])
        d["review"] = d["review"] or review
        tp += len(v["pos"]); tn += len(v["neg"]); tc += len(v["conf"])
    return {"name": det.name, "color": det.color, "active": det.name == STATE.active,
            "trained": det.head.is_trained,
            "n_pos": tp, "n_neg": tn, "n_conf": tc,
            "n_tiles": len(tiles), "n_images": len(imgs),
            "images": sorted(imgs), "scores": _score_summary(det),
            "tiles_detail": sorted(detail.values(), key=lambda r: (r["image"], r["tile"])),
            "feat_dim": det.head.feat_dim, "backbone": STATE.extractor.name}


def _score_summary(det: "Detector") -> list[dict]:
    """Per-image TP/FP/FN + precision/recall/F1 for one detector.

    Two marking scopes share the scorecard: per-tile review (key image::tileId)
    and whole-image review (key image::__full__). If a whole-image review exists
    for an image, it is authoritative for that image (per-tile marks are ignored
    there) so the two don't double-count.
    """
    full: dict = {}
    tiles: dict = {}
    for k, v in det.scores.items():
        img, tile = k.split("::", 1)
        bucket = full if tile == "__full__" else tiles
        d = bucket.setdefault(img, {"tp": 0, "fp": 0, "fn": 0})
        d["tp"] += v["tp"]; d["fp"] += v["fp"]; d["fn"] += v["fn"]
    rows = []
    for img in sorted(set(full) | set(tiles)):
        d = full.get(img) or tiles[img]
        tp, fp, fn = d["tp"], d["fp"], d["fn"]
        prec = tp / (tp + fp) if (tp + fp) else None
        rec = tp / (tp + fn) if (tp + fn) else None
        f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
        rows.append({"image": img, "scope": "full" if img in full else "tiles",
                     "tp": tp, "fp": fp, "fn": fn,
                     "precision": round(prec, 3) if prec is not None else None,
                     "recall": round(rec, 3) if rec is not None else None,
                     "f1": round(f1, 3) if f1 is not None else None})
    return rows


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class EmbedReq(BaseModel):
    tile_id: str


class DetectParams(BaseModel):
    threshold: float = 0.5
    min_cells: int = 1
    max_cells: Optional[int] = None
    pad_px: int = 0
    merge_cells: int = 0       # dilate mask by N cells to merge object parts


class TrainReq(BaseModel):
    tile_id: str
    positives: list[dict]          # [{x,y}] tile-pixel coords
    negatives: list[dict]          # generic background
    confusers: list[dict] = []     # look-alikes: negatives, weighted heavier
    conf_weight: float = 3.0
    params: DetectParams = DetectParams()


class InferReq(BaseModel):
    tile_id: str
    params: DetectParams = DetectParams()


class SceneInferReq(BaseModel):
    params: DetectParams = DetectParams()


class NameReq(BaseModel):
    name: str


class LoadImageReq(BaseModel):
    path: str
    gsd: Optional[float] = None
    tile_km: Optional[float] = None


class ScoreReq(BaseModel):
    tile_id: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    detector: Optional[str] = None   # default: the active detector


class MultiInferReq(BaseModel):
    tile_id: str
    names: list[str] = []            # detectors to run (empty = all trained)
    params: DetectParams = DetectParams()


class MultiSceneReq(BaseModel):
    names: list[str] = []
    params: DetectParams = DetectParams()


class ReviewRetrainReq(BaseModel):
    # all coords are SCENE pixels; boxes carry the detector they belong to
    fps: list[dict] = []   # {x,y,w,h,detector}  false positives  -> confuser
    tps: list[dict] = []   # {x,y,w,h,detector}  confirmed hits    -> positive (reinforce)
    fns: list[dict] = []   # {x,y}               misses            -> positive (active det)
    active: Optional[str] = None


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/scene")
def api_scene():
    s = STATE.scene
    if s is None:
        return {"ready": False, "backbone": STATE.extractor.name,
                "gsd_m": STATE.gsd, "density": STATE.density, "tiles": []}
    overview = s.overview(max_side=2048)
    tiles = s.tiles()
    STATE.tiles_by_id = {t.id: t for t in tiles}
    return {
        "ready": True,
        "width": s.width,
        "height": s.height,
        "gsd_m": s.gsd_m,
        "tile_px": s.tile_px,
        "tile_km2": round((s.tile_px * s.gsd_m / 1000.0) ** 2, 3),
        "backbone": STATE.extractor.name,
        "stride": eff_stride(),
        "base_stride": STATE.extractor.patch_stride,
        "density": STATE.density,
        "overview_w": overview.shape[1],
        "overview_h": overview.shape[0],
        "overview_png": "data:image/png;base64,"
                        + base64.b64encode(png_bytes(overview)).decode(),
        "tiles": [{"id": t.id, **t.__dict__} for t in tiles],
    }


@app.get("/api/tile/{tile_id}.png")
def api_tile(tile_id: str):
    spec = get_tile_spec(tile_id)
    rgb = STATE.scene.read_tile(spec)
    return Response(content=png_bytes(rgb), media_type="image/png")


@app.post("/api/embed")
def api_embed(req: EmbedReq):
    feats = embed_tile(req.tile_id)
    return {"tile_id": req.tile_id, "grid_h": feats.shape[0],
            "grid_w": feats.shape[1], "dim": feats.shape[2]}


class DensityReq(BaseModel):
    density: int = 1


@app.post("/api/set_density")
def api_set_density(req: DensityReq):
    """Set the feature-density level (1..3). Re-embeds happen on next use;
    features are cached per level so switching back is instant."""
    STATE.density = max(1, min(3, int(req.density)))
    return {"ok": True, "density": STATE.density,
            "base_stride": STATE.extractor.patch_stride,
            "cell_px": eff_stride(),
            "cell_m": round(eff_stride() * STATE.gsd, 2)}


@app.post("/api/train")
def api_train(req: TrainReq):
    det = active_det()
    feats = embed_tile(req.tile_id)
    stride = eff_stride()
    pos = sample_features_at(feats, req.positives, stride)
    neg = sample_features_at(feats, req.negatives, stride)
    conf = sample_features_at(feats, req.confusers, stride)
    if len(pos) + len(neg) + len(conf) == 0:
        return {"ok": False, "error": "Add some doodles on this tile first."}

    # Store (overwrite) THIS tile's contribution to the ACTIVE detector, keyed by
    # image+tile, then refit on everything it has seen across tiles and images.
    det.conf_weight = req.conf_weight
    key = f"{STATE.image_id}::{req.tile_id}"
    det.samples[key] = {"pos": pos, "neg": neg, "conf": conf}

    if not _refit(det):
        del det.samples[key]
        return {"ok": False,
                "error": "This detector needs at least one target and one "
                         "background or confuser sample in total."}

    extra = {"ok": True}
    extra.update(_totals(det))
    return _infer_response(det, req.tile_id, req.params, extra=extra)


@app.post("/api/infer_tile")
def api_infer_tile(req: InferReq):
    det = active_det()
    if not det.head.is_trained:
        return {"ok": False, "error": "Train this detector first."}
    return _infer_response(det, req.tile_id, req.params, extra={"ok": True})


def _boxes_for(det: "Detector", tile_id: str, params: DetectParams):
    feats = embed_tile(tile_id)
    prob = det.head.predict_grid(feats)
    stride = eff_stride()
    return prob, head_mod.heatmap_to_boxes(
        prob, stride, threshold=params.threshold, min_cells=params.min_cells,
        max_cells=params.max_cells, pad_px=params.pad_px,
        merge_cells=params.merge_cells)


def _infer_response(det: "Detector", tile_id: str, params: DetectParams, extra: dict) -> dict:
    spec = get_tile_spec(tile_id)
    prob, boxes = _boxes_for(det, tile_id, params)
    overlay = heatmap_rgba(prob, spec.h, spec.w, params.threshold)
    out = {
        "tile_id": tile_id,
        "boxes": [b.as_dict() for b in boxes],
        "n_boxes": len(boxes),
        "color": det.color,
        "heatmap_png": "data:image/png;base64,"
                       + base64.b64encode(png_bytes(overlay)).decode(),
    }
    out.update(extra)
    return out


def _selected(names: list[str]) -> list["Detector"]:
    if names:
        dets = [STATE.detectors[n] for n in names if n in STATE.detectors]
    else:
        dets = list(STATE.detectors.values())
    return [d for d in dets if d.head.is_trained]


@app.post("/api/infer_tile_multi")
def api_infer_tile_multi(req: MultiInferReq):
    """Run several detectors on one tile; return colour-tagged boxes per class."""
    dets = _selected(req.names)
    if not dets:
        return {"ok": False, "error": "None of the chosen detectors are trained yet."}
    results = []
    for det in dets:
        _, boxes = _boxes_for(det, req.tile_id, req.params)
        results.append({"name": det.name, "color": det.color,
                        "n_boxes": len(boxes), "boxes": [b.as_dict() for b in boxes]})
    return {"ok": True, "tile_id": req.tile_id, "results": results}


@app.post("/api/infer_scene")
def api_infer_scene(req: SceneInferReq):
    """Run the ACTIVE detector over every tile, streaming progress + detections."""
    det = active_det()
    if not det.head.is_trained:
        return {"ok": False, "error": "Train this detector first."}
    return _scene_stream([det], req.params)


@app.post("/api/infer_scene_multi")
def api_infer_scene_multi(req: MultiSceneReq):
    """Run several detectors over every tile; detections tagged by class."""
    dets = _selected(req.names)
    if not dets:
        return {"ok": False, "error": "None of the chosen detectors are trained yet."}
    return _scene_stream(dets, req.params)


def _scene_stream(dets: list["Detector"], params: DetectParams) -> StreamingResponse:
    tiles = STATE.scene.tiles()

    def gen():
        total = len(tiles)
        all_dets = []
        for i, spec in enumerate(tiles):
            new_here = 0
            for det in dets:
                _, boxes = _boxes_for(det, spec.id, params)
                for b in boxes:
                    gx, gy = int(spec.x + b.x), int(spec.y + b.y)
                    rec = {"x": gx, "y": gy, "w": int(b.w), "h": int(b.h),
                           "score": round(float(b.score), 4), "tile": spec.id,
                           "detector": det.name, "color": det.color}
                    ll = STATE.scene.pixel_to_lonlat(gx + b.w / 2, gy + b.h / 2)
                    if ll:
                        rec["lon"], rec["lat"] = ll
                    all_dets.append(rec)
                    new_here += 1
            yield json.dumps({"type": "progress", "done": i + 1, "total": total,
                              "tile": spec.id, "new_boxes": new_here}) + "\n"
        yield json.dumps({"type": "done", "total_boxes": len(all_dets),
                          "detections": all_dets}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/reset_head")
def api_reset_head():
    """Clear the active detector's fitted head but keep its samples."""
    det = active_det()
    det.head = head_mod.DoodleHead()
    _refit(det)
    return {"ok": True, **_totals(det)}


# --------------------------------------------------------------------------- #
# Detector: accumulate / inspect / save / load  (a detector = one object class)
# --------------------------------------------------------------------------- #
import os
import pickle

DETECTORS_DIR = os.path.join(os.path.dirname(__file__), "..", "detectors")


@app.get("/api/detector/state")
def api_detector_state():
    return _totals(active_det())


@app.get("/api/detectors")
def api_detectors():
    """All detectors currently in memory, plus which one is active."""
    return {"active": STATE.active,
            "detectors": [_totals(d) for d in STATE.detectors.values()]}


@app.post("/api/detector/new")
def api_detector_new(req: NameReq):
    """Create a new, empty detector for a new object class, and make it active."""
    d = _new_detector(req.name)
    STATE.active = d.name
    return {"ok": True, "active": STATE.active, **_totals(d)}


@app.post("/api/detector/activate")
def api_detector_activate(req: NameReq):
    if req.name not in STATE.detectors:
        return {"ok": False, "error": f"No detector named '{req.name}'."}
    STATE.active = req.name
    return {"ok": True, "active": STATE.active, **_totals(active_det())}


@app.post("/api/detector/delete")
def api_detector_delete(req: NameReq):
    STATE.detectors.pop(req.name, None)
    if not STATE.detectors:
        _new_detector("untitled")
    if STATE.active not in STATE.detectors:
        STATE.active = next(iter(STATE.detectors))
    return {"ok": True, "active": STATE.active}


@app.get("/api/detector/list")
def api_detector_list():
    os.makedirs(DETECTORS_DIR, exist_ok=True)
    names = [f[:-4] for f in os.listdir(DETECTORS_DIR) if f.endswith(".pkl")]
    return {"detectors": sorted(names)}


@app.post("/api/detector/save")
def api_detector_save(req: NameReq):
    os.makedirs(DETECTORS_DIR, exist_ok=True)
    det = STATE.detectors.get(req.name.strip()) or active_det()
    name = req.name.strip() or det.name
    payload = {
        "name": name, "color": det.color,
        "backbone": STATE.extractor.name, "feat_dim": det.head.feat_dim,
        "conf_weight": det.conf_weight,
        "samples": det.samples, "scores": det.scores,
    }
    with open(os.path.join(DETECTORS_DIR, f"{name}.pkl"), "wb") as f:
        pickle.dump(payload, f)
    return {"ok": True, "saved": name, **_totals(det)}


@app.post("/api/detector/load")
def api_detector_load(req: NameReq):
    path = os.path.join(DETECTORS_DIR, f"{req.name.strip()}.pkl")
    if not os.path.exists(path):
        return {"ok": False, "error": f"No saved detector named '{req.name}'."}
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if payload.get("backbone") != STATE.extractor.name:
        return {"ok": False,
                "error": f"Detector was trained with backbone "
                         f"'{payload.get('backbone')}', but this server runs "
                         f"'{STATE.extractor.name}'. Restart with the matching backbone."}
    name = payload.get("name", req.name)
    d = _new_detector(name)
    d.color = payload.get("color", d.color)
    d.samples = payload.get("samples", {})
    d.scores = payload.get("scores", {})
    d.conf_weight = payload.get("conf_weight", 3.0)
    d.head = head_mod.DoodleHead()
    _refit(d)
    STATE.active = name
    return {"ok": True, "loaded": name, "active": STATE.active, **_totals(d)}


# --------------------------------------------------------------------------- #
# Load a different image at runtime (keeps the detector; tests generalisation)
# --------------------------------------------------------------------------- #
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")


@app.post("/api/load_image")
def api_load_image(req: LoadImageReq):
    path = req.path.strip().strip('"')
    if not os.path.exists(path):
        return {"ok": False, "error": f"Path not found: {path}"}
    gsd = req.gsd if req.gsd is not None else STATE.gsd
    tile_km = req.tile_km if req.tile_km is not None else STATE.tile_km
    try:
        _set_scene_from_path(path, gsd, tile_km)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"Could not open image: {e}"}
    return {"ok": True, "image_id": STATE.image_id,
            "width": STATE.scene.width, "height": STATE.scene.height}


@app.post("/api/upload_image")
async def api_upload_image(request: Request, name: str = "upload.png",
                           gsd: Optional[float] = None,
                           tile_km: Optional[float] = None):
    """Receive raw image bytes in the request body, save, and load as scene."""
    data = await request.body()
    if not data:
        return {"ok": False, "error": "Empty upload."}
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe = os.path.basename(name) or "upload.png"
    path = os.path.join(UPLOAD_DIR, safe)
    with open(path, "wb") as f:
        f.write(data)
    g = gsd if gsd is not None else STATE.gsd
    tk = tile_km if tile_km is not None else STATE.tile_km
    try:
        _set_scene_from_path(path, g, tk)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"Could not open image: {e}"}
    return {"ok": True, "image_id": STATE.image_id,
            "width": STATE.scene.width, "height": STATE.scene.height}


@app.get("/api/images")
def api_images():
    """Images loaded this session, so the UI can switch between them."""
    ids = set(STATE.images)
    if STATE.image_id:
        ids.add(STATE.image_id)
    return {"current": STATE.image_id,
            "images": [{"id": i} for i in sorted(ids)]}


@app.post("/api/select_image")
def api_select_image(req: NameReq):
    """Switch to an already-loaded image by id (no re-upload / no restart)."""
    path = STATE.images.get(req.name)
    if not path or not os.path.exists(path):
        return {"ok": False, "error": f"Image '{req.name}' is not loaded."}
    try:
        _set_scene_from_path(path, STATE.gsd, STATE.tile_km)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"Could not open image: {e}"}
    return {"ok": True, "image_id": STATE.image_id,
            "width": STATE.scene.width, "height": STATE.scene.height}


# --------------------------------------------------------------------------- #
# Analyst scoring: mark predictions per tile; aggregate per image
# --------------------------------------------------------------------------- #
@app.post("/api/score")
def api_score(req: ScoreReq):
    det = STATE.detectors.get(req.detector) if req.detector else active_det()
    if det is None:
        det = active_det()
    key = f"{STATE.image_id}::{req.tile_id}"
    if req.tp == 0 and req.fp == 0 and req.fn == 0:
        det.scores.pop(key, None)
    else:
        det.scores[key] = {"tp": req.tp, "fp": req.fp, "fn": req.fn}
    return {"ok": True, "scores": _score_summary(active_det())}


@app.post("/api/reset_scores")
def api_reset_scores():
    det = active_det()
    det.scores = {}
    return {"ok": True, "scores": _score_summary(det)}


@app.post("/api/retrain_from_review")
def api_retrain_from_review(req: ReviewRetrainReq):
    """Turn whole-image review marks into training samples and refit.

    false positive -> confuser for that box's detector
    confirmed hit  -> positive  (reinforce) for that box's detector
    miss           -> positive for the active detector, sampled over a small
                      neighbourhood (not just one patch) so the object is covered

    Review-derived samples are stored under a separate '::rev' key per tile so a
    later re-doodle of the same tile doesn't wipe them.
    """
    active = req.active or STATE.active
    stride = eff_stride()
    specs = list(STATE.tiles_by_id.values())
    touched: set = set()

    def tile_of(px, py):
        for s in specs:
            if s.x <= px < s.x + s.w and s.y <= py < s.y + s.h:
                return s
        return None

    def add(detname, spec, local_pts, kind):
        det = STATE.detectors.get(detname)
        if det is None:
            return
        feats = embed_tile(spec.id)
        F = sample_features_at(feats, local_pts, stride)
        if len(F) == 0:
            return
        key = f"{STATE.image_id}::{spec.id}::rev"
        rec = det.samples.get(key)
        if rec is None:
            z = np.zeros((0, F.shape[1]), np.float32)
            rec = {"pos": z, "neg": z, "conf": z}
        rec[kind] = np.concatenate([rec[kind], F], 0) if len(rec[kind]) else F
        det.samples[key] = rec
        touched.add(detname)

    for b in req.fps + req.tps:
        cx, cy = b["x"] + b.get("w", 0) / 2, b["y"] + b.get("h", 0) / 2
        spec = tile_of(cx, cy)
        if spec:
            kind = "conf" if b in req.fps else "pos"
            add(b.get("detector", active), spec,
                [{"x": cx - spec.x, "y": cy - spec.y}], kind)

    for p in req.fns:
        spec = tile_of(p["x"], p["y"])
        if not spec:
            continue
        lx, ly = p["x"] - spec.x, p["y"] - spec.y
        pts = [{"x": lx + dx, "y": ly + dy}
               for dx in (-stride, 0, stride) for dy in (-stride, 0, stride)]
        add(active, spec, pts, "pos")

    for name in touched:
        _refit(STATE.detectors[name])
    return {"ok": True, "touched": sorted(touched),
            "n_fp": len(req.fps), "n_tp": len(req.tps), "n_fn": len(req.fns),
            **_totals(active_det())}


# --------------------------------------------------------------------------- #
# Static frontend
# --------------------------------------------------------------------------- #
import os
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(FRONTEND_DIR, "index.html"), encoding="utf-8") as f:
        return f.read()


# --------------------------------------------------------------------------- #
# Startup / CLI
# --------------------------------------------------------------------------- #
def _set_scene_from_path(path: str, gsd: float, tile_km: float):
    """(Re)build STATE.scene from an image path; keep the detector intact."""
    if path.lower().endswith((".tif", ".tiff")):
        STATE.scene = scene_mod.RasterScene(path, gsd_m=gsd, tile_km=tile_km)
    else:
        STATE.scene = scene_mod.ArrayScene(
            np.array(Image.open(path).convert("RGB")), gsd_m=gsd, tile_km=tile_km)
    STATE.gsd, STATE.tile_km = gsd, tile_km
    STATE.image_id = os.path.basename(path)
    STATE.images[STATE.image_id] = path
    STATE.cache.clear()
    STATE.tiles_by_id = {t.id: t for t in STATE.scene.tiles()}


def build_state(args):
    STATE.density = max(1, min(3, int(getattr(args, "feature_density", 1))))
    if args.image:
        _set_scene_from_path(args.image, args.gsd, args.tile_km)
    elif args.demo:
        from .sample_scene import make_demo_scene
        img = make_demo_scene()
        STATE.scene = scene_mod.ArrayScene(img, gsd_m=args.gsd, tile_km=args.tile_km)
        STATE.gsd, STATE.tile_km = args.gsd, args.tile_km
        STATE.image_id = "demo_scene"
        STATE.tiles_by_id = {t.id: t for t in STATE.scene.tiles()}
    else:
        # Boot with no image; the user uploads/selects one from the web UI.
        STATE.scene = None
        STATE.image_id = ""
        STATE.gsd, STATE.tile_km = args.gsd, args.tile_km
        STATE.tiles_by_id = {}

    if args.backbone == "dinov2":
        STATE.extractor = backbones.build_extractor("dinov2", variant=args.dino_variant)
    elif args.backbone == "mae":
        STATE.extractor = backbones.build_extractor("mae", checkpoint_path=args.mae_ckpt)
    else:
        STATE.extractor = backbones.build_extractor("mock", patch_stride=args.stride)

    # start with one empty detector ready to train
    STATE.detectors = {}
    _new_detector("untitled")
    STATE.active = "untitled"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="mock", choices=["mock", "dinov2", "mae"])
    ap.add_argument("--image", default=None, help="GeoTIFF or regular image")
    ap.add_argument("--demo", action="store_true", help="use synthetic scene")
    ap.add_argument("--gsd", type=float, default=0.3, help="metres/pixel")
    ap.add_argument("--tile-km", type=float, default=0.5, help="work-tile side (km)")
    ap.add_argument("--stride", type=int, default=14, help="mock feature stride")
    ap.add_argument("--feature-density", type=int, default=1, choices=[1, 2, 3],
                    help="overlapping-window feature density (1x..3x); finer grid, ~d^2 cost")
    ap.add_argument("--dino-variant", default="dinov2_vits14")
    ap.add_argument("--mae-ckpt", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    build_state(args)
    import uvicorn
    if STATE.scene is not None:
        print(f"Scene: {STATE.scene.width}x{STATE.scene.height}px  "
              f"tile={STATE.scene.tile_px}px (~{args.tile_km} km)  "
              f"backbone={STATE.extractor.name}")
    else:
        print(f"No image loaded — open the web UI to upload or select one.  "
              f"backbone={STATE.extractor.name}")
    print(f"Open http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
