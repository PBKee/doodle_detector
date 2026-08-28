"""
The "trained on the fly" part.

A DoodleHead is a tiny linear classifier over frozen feature vectors. Because
the features are L2-normalized, a linear head is effectively a learned cosine
similarity, which is what lets it transfer from the 1 km2 work tile to the rest
of the scene. Fitting takes milliseconds, so retraining after every doodle
feels instant.

Also here: turning a per-cell probability grid into bounding boxes via
thresholding + connected components, with simple area/score filtering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    score: float

    def as_dict(self) -> dict:
        return {"x": int(self.x), "y": int(self.y),
                "w": int(self.w), "h": int(self.h),
                "score": round(float(self.score), 4)}


@dataclass
class DoodleHead:
    """Linear classifier over feature vectors. One head = one target concept."""

    clf: LogisticRegression | None = None
    n_pos: int = 0
    n_neg: int = 0
    n_conf: int = 0
    feat_dim: int | None = None
    meta: dict = field(default_factory=dict)

    @property
    def is_trained(self) -> bool:
        return self.clf is not None

    def fit(self, pos: np.ndarray, neg: np.ndarray,
            conf: np.ndarray | None = None, conf_weight: float = 3.0) -> None:
        """
        pos:  [Np, C] target feature vectors.
        neg:  [Nn, C] generic-background feature vectors.
        conf: [Nc, C] "confuser" feature vectors -- things that LOOK like the
              target but are not. These are trained as negatives, but each one
              carries `conf_weight`x the pull of a plain background sample, so
              the decision boundary is forced to run between the target and its
              look-alikes (where precision comes from) rather than just between
              the target and empty ground.

        Needs at least one positive and at least one negative-or-confuser.
        """
        conf = conf if conf is not None else np.zeros((0, pos.shape[1] if len(pos) else 0), np.float32)
        neg_total = len(neg) + len(conf)
        if len(pos) == 0 or neg_total == 0:
            raise ValueError("Need at least one positive and one negative/confuser sample")

        parts, ys, ws = [pos], [np.ones(len(pos))], [np.ones(len(pos))]
        if len(neg):
            parts.append(neg); ys.append(np.zeros(len(neg))); ws.append(np.ones(len(neg)))
        if len(conf):
            parts.append(conf); ys.append(np.zeros(len(conf)))
            ws.append(np.full(len(conf), float(conf_weight)))

        X = np.concatenate(parts, axis=0).astype(np.float32)
        y = np.concatenate(ys).astype(np.int64)
        sample_weight = np.concatenate(ws).astype(np.float64)

        # class_weight balances the usual pos<<neg imbalance; sample_weight then
        # multiplies on top so confusers weigh more. liblinear is fast and stable
        # for the small, wide problems we get from a few doodles.
        clf = LogisticRegression(
            class_weight="balanced", C=1.0, max_iter=1000, solver="liblinear"
        )
        clf.fit(X, y, sample_weight=sample_weight)
        self.clf = clf
        self.n_pos = int(len(pos))
        self.n_neg = int(len(neg))
        self.n_conf = int(len(conf))
        self.feat_dim = int(X.shape[1])

    def predict_grid(self, feats: np.ndarray) -> np.ndarray:
        """feats: [Hf, Wf, C] -> probability grid [Hf, Wf] in [0,1]."""
        if not self.is_trained:
            raise RuntimeError("Head is not trained yet")
        Hf, Wf, C = feats.shape
        flat = feats.reshape(-1, C)
        prob = self.clf.predict_proba(flat)[:, 1]
        return prob.reshape(Hf, Wf).astype(np.float32)


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label 8-connected components. Uses scipy if present, else a BFS fallback."""
    try:
        from scipy import ndimage
        structure = np.ones((3, 3), dtype=int)
        labels, n = ndimage.label(mask, structure=structure)
        return labels, n
    except Exception:
        labels = np.zeros(mask.shape, dtype=np.int32)
        n = 0
        H, W = mask.shape
        for i in range(H):
            for j in range(W):
                if mask[i, j] and labels[i, j] == 0:
                    n += 1
                    stack = [(i, j)]
                    labels[i, j] = n
                    while stack:
                        y, x = stack.pop()
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                ny, nx = y + dy, x + dx
                                if (0 <= ny < H and 0 <= nx < W
                                        and mask[ny, nx] and labels[ny, nx] == 0):
                                    labels[ny, nx] = n
                                    stack.append((ny, nx))
        return labels, n


def heatmap_to_boxes(
    prob_grid: np.ndarray,
    stride: int,
    threshold: float = 0.5,
    min_cells: int = 1,
    max_cells: int | None = None,
    pad_px: int = 0,
    merge_cells: int = 0,
) -> list[Detection]:
    """
    Convert a [Hf, Wf] probability grid into pixel-space boxes.

    stride:   ground pixels per feature cell (extractor.patch_stride).
    threshold: probability cut for "target".
    min_cells / max_cells: component-size filter in feature cells (kills specks
              and rejects huge blobs that are usually background bleed).
    pad_px:   grow each box by this many pixels on every side.
    merge_cells: dilate the mask by this many cells before grouping, so the
              separate high-probability patches of ONE object (wings, fuselage,
              tail) merge into a single box instead of several. Box extent and
              size filtering still use the ORIGINAL detected cells, so merging
              doesn't inflate scores or defeat the size filters.
    """
    mask = prob_grid >= threshold
    grow = mask
    if merge_cells and merge_cells > 0:
        try:
            from scipy import ndimage
            grow = ndimage.binary_dilation(mask, iterations=int(merge_cells))
        except Exception:
            grow = mask
    labels, n = _connected_components(grow)
    dets: list[Detection] = []
    for lab in range(1, n + 1):
        comp = labels == lab
        orig = comp & mask                     # real detected cells in this group
        ys, xs = np.where(orig if orig.any() else comp)
        size = int(len(xs))
        if size < min_cells:
            continue
        if max_cells is not None and size > max_cells:
            continue
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        score = float(prob_grid[ys, xs].mean())
        px = int(round(x0 * stride)) - pad_px
        py = int(round(y0 * stride)) - pad_px
        pw = int(round((x1 - x0) * stride)) + 2 * pad_px
        ph = int(round((y1 - y0) * stride)) + 2 * pad_px
        dets.append(Detection(max(px, 0), max(py, 0), pw, ph, score))
    dets.sort(key=lambda d: d.score, reverse=True)
    return dets


def upsample_grid(prob_grid: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest-neighbor upsample the [Hf,Wf] prob grid to pixel resolution."""
    Hf, Wf = prob_grid.shape
    ys = (np.arange(out_h) * Hf // max(out_h, 1)).clip(0, Hf - 1)
    xs = (np.arange(out_w) * Wf // max(out_w, 1)).clip(0, Wf - 1)
    return prob_grid[ys][:, xs]
