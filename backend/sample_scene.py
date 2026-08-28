"""
Synthetic scene generator so the full doodle -> train -> detect -> scale loop
is demoable without a real GeoTIFF.

Produces a terrain-ish background with two kinds of scattered rectangles:
  * "target" objects  - bright metallic vehicles (what you'll doodle as +)
  * "distractor"      - dark/vegetation blobs and buildings (doodle as -)

It is deliberately easy: the mock backbone can separate the classes, so you can
verify the plumbing end to end. Swap in a real scene + real backbone for
anything meaningful.
"""

from __future__ import annotations

import numpy as np


def make_demo_scene(width: int = 4000, height: int = 4000,
                    n_targets: int = 220, n_distractors: int = 400,
                    seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # base terrain: low-frequency noise blended greens/browns
    small = rng.random((height // 40, width // 40, 3))
    ys = (np.arange(height) * small.shape[0] // height).clip(0, small.shape[0] - 1)
    xs = (np.arange(width) * small.shape[1] // width).clip(0, small.shape[1] - 1)
    terrain = small[ys][:, xs]
    tint = np.array([0.35, 0.42, 0.28])  # olive
    img = (terrain * 0.4 + tint) * 255
    img += rng.normal(0, 6, img.shape)   # fine grain
    img = np.clip(img, 0, 255).astype(np.uint8)

    def stamp(cx, cy, w, h, color, jitter=10):
        y0, y1 = max(cy - h // 2, 0), min(cy + h // 2, height)
        x0, x1 = max(cx - w // 2, 0), min(cx + w // 2, width)
        patch = np.array(color) + rng.normal(0, jitter, 3)
        img[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)

    targets = []
    for _ in range(n_targets):
        cx, cy = rng.integers(20, width - 20), rng.integers(20, height - 20)
        w, h = rng.integers(14, 26), rng.integers(10, 20)
        stamp(cx, cy, w, h, (205, 205, 195))  # bright metallic
        targets.append((cx, cy))

    for _ in range(n_distractors):
        cx, cy = rng.integers(20, width - 20), rng.integers(20, height - 20)
        w, h = rng.integers(18, 60), rng.integers(18, 60)
        if rng.random() < 0.5:
            stamp(cx, cy, w, h, (60, 70, 50))    # dark vegetation
        else:
            stamp(cx, cy, w, h, (120, 110, 100))  # building/road

    return img


if __name__ == "__main__":
    from PIL import Image
    img = make_demo_scene()
    Image.fromarray(img).save("demo_scene.png")
    print("wrote demo_scene.png", img.shape)
