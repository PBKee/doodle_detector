# Doodle Detector

Interactive, doodle-driven object detection for satellite / aerial imagery.

Scribble a few examples of a target on a small tile, and a lightweight linear head trains
in milliseconds on top of a **frozen** vision backbone (DINOv2). The resulting detector is
applied to the whole image. Multiple object classes can be trained side by side, saved,
reloaded, and run together; performance can be scored by hand per tile or per whole image
to measure how well a detector generalises across images.

The backbone is never fine-tuned — all learning lives in a small logistic-regression head
over frozen patch features, which is why training feels instant and detectors are cheap to
accumulate, save, and combine.


## Features

- Frozen backbone (DINOv2 ViT-S/14) → dense patch features, cached per tile.
- Doodle brushes: **target**, **confuser** (weighted look-alikes), **background**.
- Millisecond training; live threshold / min-cell / max-cell / opacity controls.
- **Review mode**: mark predictions correct / false / missed; corrections fold back into
  training, and marks populate a precision / recall / F1 scorecard.
- **Multiple detectors** (one per object class), each with its own colour, samples, and
  scorecard. Run any subset together with colour-coded boxes.
- Accumulates learning across tiles **and** images.
- Save / load detectors to disk; upload or switch images from the browser (no restart).
- Full-image run with streaming progress, a crisp high-res overview, and JSON export
  (with lon/lat when the source is a GeoTIFF).


## Requirements

- Python 3.12
- Core: `fastapi`, `uvicorn`, `numpy`, `pillow`, `scikit-learn`, `scipy`
- Optional: `torch` (for the DINOv2 backbone), `rasterio` (for GeoTIFF input)

See `requirements.txt` in the project root.


## Installation

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt

# For the real backbone, install PyTorch. Choose ONE of the following:

# CPU only:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# NVIDIA GPU (CUDA) — pick the build matching your setup (cu126 = CUDA 12.6;
# other current options: cu124, cu118). The app auto-detects and uses the GPU.
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

> **GPU note:** you don't need to install the CUDA Toolkit separately — the pip wheel
> bundles its own CUDA runtime; you only need a reasonably recent NVIDIA driver. If a CPU
> build is already installed, run `pip uninstall torch` first. Verify with
> `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`.
> The official picker at <https://pytorch.org/get-started/locally/> confirms the exact
> command for your machine.

On Windows, if PowerShell blocks the activation script:
```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
```


## Usage

Run as a module from the project root:

```bash
python -m backend.app --backbone dinov2 --gsd 0.5 --tile-km 0.5
```

Then open <http://127.0.0.1:8000> (leave the server process running). The app boots with
no image loaded — use **Upload image…** in the browser to load one.

### Command-line options

| Flag             | Default | Description                                              |
|------------------|---------|----------------------------------------------------------|
| `--backbone`     | `mock`  | `mock` (no deps), `dinov2` (needs torch), `mae` (stub)   |
| `--gsd`          | `0.3`   | Ground sample distance, metres/pixel                     |
| `--tile-km`      | `0.5`   | Work-tile side length in km (0.5 km side = 0.25 km²)     |
| `--image PATH`   | –       | Seed a first image at startup (optional)                 |
| `--demo`         | –       | Use a synthetic demo scene (plumbing only)               |
| `--dino-variant` | `dinov2_vits14` | `dinov2_vits14/vitb14/vitl14/vitg14`             |
| `--mae-ckpt`     | –       | Checkpoint path for the in-house MAE backbone            |

With neither `--image` nor `--demo`, the server waits for a UI upload.

### Typical workflow

1. Upload an image and pick a work tile from the overview.
2. **Draw examples** — paint targets, confusers, and background; press **Train detector**.
3. **Review predictions** — mark false/missed detections; re-train to fold the corrections in.
4. Add more classes with **+ New** (e.g. `tank`, `artillery`); each trains independently.
5. Tick the classes under **Run classes** and **Run on full image** for colour-coded results.
6. Review the full image and read each class's precision / recall / F1 in the scorecard;
   switch images to test generalisation.
7. **Save active** to persist a detector; **Load…** to bring it back later.


## Project structure

```
doodle-detector/
├── backend/
│   ├── __init__.py
│   ├── app.py           # FastAPI app: endpoints, detector registry, scene/image handling
│   ├── backbones.py     # FeatureExtractor: DinoV2 / Mae (stub) / Mock; build_extractor()
│   ├── head.py          # DoodleHead (logistic regression + confuser weighting), box extraction
│   ├── scene.py         # ArrayScene / RasterScene (windowed GeoTIFF), tiling, overview, cache
│   └── sample_scene.py  # synthetic demo scene
├── frontend/
│   └── index.html       # entire single-file UI (vanilla JS + canvas, no CDNs)
├── detectors/           # saved detectors (*.pkl), created on first save
├── uploads/             # images uploaded via the UI, created on first upload
├── requirements.txt     # Python dependencies
├── run.sh
└── README.md
```


## Architecture notes

- **Frozen backbone + fast head.** Features are computed once per tile and cached; doodles
  become positive/negative feature samples; a `LogisticRegression` head is fit in
  milliseconds. Real-backbone features are L2-normalised, so the head behaves like learned
  cosine similarity and transfers across tiles and images.
- **Detectors are independent one-vs-rest classifiers** (not a single multinomial model),
  which is why they save, load, and run separately, and why two classes may both fire on
  the same object — a useful confusability signal.
- **Single-user, in-memory state.** Saved detectors and uploaded images persist on disk;
  everything else is per-session.


## Swapping the backbone

`backbones.py` exposes a `FeatureExtractor` interface. `DinoV2Extractor` is ready to use.
For an in-house MAE ViT, implement the two marked hooks (`_load` and `_window_features`) in
`MaeExtractor` and run with `--backbone mae --mae-ckpt path/to/checkpoint.pt`. A detector
saved under one backbone will refuse to load under another, since the feature spaces differ.


## Limitations

- DINOv2-S is pretrained on natural images, not overhead EO; generalisation at sub-metre GSD
  is exactly what the scorecard is there to measure.
- Localisation is patch-blobby (per-patch grid), not tight boxes.
- A linear head on frozen features is weaker than a fine-tuned detector; this tool is a
  labelling / triage / novel-target aid, complementary to an offline-trained detector.
- Scoring is manual (no ground-truth mAP). Labelled dataset splits can be added later for
  automatic evaluation.