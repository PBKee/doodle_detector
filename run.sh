#!/usr/bin/env bash
# Convenience launcher. Run from the project root.
set -e
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)"

MODE="${1:-demo}"   # demo | dinov2 | mae

case "$MODE" in
  demo)
    python -m backend.app --demo --backbone mock --gsd 0.3 --tile-km 1.0 ;;
  dinov2)
    # needs: pip install torch --index-url https://download.pytorch.org/whl/cu128
    python -m backend.app --image "${2:?path to image/GeoTIFF}" \
      --backbone dinov2 --dino-variant dinov2_vits14 --gsd "${3:-0.3}" --tile-km 1.0 ;;
  mae)
    python -m backend.app --image "${2:?path to image/GeoTIFF}" \
      --backbone mae --mae-ckpt "${3:?path to MAE checkpoint}" --gsd "${4:-0.3}" ;;
  *)
    echo "usage: ./run.sh [demo | dinov2 <image> [gsd] | mae <image> <ckpt> [gsd]]" ; exit 1 ;;
esac
