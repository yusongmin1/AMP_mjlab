#!/usr/bin/env bash
# Convert motion_data_csv/amp/*.csv (30 Hz) -> WalkandRun/*.npz (50 Hz)
#
# Usage:
#   cd /home/zju/lab/AMP_mjlab
#   bash scripts/convert_amp_csv_to_npz.sh
#   bash scripts/convert_amp_csv_to_npz.sh cuda:0
#   bash scripts/convert_amp_csv_to_npz.sh cuda:0 /path/to/csv_dir /path/to/out_dir

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-cuda:0}"
INPUT_DIR="${2:-$ROOT/motion_data_csv/amp}"
OUTPUT_DIR="${3:-$ROOT/src/assets/motions/g1/amp/WalkandRun}"
INPUT_FPS=30
OUTPUT_FPS=50

# Prefer the isaaclab env if present
if [[ -x /home/zju/miniconda3/envs/env_isaaclab/bin/python ]]; then
  PYTHON=/home/zju/miniconda3/envs/env_isaaclab/bin/python
else
  PYTHON="${PYTHON:-python}"
fi

mkdir -p "$OUTPUT_DIR"

echo "python : $PYTHON"
echo "input  : $INPUT_DIR  (${INPUT_FPS} Hz)"
echo "output : $OUTPUT_DIR (${OUTPUT_FPS} Hz)"
echo "device : $DEVICE"
echo "csv count: $(find "$INPUT_DIR" -maxdepth 1 -name '*.csv' | wc -l)"

"$PYTHON" "$ROOT/scripts/csv_to_npz.py" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --input-fps "$INPUT_FPS" \
  --output-fps "$OUTPUT_FPS" \
  --device "$DEVICE" \
  --render False

echo "Done. npz:"
ls -1 "$OUTPUT_DIR"/*.npz 2>/dev/null | wc -l
