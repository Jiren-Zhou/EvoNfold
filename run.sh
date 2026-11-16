#!/usr/bin/env bash
# Code Ocean run script for EvoNfold public benchmark inference.
# It installs the assets, runs one inference case, and writes
# predicted structures to the platform results directory.

set -euo pipefail

cd "$(dirname "$0")"

ASSET_ZIP="${EvoNfold_ASSET_ZIP:-/data/EvoNfold_assets_20261117.zip}"
ASSET_SHA="${EvoNfold_ASSET_SHA:-0d7581f4ef33b882d6cc0ddebeba0937803cb73216708f762e0371fbaa3bb14c}"
RESULTS_DIR="${RESULTS_DIR:-/results}"

python scripts/fetch_weights.py "$ASSET_ZIP" --weight-name EvoNfold_public.pt --sha256 "$ASSET_SHA"
python run_public_infer.py \
  --parquet assets/EvoNfold_assets/data/EvoNfold_public.parquet \
  --output-dir "$RESULTS_DIR"

cat > "$RESULTS_DIR/run_summary.txt" <<EOF
EvoNfold public benchmark inference completed
asset: $(basename "$ASSET_ZIP")
sha256: $ASSET_SHA
checkpoint: weights/EvoNfold.pt
outputs:
  $RESULTS_DIR/predicted_structure.tsv
  $RESULTS_DIR/predicted_structure.ct
EOF
