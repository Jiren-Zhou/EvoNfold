#!/usr/bin/env bash
# Code Ocean run script for PCfold public benchmark inference.
# It installs the public-only assets, runs one inference case, and writes
# predicted structures to the platform results directory.

set -euo pipefail

cd "$(dirname "$0")"

ASSET_TAR="${PCFOLD_ASSET_TAR:-/data/PCfold_public_infer_20260615.tar.gz}"
ASSET_SHA="${PCFOLD_ASSET_SHA:-cb8e55e6aa5592481ea625f77bd90bb39d5049f0fb315d173a0a8957c651e98d}"
RESULTS_DIR="${RESULTS_DIR:-/results}"

python scripts/fetch_weights.py "$ASSET_TAR" --weight-name PCfold_public.pt --sha256 "$ASSET_SHA"
python run_public_infer.py \
  --parquet assets/PCfold_public_infer/data/PCfold_public.parquet \
  --output-dir "$RESULTS_DIR"

cat > "$RESULTS_DIR/run_summary.txt" <<EOF
PCfold public benchmark inference completed
asset: $(basename "$ASSET_TAR")
sha256: $ASSET_SHA
checkpoint: weights/pcfold.pt
outputs:
  $RESULTS_DIR/predicted_structure.tsv
  $RESULTS_DIR/predicted_structure.ct
EOF
