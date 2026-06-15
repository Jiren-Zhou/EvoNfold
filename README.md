# PCfold

PCfold is a reproducible RNA contact inference package. It contains the code needed to load released checkpoints and run inference.


## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

Place the checkpoint file at:

```text
weights/pcfold.pt
```

The reproducibility assets are deposited at:

```text
https://zenodo.org/records/20603698
```

For Code Ocean inference, download `PCfold_public_infer_20260615.tar.gz` from
the Zenodo record. The expected SHA256 checksum is:

```text
cb8e55e6aa5592481ea625f77bd90bb39d5049f0fb315d173a0a8957c651e98d
```

The public inference asset contains only:

```text
data/PCfold_public.parquet
weights/PCfold_public.pt
```

The helper installs the asset tarball, verifies the checksum, and copies the
public checkpoint to `weights/pcfold.pt`:

```bash
python scripts/fetch_weights.py /path/to/PCfold_public_infer_20260615.tar.gz --weight-name PCfold_public.pt --sha256 cb8e55e6aa5592481ea625f77bd90bb39d5049f0fb315d173a0a8957c651e98d
```

## Run

For Code Ocean, upload `PCfold_public_infer_20260615.tar.gz` as a data asset
and run:

```bash
bash run.sh
```

For a local check, set the asset path and output directory explicitly:

```bash
PCFOLD_ASSET_TAR=/path/to/PCfold_public_infer_20260615.tar.gz RESULTS_DIR=outputs bash run.sh
```

The run writes:

```text
predicted_structure.tsv
predicted_structure.ct
run_summary.txt
```

The outputs contain only the selected public record index, sequence length, and
predicted base pairs/CT structure.
