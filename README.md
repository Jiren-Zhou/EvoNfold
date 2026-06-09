# PKfold

PKfold is a reproducible RNA contact inference package. It contains the code needed to load released checkpoints and run inference.


## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

Place the checkpoint file at:

```text
weights/pkfold.pt
```

The helper can also install a local released assets tarball or unpacked assets
directory. It copies the selected weight to the checkpoint path above and keeps the
full assets package under `assets/`:

```bash
python scripts/fetch_weights.py /path/to/PKfold_assets_20260609.tar.gz
```

Use `--weight-name PKfold_test2.pt` or another bundled weight name to select a
different checkpoint. If `assets_manifest.json` is present, running
`scripts/fetch_weights.py` without arguments downloads and verifies the declared
assets.

## Run

```bash
python -m pkfold.infer
```

The command runs a synthetic input and writes:

```text
outputs/predictions.tsv
```

The output contains only a synthetic record ID, sequence length, and predicted
0-based pair indices.
