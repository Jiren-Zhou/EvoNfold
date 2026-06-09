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

Download `PCfold_assets_20260609.tar.gz` from the Zenodo record. The expected
SHA256 checksum is:

```text
5db11f539e6a89e91200baccda22c7cef327bae2b66cf2bd58802a7daf2af1a5
```

The helper installs a local assets tarball or unpacked assets directory. It
copies the selected weight to the checkpoint path above and keeps the full
assets package under `assets/`:

```bash
python scripts/fetch_weights.py /path/to/PCfold_assets_20260609.tar.gz --sha256 5db11f539e6a89e91200baccda22c7cef327bae2b66cf2bd58802a7daf2af1a5
```

Use `--weight-name PCfold_test2.pt` or another bundled weight name to select a
different checkpoint. If `assets_manifest.json` is present, running
`scripts/fetch_weights.py` without arguments downloads and verifies the declared
assets.

## Run

```bash
python -m pcfold.infer
```

The command runs a synthetic input and writes:

```text
outputs/predictions.tsv
```

The output contains only a synthetic record ID, sequence length, and predicted
0-based pair indices.
