# EvoNfold

EvoNfold is a reproducible RNA contact inference package. It contains the code needed to load released checkpoints and run inference.


## Naming and released assets

The method, Python package, asset filenames and checkpoint format identifier
use EvoNfold. The model architecture, inference calculations and weight tensor
values are unchanged. This version expects the renamed asset package
`EvoNfold_assets_20261117.zip` and checkpoint format `EvoNfold.v1`.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

Place the checkpoint file at:

```text
weights/EvoNfold.pt
```

The reproducibility assets are deposited at:

```text
https://zenodo.org/records/20604536
```

For this code version, use the renamed `EvoNfold_assets_20261117.zip`
package. The record URL above identifies the source assets; this code update
does not upload or replace the Zenodo files. The expected ZIP SHA256 is:

```text
0d7581f4ef33b882d6cc0ddebeba0937803cb73216708f762e0371fbaa3bb14c
```

The package contains public and five split-specific datasets and checkpoints.
The public inference entry point uses:

```text
EvoNfold_assets/data/EvoNfold_public.parquet
EvoNfold_assets/weights/EvoNfold_public.pt
```

The helper installs the asset ZIP, verifies the checksum, and copies the
public checkpoint to `weights/EvoNfold.pt`:

```bash
python scripts/fetch_weights.py /path/to/EvoNfold_assets_20261117.zip --weight-name EvoNfold_public.pt --sha256 0d7581f4ef33b882d6cc0ddebeba0937803cb73216708f762e0371fbaa3bb14c
```

## Run

For Code Ocean, upload `EvoNfold_assets_20261117.zip` as a data asset
and run:

```bash
bash run.sh
```

For a local check, set the asset path and output directory explicitly:

```bash
EvoNfold_ASSET_ZIP=/path/to/EvoNfold_assets_20261117.zip RESULTS_DIR=outputs bash run.sh
```

The run writes:

```text
predicted_structure.tsv
predicted_structure.ct
run_summary.txt
```

## Phylogenetic tree input

EvoNfold accepts either a multiple sequence alignment (Stockholm format, `.sto`) alone or together with a corresponding phylogenetic tree.

- If only an MSA is provided, the pipeline automatically infers a phylogenetic tree using the bundled FastTree implementation before downstream covariation analysis.
- If a phylogenetic tree is provided, the pipeline uses the supplied tree directly and does not perform tree reconstruction.

The outputs contain only the selected public record index, sequence length, and
predicted base pairs/CT structure.
