# EvoNfold

EvoNfold is a reproducible RNA contact inference package. It contains the code needed to load released checkpoints and run inference.


## Naming and released assets

The method and Python package are now named EvoNfold (formerly PCfold). This is a naming-only update: the model architecture, inference calculations and released weights are unchanged. The GitHub repository URL, Zenodo record and existing Code Ocean resources are retained. Published asset filenames, the `pcfold.v1` checkpoint format identifier and the `PCFOLD_ASSET_TAR` / `PCFOLD_ASSET_SHA` environment variables remain unchanged so the existing releases continue to work. The installed local checkpoint is now `weights/EvoNfold.pt`.

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
public checkpoint to `weights/EvoNfold.pt`:

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

## Phylogenetic tree input

EvoNfold accepts either a multiple sequence alignment (Stockholm format, `.sto`) alone or together with a corresponding phylogenetic tree.

- If only an MSA is provided, the pipeline automatically infers a phylogenetic tree using the bundled FastTree implementation before downstream covariation analysis.
- If a phylogenetic tree is provided, the pipeline uses the supplied tree directly and does not perform tree reconstruction.

The outputs contain only the selected public record index, sequence length, and
predicted base pairs/CT structure.
