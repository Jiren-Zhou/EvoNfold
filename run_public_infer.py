"""Run EvoNfold inference on one record from the public benchmark parquet.

This script is the Code Ocean entry point after the assets have been installed:
it reads a single sequence plus covariance-prior edges, runs EvoNfold, and writes
the predicted structure as pair indices and CT format without computing metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
import torch

from EvoNfold.features import InferenceRecord, build_pair_features, normalize_sequence
from EvoNfold.model import load_model
from EvoNfold.postprocess import decode_pairs


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _default_parquet(root: Path) -> Path:
    candidates = sorted((root / "assets").glob("*/data/EvoNfold_public.parquet"))
    if candidates:
        return candidates[0]
    return root / "assets" / "EvoNfold_assets" / "data" / "EvoNfold_public.parquet"


def _to_tuple(value: object, *, cast: type) -> tuple:
    if value is None:
        return ()
    if isinstance(value, list):
        return tuple(cast(item) for item in value)
    if isinstance(value, tuple):
        return tuple(cast(item) for item in value)
    return tuple(cast(item) for item in list(value))


def _load_record(parquet_path: Path, record_index: int) -> InferenceRecord:
    table = pq.read_table(
        parquet_path,
        columns=["sequence", "cov_edge_u", "cov_edge_v", "cov_edge_w"],
    )
    if table.num_rows <= 0:
        raise ValueError(f"{parquet_path} contains no records")
    if record_index < 0 or record_index >= table.num_rows:
        raise IndexError(f"record index {record_index} out of range for {table.num_rows} records")
    row = table.slice(record_index, 1).to_pydict()
    return InferenceRecord(
        record_id=f"public_{record_index:04d}",
        sequence=str(row["sequence"][0]),
        cov_edge_u=_to_tuple(row["cov_edge_u"][0], cast=int),
        cov_edge_v=_to_tuple(row["cov_edge_v"][0], cast=int),
        cov_edge_w=_to_tuple(row["cov_edge_w"][0], cast=float),
    )


def _format_pairs(pairs: Iterable[tuple[int, int]]) -> str:
    return ";".join(f"{i}-{j}" for i, j in pairs)


def _write_pairs_tsv(path: Path, *, record: InferenceRecord, length: int, pairs: list[tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("record_id\tseq_length\tpairs_0based\n")
        handle.write(f"{record.record_id}\t{length}\t{_format_pairs(pairs)}\n")


def _write_ct(path: Path, *, record: InferenceRecord, pairs: list[tuple[int, int]]) -> None:
    sequence = normalize_sequence(record.sequence)
    pair_map = {idx: 0 for idx in range(len(sequence))}
    for i, j in pairs:
        if 0 <= i < len(sequence) and 0 <= j < len(sequence):
            pair_map[i] = j + 1
            pair_map[j] = i + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(sequence)}\t{record.record_id}\n")
        for idx, base in enumerate(sequence, start=1):
            prev_idx = idx - 1 if idx > 1 else 0
            next_idx = idx + 1 if idx < len(sequence) else 0
            handle.write(f"{idx}\t{base}\t{prev_idx}\t{next_idx}\t{pair_map[idx - 1]}\t{idx}\n")


def _parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Run EvoNfold on one public benchmark record.")
    parser.add_argument("--parquet", type=Path, default=_default_parquet(root), help="public benchmark parquet")
    parser.add_argument("--checkpoint", type=Path, default=root / "weights" / "EvoNfold.pt", help="installed checkpoint")
    parser.add_argument("--record-index", type=int, default=0, help="0-based row index to infer")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs", help="directory for prediction outputs")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    record = _load_record(args.parquet, args.record_index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device=device)
    features, length = build_pair_features(record, device=device, input_channels=model.input_channels)
    with torch.no_grad():
        logits = model(features)[0]
    pairs = decode_pairs(logits, sequence=record.sequence)

    _write_pairs_tsv(args.output_dir / "predicted_structure.tsv", record=record, length=length, pairs=pairs)
    _write_ct(args.output_dir / "predicted_structure.ct", record=record, pairs=pairs)
    print(f"Wrote {args.output_dir / 'predicted_structure.tsv'}")
    print(f"Wrote {args.output_dir / 'predicted_structure.ct'}")


if __name__ == "__main__":
    main()
