"""Run fixed PKfold inference on the bundled synthetic input."""

from __future__ import annotations

from pathlib import Path

import torch

from pkfold.features import build_pair_features
from pkfold.model import load_model
from pkfold.postprocess import decode_pairs
from pkfold.sample import synthetic_record


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _format_pairs(pairs: list[tuple[int, int]]) -> str:
    return ";".join(f"{i}-{j}" for i, j in pairs)


def main() -> None:
    root = _repo_root()
    checkpoint = root / "weights" / "pkfold.pt"
    if not checkpoint.exists():
        raise SystemExit(
            "Missing weights/pkfold.pt. "
            "Place the Zenodo weight file there or run scripts/fetch_weights.py after the Zenodo URL is configured."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    record = synthetic_record()
    model = load_model(checkpoint, device=device)
    features, length = build_pair_features(record, device=device, input_channels=model.input_channels)
    with torch.no_grad():
        logits = model(features)[0]
    pairs = decode_pairs(logits, sequence=record.sequence)

    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.tsv"
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("record_id\tseq_length\tpairs_0based\n")
        handle.write(f"{record.record_id}\t{length}\t{_format_pairs(pairs)}\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
