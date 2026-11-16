"""Synthetic input for EvoNfold inference.

The bundled record is generated from scratch and intentionally contains no
database accession, family label, source coordinate, or real record.
"""

from __future__ import annotations

from EvoNfold.features import InferenceRecord


def synthetic_record() -> InferenceRecord:
    sequence = (
        "GGCAUACGUACGUAUCGCAU"
        "AUGCGAAUCGUACGUAUGCC"
        "GCAUACGUAGCAUACGUAUG"
    )
    cov_edge_u = (2, 5, 9, 14, 18, 23, 28, 33, 38, 43, 48, 53)
    cov_edge_v = (57, 52, 49, 45, 41, 36, 31, 26, 21, 16, 11, 6)
    cov_edge_w = (1.0, 0.96, 0.98, 0.95, 0.97, 0.99, 0.95, 0.96, 1.0, 0.98, 0.97, 0.96)
    return InferenceRecord(
        record_id="synthetic_record",
        sequence=sequence,
        cov_edge_u=cov_edge_u,
        cov_edge_v=cov_edge_v,
        cov_edge_w=cov_edge_w,
    )
