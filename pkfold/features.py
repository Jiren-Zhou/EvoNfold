"""Build fixed pair-map features for PKfold inference.

This module turns a single RNA sequence plus optional covariance-prior edges
into the 17-channel pair tensor expected by the released model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class InferenceRecord:
    record_id: str
    sequence: str
    cov_edge_u: tuple[int, ...] = ()
    cov_edge_v: tuple[int, ...] = ()
    cov_edge_w: tuple[float, ...] = ()


def normalize_sequence(sequence: str) -> str:
    return "".join(("U" if ch == "T" else ch) for ch in str(sequence).upper())


def one_hot_sequence(sequence: str, *, device: torch.device) -> torch.Tensor:
    seq = normalize_sequence(sequence)
    mapping = {"A": 0, "U": 1, "C": 2, "G": 3}
    out = torch.zeros((len(seq), 4), dtype=torch.float32, device=device)
    for idx, base in enumerate(seq):
        channel = mapping.get(base)
        if channel is not None:
            out[idx, channel] = 1.0
    return out


def padded_length(length: int, *, min_length: int = 80, multiple: int = 16) -> int:
    n = int(length)
    if n <= int(min_length):
        return int(min_length)
    return (((n - 1) // int(multiple)) + 1) * int(multiple)


def cov_prior_from_edges(
    *,
    length: int,
    cov_edge_u: Sequence[int],
    cov_edge_v: Sequence[int],
    cov_edge_w: Sequence[float],
    device: torch.device,
    scale: float = 1.0,
    clip: float = 10.0,
) -> torch.Tensor:
    prior = torch.zeros((int(length), int(length)), dtype=torch.float32, device=device)
    limit = float(clip)
    use_clip = limit > 0.0
    for raw_i, raw_j, raw_w in zip(cov_edge_u, cov_edge_v, cov_edge_w):
        try:
            i = int(raw_i)
            j = int(raw_j)
            weight = float(raw_w) * float(scale)
        except (TypeError, ValueError):
            continue
        if i == j or i < 0 or j < 0 or i >= int(length) or j >= int(length):
            continue
        if use_clip:
            weight = max(-limit, min(limit, weight))
        a, b = (i, j) if i < j else (j, i)
        if abs(weight) >= abs(float(prior[a, b].item())):
            prior[a, b] = weight
            prior[b, a] = weight
    return prior


def build_pair_features(
    record: InferenceRecord,
    *,
    device: torch.device,
    input_channels: int = 17,
) -> tuple[torch.Tensor, int]:
    seq = normalize_sequence(record.sequence)
    length = len(seq)
    if length <= 0:
        raise ValueError("sequence must not be empty")
    channels = max(17, int(input_channels))
    pad_len = padded_length(length)
    onehot = one_hot_sequence(seq, device=device)
    features = torch.zeros((1, channels, pad_len, pad_len), dtype=torch.float32, device=device)

    channel = 0
    for i in range(4):
        left = onehot[:, i].view(-1, 1)
        for j in range(4):
            right = onehot[:, j].view(1, -1)
            features[0, channel, :length, :length] = left @ right
            channel += 1

    prior = cov_prior_from_edges(
        length=length,
        cov_edge_u=record.cov_edge_u,
        cov_edge_v=record.cov_edge_v,
        cov_edge_w=record.cov_edge_w,
        device=device,
    )
    features[0, 16, :length, :length] = prior
    return features, length
