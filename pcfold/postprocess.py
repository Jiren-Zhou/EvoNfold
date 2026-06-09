"""Fixed postprocessing for PCfold inference.

This module uses one fixed decode path: all concrete RNA base types can pair,
N-containing pairs are masked out, and a contact projection is applied before
thresholding.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

BASE_TO_INDEX = {"A": 0, "U": 1, "C": 2, "G": 3, "N": 4}


def _soft_sign(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-2.0 * x))


def _contact_projection(a_hat: torch.Tensor, constraint_mask: torch.Tensor) -> torch.Tensor:
    a = a_hat * a_hat
    a = 0.5 * (a + a.transpose(-1, -2))
    return a * constraint_mask


def _constraint_mask(sequence: str, *, device: torch.device) -> torch.Tensor:
    seq = str(sequence).upper().replace("T", "U")
    length = len(seq)
    ids = torch.tensor([BASE_TO_INDEX.get(ch, 4) for ch in seq], dtype=torch.long, device=device).view(1, length)
    allowed_lookup = torch.zeros((5, 5), dtype=torch.bool, device=device)
    allowed_lookup[:4, :4] = True
    base_i = ids.unsqueeze(2).expand(1, length, length)
    base_j = ids.unsqueeze(1).expand(1, length, length)
    allowed = allowed_lookup[base_i, base_j]
    idx = torch.arange(length, device=device)
    non_diag = idx[None, :, None] != idx[None, None, :]
    return (allowed & non_diag).to(dtype=torch.float32)


def _project_logits(logits: torch.Tensor, constraint_mask: torch.Tensor) -> torch.Tensor:
    threshold_s = math.log(9.0)
    lr_min = 0.01
    lr_max = 0.1
    rho = 1.6
    u = _soft_sign(logits - threshold_s) * logits
    gate = _soft_sign(logits - threshold_s).detach()
    a_hat = torch.sigmoid(u) * gate
    lmbd = F.relu(torch.sum(_contact_projection(a_hat, constraint_mask), dim=-1) - 1.0).detach()
    for _ in range(100):
        occupancy = torch.sum(_contact_projection(a_hat, constraint_mask), dim=-1) - 1.0
        grad_a = (lmbd * _soft_sign(occupancy)).unsqueeze(-1).expand_as(u) - u / 2.0
        grad = a_hat * constraint_mask * (grad_a + grad_a.transpose(-1, -2))
        a_hat = a_hat - lr_min * grad
        lr_min *= 0.99
        a_hat = F.relu(torch.abs(a_hat) - rho * lr_min)
        lmbd_grad = F.relu(torch.sum(_contact_projection(a_hat, constraint_mask), dim=-1) - 1.0)
        lmbd = lmbd + lr_max * lmbd_grad
        lr_max *= 0.99
    return _contact_projection(a_hat, constraint_mask)


def decode_pairs(logits: torch.Tensor, *, sequence: str, threshold: float = 0.5) -> list[tuple[int, int]]:
    length = len(sequence)
    if length <= 1:
        return []
    score = 0.5 * (logits[:length, :length] + logits[:length, :length].transpose(-1, -2))
    projected = _project_logits(score.unsqueeze(0), _constraint_mask(sequence, device=score.device))[0]
    hits = torch.nonzero(torch.triu(projected >= float(threshold), diagonal=1), as_tuple=False)
    pairs = [(int(i), int(j)) for i, j in hits.tolist()]
    pairs.sort()
    return pairs
