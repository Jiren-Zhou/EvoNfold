"""PKfold model definition and checkpoint loading.

The released checkpoint stores only a model state dict and the small
architecture descriptor required to rebuild it for inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F


class _ModernResBlock2D(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, norm_groups: int) -> None:
        super().__init__()
        groups = max(1, min(int(norm_groups), int(out_channels)))
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = torch.nn.GroupNorm(groups, out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = torch.nn.GroupNorm(groups, out_channels)
        self.skip = (
            torch.nn.Identity()
            if int(in_channels) == int(out_channels)
            else torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        y = F.silu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.silu(y + residual)


class _ModernDown2D(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, norm_groups: int) -> None:
        super().__init__()
        self.down = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.block = _ModernResBlock2D(out_channels, out_channels, norm_groups=norm_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x))


class _ModernUp2D(torch.nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, *, norm_groups: int) -> None:
        super().__init__()
        self.reduce = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.block = _ModernResBlock2D(out_channels + skip_channels, out_channels, norm_groups=norm_groups)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        return self.block(torch.cat([x, skip], dim=1))


class ModernResUNet2D(torch.nn.Module):
    def __init__(
        self,
        *,
        img_ch: int,
        base_channels: int = 32,
        depth: int = 4,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.img_ch = int(img_ch)
        self.base_channels = int(base_channels)
        self.depth = int(depth)
        self.norm_groups = int(norm_groups)
        if self.depth < 1:
            raise ValueError("depth must be >= 1")

        channels = [self.base_channels * (2**idx) for idx in range(self.depth + 1)]
        self.stem = _ModernResBlock2D(self.img_ch, channels[0], norm_groups=self.norm_groups)
        self.downs = torch.nn.ModuleList(
            _ModernDown2D(channels[idx], channels[idx + 1], norm_groups=self.norm_groups)
            for idx in range(self.depth)
        )
        self.bottleneck = _ModernResBlock2D(channels[-1], channels[-1], norm_groups=self.norm_groups)
        self.ups = torch.nn.ModuleList(
            _ModernUp2D(channels[idx + 1], channels[idx], channels[idx], norm_groups=self.norm_groups)
            for idx in reversed(range(self.depth))
        )
        self.head = torch.nn.Conv2d(channels[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expected [B,C,H,W], got {tuple(x.shape)}")
        orig_h, orig_w = int(x.shape[-2]), int(x.shape[-1])
        skips: list[torch.Tensor] = []
        y = self.stem(x)
        skips.append(y)
        for down in self.downs:
            y = down(y)
            skips.append(y)
        y = self.bottleneck(y)
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            y = up(y, skip)
        logits = self.head(y)[:, 0, :orig_h, :orig_w]
        return 0.5 * (logits + logits.transpose(-1, -2))


class PKfoldModel(torch.nn.Module):
    def __init__(
        self,
        *,
        input_channels: int = 17,
        base_channels: int = 32,
        depth: int = 4,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.u_net = ModernResUNet2D(
            img_ch=self.input_channels,
            base_channels=int(base_channels),
            depth=int(depth),
            norm_groups=int(norm_groups),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.u_net(features)


def _model_config(payload: Mapping[str, Any]) -> dict[str, int]:
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("checkpoint is missing model metadata")
    return {
        "input_channels": int(model.get("input_channels", 17)),
        "base_channels": int(model.get("base_channels", 32)),
        "depth": int(model.get("depth", 4)),
        "norm_groups": int(model.get("norm_groups", 8)),
    }


def load_model(checkpoint_path: str | Path, *, device: torch.device) -> PKfoldModel:
    payload = torch.load(Path(checkpoint_path), map_location=device)
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint payload must be a mapping")
    if str(payload.get("format")) != "pkfold.v1":
        raise ValueError("unsupported checkpoint format")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint is missing state_dict")
    model = PKfoldModel(**_model_config(payload)).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
