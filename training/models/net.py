"""SE-ResNet-6x64 dual-head network, exactly as specified in SPEC.md §4."""

from __future__ import annotations

import torch
from torch import nn

IN_PLANES = 18
CHANNELS = 64
NUM_BLOCKS = 6
SE_RATIO = 4
NUM_MOVES = 4168
NUM_WDL = 3


def conv_bn(in_ch: int, out_ch: int, kernel: int) -> nn.Sequential:
    """Conv (no bias; BN supplies the shift) followed by BatchNorm."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(out_ch),
    )


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: global pool -> FC -> ReLU -> FC -> sigmoid channel scale."""

    def __init__(self, channels: int = CHANNELS, ratio: int = SE_RATIO):
        super().__init__()
        hidden = channels // ratio
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3))
        s = torch.sigmoid(self.fc2(torch.relu(self.fc1(s))))
        return x * s[:, :, None, None]


class ResBlock(nn.Module):
    """conv3x3-BN-ReLU-conv3x3-BN-SE, plus identity skip, then ReLU."""

    def __init__(self, channels: int = CHANNELS):
        super().__init__()
        self.conv1 = conv_bn(channels, channels, 3)
        self.conv2 = conv_bn(channels, channels, 3)
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.conv1(x))
        out = self.se(self.conv2(out))
        return torch.relu(out + x)


class ChessNet(nn.Module):
    """Input `(B, 18, 8, 8)` -> (`policy_logits (B, 4168)`, `wdl_logits (B, 3)`)."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(conv_bn(IN_PLANES, CHANNELS, 3), nn.ReLU())
        self.trunk = nn.Sequential(*(ResBlock() for _ in range(NUM_BLOCKS)))
        self.policy_head = nn.Sequential(
            conv_bn(CHANNELS, 32, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 64, NUM_MOVES),
        )
        self.value_head = nn.Sequential(
            conv_bn(CHANNELS, 8, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 64, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_WDL),
        )

    def forward(self, input_planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(self.stem(input_planes))
        return self.policy_head(x), self.value_head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
