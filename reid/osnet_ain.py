"""
SIH 2026 NETRA AI ENGINE - Re-Identification Module
===================================================
Module: reid/osnet_ain.py

Description:
    Pure PyTorch implementation of the Omni-Scale Network with Adaptive Instance
    Normalization (OSNet-AIN x1.0) for Vehicle Re-Identification.

Key Characteristics:
    - Pure PyTorch (requires only torch and torch.nn; zero external heavy dependencies).
    - Architecture: Omni-scale multi-branch depthwise convolutions + channel gating.
    - Input: (B, 3, 208, 208) RGB image tensor normalized to [0.0, 1.0].
    - Output: (B, 512) raw appearance feature embedding.
    - Target: Pretrained on VeRi-776 vehicle surveillance re-identification dataset.
    - Parameter Count: ~2.19 Million (~8.8 MB model weights).
    - License: MIT License (Kaiyang Zhou / Intel OpenVINO Open Model Zoo).
"""

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. CORE BUILDING BLOCKS
# =============================================================================

class ConvLayer(nn.Module):
    """Standard Convolution block: Conv2d + (BatchNorm2d or InstanceNorm2d) + ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        IN: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups,
        )
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    """1x1 Convolution block: Conv2d(1x1) + BatchNorm2d + ReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=False,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    """1x1 Linear Convolution block: Conv2d(1x1) + optional BatchNorm2d (no non-linearity)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, bn: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels) if bn else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class LightConv3x3(nn.Module):
    """Lightweight 3x3 depthwise-separable convolution: Conv1x1 + Depthwise-Conv3x3 + BN + ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            groups=out_channels,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class LightConvStream(nn.Module):
    """A cascade of `depth` lightweight 3x3 convolutions capturing various receptive field scales."""

    def __init__(self, in_channels: int, out_channels: int, depth: int):
        super().__init__()
        layers = [LightConv3x3(in_channels, out_channels)]
        for _ in range(depth - 1):
            layers.append(LightConv3x3(out_channels, out_channels))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ChannelGate(nn.Module):
    """Squeeze-and-Excitation Channel Attention Gating module."""

    def __init__(self, in_channels: int, num_gates: Optional[int] = None, reduction: int = 16):
        super().__init__()
        if num_gates is None:
            num_gates = in_channels
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, num_gates, 1, bias=True)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.global_avgpool(x)
        x = self.relu(self.fc1(x))
        x = self.gate_activation(self.fc2(x))
        return inp * x


# =============================================================================
# 2. OMNI-SCALE BLOCKS
# =============================================================================

class OSBlock(nn.Module):
    """Omni-Scale residual feature learning block with 4 scale streams (T=4)."""

    def __init__(self, in_channels: int, out_channels: int, reduction: int = 4, T: int = 4):
        super().__init__()
        mid_channels = out_channels // reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList([
            LightConvStream(mid_channels, mid_channels, t) for t in range(1, T + 1)
        ])
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = (
            Conv1x1Linear(in_channels, out_channels) if in_channels != out_channels else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x) if self.downsample is not None else x
        x1 = self.conv1(x)
        x2 = sum(self.gate(stream(x1)) for stream in self.conv2)
        x3 = self.conv3(x2)
        return F.relu(x3 + identity)


class OSBlockINin(nn.Module):
    """Omni-Scale residual block with Instance Normalization applied inside the residual path."""

    def __init__(self, in_channels: int, out_channels: int, reduction: int = 4, T: int = 4):
        super().__init__()
        mid_channels = out_channels // reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList([
            LightConvStream(mid_channels, mid_channels, t) for t in range(1, T + 1)
        ])
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels, bn=False)
        self.downsample = (
            Conv1x1Linear(in_channels, out_channels) if in_channels != out_channels else None
        )
        self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x) if self.downsample is not None else x
        x1 = self.conv1(x)
        x2 = sum(self.gate(stream(x1)) for stream in self.conv2)
        x3 = self.conv3(x2)
        return F.relu(self.IN(x3) + identity)


# =============================================================================
# 3. FULL OSNET-AIN NETWORK
# =============================================================================

class OSNetAIN(nn.Module):
    """
    Omni-Scale Network with Adaptive Instance Normalization (OSNet-AIN x1.0).
    Configured for 512-dimensional vehicle appearance feature extraction.
    """

    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim

        # Input instance normalization (standardizes illumination variations)
        self.input_IN = nn.InstanceNorm2d(3, affine=True)

        # Stage 1: Initial feature extraction
        self.conv1 = ConvLayer(3, 64, 7, stride=2, padding=3, IN=True)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)

        # Stage 2: Low-level omni-scale features (channels: 64 -> 256)
        self.conv2 = nn.Sequential(OSBlockINin(64, 256), OSBlockINin(256, 256))
        self.pool2 = nn.Sequential(Conv1x1(256, 256), nn.AvgPool2d(2, stride=2))

        # Stage 3: Mid-level omni-scale features (channels: 256 -> 384)
        self.conv3 = nn.Sequential(OSBlock(256, 384), OSBlockINin(384, 384))
        self.pool3 = nn.Sequential(Conv1x1(384, 384), nn.AvgPool2d(2, stride=2))

        # Stage 4: High-level omni-scale features (channels: 384 -> 512)
        self.conv4 = nn.Sequential(OSBlockINin(384, 512), OSBlock(512, 512))

        # Stage 5: Transition layer (channels: 512 -> 512)
        self.conv5 = Conv1x1(512, 512)

        # Dual embedding projection heads (256 + 256 = 512 dimensions)
        half_dim = feature_dim // 2
        self.fc = nn.ModuleList([
            nn.Sequential(nn.Linear(512, half_dim), nn.BatchNorm1d(half_dim)),
            nn.Sequential(nn.Linear(512, half_dim), nn.BatchNorm1d(half_dim)),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for feature embedding extraction.

        Args:
            x: Input tensor of shape (B, 3, H, W), typically (B, 3, 208, 208).

        Returns:
            Extracted feature embedding tensor of shape (B, 512).
        """
        x = self.input_IN(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # Global spatial average pooling (B, 512, H, W) -> (B, 512)
        v = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)

        # Dual head projection and concatenation -> (B, 512)
        e0 = self.fc[0](v)
        e1 = self.fc[1](v)
        return torch.cat([e0, e1], dim=1)


def build_osnet_ain_x1_0(
    weights_path: Optional[Union[str, Path]] = None,
    device: str = "cpu",
) -> OSNetAIN:
    """
    Factory function to construct and optionally initialize OSNet-AIN x1.0.

    Args:
        weights_path: Optional path to a PyTorch state_dict file (.pt).
        device: Target compute device ('cuda', 'cuda:0', 'cpu').

    Returns:
        OSNetAIN model loaded into eval mode on the specified device.
    """
    model = OSNetAIN(feature_dim=512)

    if weights_path is not None:
        p = Path(weights_path).resolve()
        if not p.exists():
            raise FileNotFoundError(
                f"OSNet-AIN vehicle Re-ID weights not found at:\n  {p}\n"
                "Please verify that the weight file exists before loading."
            )
        state_dict = torch.load(str(p), map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()
    return model
