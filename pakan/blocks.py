"""Drop-in blocks built from the PAKAN response core.

PAKAN2to1 is the fusion interface (two streams -> one), used in U2Net.
PAKAN1to1 is the refinement interface (one stream -> one), used in PanNet
and FusionNet.
"""

import torch
import torch.nn as nn

from .module import AdaptiveKAN


class PAKAN2to1(nn.Module):
    """Fusion interface for two-stream hosts.

    Args:
        dim: channel width of the two input streams.
        width: internal width D of one branch (paper: 32).
        grid_size, spline_order: spline configuration (paper: 5, 3).
        num_groups: channel groups (paper: 1).

    Input:  x, y each (B, dim, H, W).
    Output: (B, dim, H, W).
    """

    def __init__(self, dim, width=32, grid_size=5, spline_order=3, num_groups=1):
        super().__init__()
        self.convx = nn.Conv2d(dim, width, 3, 1, 1)
        self.convy = nn.Conv2d(dim, width, 3, 1, 1)
        self.kan_spa = AdaptiveKAN(
            2 * width, width, grid_size, spline_order, num_groups,
            adapt_dims=(0, 2, 3), channel_act="sigmoid",
        )
        self.kan_chn = AdaptiveKAN(
            2 * width, width, grid_size, spline_order, num_groups,
            adapt_dims=(0, 1), channel_act="sigmoid",
        )
        self.skip_swap = nn.Parameter(torch.tensor([1.0]))
        self.final_conv = nn.Conv2d(width, dim, 1)

    def forward(self, x, y):
        x_conv = self.convx(x)
        y_conv = self.convy(y)

        alpha = torch.sigmoid(self.skip_swap)
        skip = x_conv * alpha + y_conv * (1.0 - alpha)

        # Both branches read the channel-wise concatenation of the two streams.
        joint = torch.cat([x_conv, y_conv], dim=1)
        spa = self.kan_spa(joint)
        chn = self.kan_chn(joint)

        return self.final_conv(spa * chn + skip)


class PAKAN1to1(nn.Module):
    """Refinement interface for single-stream hosts.

    Args:
        dim: input and output channel width.
        width: internal width D of one branch (paper: 48 for PanNet, 32 for FusionNet).
        grid_size, spline_order: spline configuration.
        num_groups: channel groups (paper: 1).
        channel_act: activation of the channel gate ("silu" in the paper).

    Input:  (B, dim, H, W).
    Output: (B, dim, H, W).
    """

    def __init__(
        self, dim, width=48, grid_size=6, spline_order=4,
        num_groups=1, channel_act="silu",
    ):
        super().__init__()
        self.convx = nn.Conv2d(dim, width, 3, 1, 1)
        self.kan_spa = AdaptiveKAN(
            width, width, grid_size, spline_order, num_groups,
            adapt_dims=(0, 2, 3), channel_act="sigmoid",
        )
        self.kan_chn = AdaptiveKAN(
            width, width, grid_size, spline_order, num_groups,
            adapt_dims=(0, 1), channel_act=channel_act,
        )
        self.final_conv = nn.Conv2d(width, dim, 1)

    def forward(self, x):
        feat = self.convx(x)
        return self.final_conv(self.kan_spa(feat) * self.kan_chn(feat))
