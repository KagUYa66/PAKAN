"""Pixel-Adaptive Kolmogorov--Arnold Network (PAKAN) modules.

This is the module used in the paper. Each branch evaluates a fixed
power-triangular spline basis on the input features and modulates the
resulting edge functions with a gate derived from the same features.
"""

import math

import torch
import torch.nn as nn


class AdaptiveKAN(nn.Module):
    """A spline response whose edge functions are gated by the input.

    Args:
        in_features: number of input channels.
        out_features: number of output channels.
        grid_size: number of basis functions and grid points (G).
        spline_order: exponent q applied to the triangular basis.
        num_groups: channel groups of the coefficient tensor.
        adapt_dims: dimensions of the (B, H, W, C) input that the gate varies over.
            (0, 2, 3) gives a per-location gate shared over channels;
            (0, 1) gives a per-channel gate pooled over space.
        channel_act: activation of the channel-gate generator, "silu" or "sigmoid".

    Input:  (B, C, H, W) with C == in_features.
    Output: (B, out_features, H, W).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        num_groups: int = 1,
        adapt_dims=(0, 2, 3),
        channel_act: str = "silu",
    ):
        super().__init__()
        if in_features % num_groups or out_features % num_groups:
            raise ValueError("in/out features must be divisible by num_groups")

        adapt_dims = tuple(sorted(set(adapt_dims)))
        if adapt_dims not in ((0, 1), (0, 2, 3)):
            raise ValueError("adapt_dims must be (0, 1) or (0, 2, 3)")

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.num_groups = num_groups
        self.in_per_group = in_features // num_groups
        self.out_per_group = out_features // num_groups
        self.adapt_dims = adapt_dims

        # Fixed triangular grid.
        self.register_buffer("centers", torch.linspace(-1.0, 1.0, grid_size))
        self.register_buffer("delta", torch.tensor(2.0 / max(grid_size - 1, 1)))
        self.register_buffer("eps", torch.tensor(1e-6))

        if adapt_dims == (0, 1):
            act = nn.SiLU() if channel_act == "silu" else nn.Sigmoid()
            self.gate_gen = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_features, in_features, 1),
                act,
            )
        else:
            self.gate_gen = nn.Sequential(
                nn.Conv2d(in_features, in_features * 2, 1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_features * 2, 1, 1),
                nn.Sigmoid(),
            )

        bound = 1.0 / math.sqrt(self.in_per_group)
        self.weight_spline = nn.Parameter(
            torch.empty(num_groups, self.out_per_group, self.in_per_group, grid_size)
        )
        self.weight_linear = nn.Parameter(
            torch.empty(num_groups, self.out_per_group, self.in_per_group)
        )
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.uniform_(self.weight_spline, -bound, bound)
        nn.init.uniform_(self.weight_linear, -bound, bound)

    def _spline_basis(self, values: torch.Tensor) -> torch.Tensor:
        """Power-triangular basis, normalized over the grid.

        values: (..., in_per_group) -> (..., in_per_group, grid_size)
        """
        dist = (values.unsqueeze(-1) - self.centers) / (self.delta + self.eps)
        basis = torch.relu(1.0 - dist.abs()) ** self.spline_order
        norm = basis.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return basis / norm

    def _compute_gate(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        if self.adapt_dims == (0, 1):
            return self.gate_gen(x).view(
                batch, 1, 1, self.num_groups, self.in_per_group, 1
            )
        return self.gate_gen(x).view(batch, height, width, self.num_groups, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_features:
            raise ValueError(
                f"expected (B, {self.in_features}, H, W), got {tuple(x.shape)}"
            )

        batch, _, height, width = x.shape
        grouped = x.permute(0, 2, 3, 1).contiguous().view(
            batch, height, width, self.num_groups, self.in_per_group
        )

        basis = self._spline_basis(grouped)
        gate = self._compute_gate(x)

        basis = basis * gate
        grouped = grouped * gate.squeeze(-1)

        w_spline = self.weight_spline.view(
            1, 1, 1, self.num_groups, self.out_per_group, self.in_per_group,
            self.grid_size,
        )
        w_linear = self.weight_linear.view(
            1, 1, 1, self.num_groups, self.out_per_group, self.in_per_group
        )

        spline_out = (basis.unsqueeze(-3) * w_spline).sum(dim=-1).sum(dim=-1)
        linear_out = (grouped.unsqueeze(-2) * w_linear).sum(dim=-1)

        out = spline_out + linear_out
        out = out.reshape(batch, height, width, self.out_features)
        out = out + self.bias.view(1, 1, 1, self.out_features)
        return out.permute(0, 3, 1, 2).contiguous()
