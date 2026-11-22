"""Sanity check: shapes and parameter counts for the three host configurations."""

import torch

from pakan import AdaptiveKAN, PAKAN1to1, PAKAN2to1


def count(m):
    return sum(p.numel() for p in m.parameters())


# U2Net: fusion interface, D=32, G=5, q=3.
u2 = PAKAN2to1(dim=32, width=32, grid_size=5, spline_order=3)
x = torch.randn(1, 32, 16, 16)
y = torch.randn(1, 32, 16, 16)
out = u2(x, y)
print("PAKAN2to1 out:", tuple(out.shape))
print("  two response branches:", count(u2.kan_spa) + count(u2.kan_chn))
print("  full fusion block:    ", count(u2))

# PanNet: refinement interface, D=48, G=6, q=4.
pan = PAKAN1to1(dim=4, width=48, grid_size=6, spline_order=4)
print("PAKAN1to1 (PanNet) out:", tuple(pan(torch.randn(1, 4, 16, 16)).shape))
print("  full refinement block:", count(pan))

# FusionNet: refinement interface, D=32, G=4, q=3.
fus = PAKAN1to1(dim=32, width=32, grid_size=4, spline_order=3)
print("PAKAN1to1 (FusionNet) out:", tuple(fus(torch.randn(1, 32, 16, 16)).shape))

# A single branch with the U2Net configuration.
branch = AdaptiveKAN(64, 32, grid_size=5, spline_order=3, adapt_dims=(0, 2, 3))
print("AdaptiveKAN(64->32) params:", count(branch))
