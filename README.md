# PAKAN: Pixel-Adaptive Kolmogorov–Arnold Network Modules

Reference implementation of the PAKAN modules described in

> **PAKAN: Pixel-Adaptive Kolmogorov–Arnold Network Modules for Pansharpening**

PAKAN replaces the feature-processing unit of a pansharpening backbone with a
spline response whose edge functions are gated by the input. Two interfaces are
provided: a **fusion** block (two streams → one) and a **refinement** block
(one stream → one).

## What is released

- The response core (`AdaptiveKAN`) and the two drop-in blocks.
- The exact configuration used for each host network in the paper.

**Not released:** trained weights. The module code is self-contained and can be
inserted into a host network and trained with the host's existing loss.

## Installation

```bash
pip install torch
```

No other dependency is required. Copy the `pakan/` directory into your project
or install it with `pip install -e .`.

## Usage

Fusion interface (two streams, e.g. a two-branch backbone):

```python
import torch
from pakan import PAKAN2to1

block = PAKAN2to1(dim=32, width=32, grid_size=5, spline_order=3)
x = torch.randn(1, 32, 64, 64)
y = torch.randn(1, 32, 64, 64)
out = block(x, y)            # (1, 32, 64, 64)
```

Refinement interface (single stream):

```python
from pakan import PAKAN1to1

block = PAKAN1to1(dim=4, width=48, grid_size=6, spline_order=4, channel_act="silu")
out = block(torch.randn(1, 4, 64, 64))   # (1, 4, 64, 64)
```

A single branch can also be used directly:

```python
from pakan import AdaptiveKAN

branch = AdaptiveKAN(64, 32, grid_size=5, spline_order=3, adapt_dims=(0, 2, 3))
out = branch(torch.randn(2, 64, 32, 32))   # (2, 32, 32, 32)
```

## Configurations used in the paper

| Host | Interface | Width `D` | `C_in → C_out` | `G, q` | Spatial gate | Channel gate | Groups |
|---|---|---|---|---|---|---|---|
| U2Net | fusion (2-to-1) | 32 | 2D → D | 5, 3 | sigmoid | sigmoid | 1 |
| PanNet | refinement (1-to-1) | 48 | D → D | 6, 4 | sigmoid | SiLU | 1 |
| FusionNet | refinement (1-to-1) | 32 | D → D | 4, 3 | sigmoid | SiLU | 1 |

`G` is the number of basis functions, `q` the exponent of the power-triangular
basis, and the channel-group count is one for all hosts.

## The response core

For an input feature `u` the basis is

```
B_k(u) = g_k(u) / sum_j g_j(u),    g_k(u) = [max(0, 1 - |u - c_k| / delta)]^q
c_k = -1 + 2(k-1)/(G-1),          delta = 2/(G-1)
```

with `K = G`. Centers and width are fixed buffers; only the coefficients are
learned. The branch output is

```
Y = sum_i sum_k W[j,i,k] * a(p,i,k) * B_k(F[i])
```

where `a` is the gate. In every experiment reported in the paper the gate does
not depend on the basis index (`R = 1`), so it can be taken outside the sum over
`k` and **rescales the learned edge function**; it does not change the relative
contribution of the basis functions. A gate that varies with `k` (`R = K`) is
implemented by the same expression but was not evaluated in the paper.

## Parameter counts

With the U2Net configuration (`C_in = 64`, `C_out = 32`, `K = 5`, one group),
the two response branches hold 24,640 parameters and the two gate generators
8,449 + 4,160, for a total of **37,249** per fusion block. `python examples/check.py`
reproduces these numbers.

## Citation

```bibtex
@article{zhang2026pakan,
  title   = {PAKAN: Pixel-Adaptive Kolmogorov--Arnold Network Modules for Pansharpening},
  author  = {Zhang, Haoyu and Chen, Haojing and Zhong, Zhen and Deng, Liang-Jian},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

MIT. See `LICENSE`.
