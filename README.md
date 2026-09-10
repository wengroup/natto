# natto

`natto` is a Python package for constructing, analyzing, and embedding **natural tensors** — symmetric traceless tensors that form irreducible representations of the rotation group O(3).
It provides tools to decompose arbitrary Cartesian tensors into natural tensor components and to embed natural tensors back into the original Cartesian tensor space.

## Installation

It is recommended to create a virtual environment first (e.g. using conda) before installing `natto`:

```bash
git clone https://github.com/wengroup/natto.git
cd natto
pip install -e .
```

## Example

The example below converts a rank-2 Cartesian tensor into its natural tensor components and reconstructs the original tensor.

```python
import torch
from natto.mappings import get_reduction

# Create a rank-2 tensor
T = torch.arange(9, dtype=torch.float).reshape(3, 3)

# Get the operators for the reduction
rank = 2
symmetry=None # `None` means no additional symmetry; `ij=ji` means symmetric tensors...
output = get_reduction(rank, symmetry)

all_T_prime = []
for j, out_j in output.items():
    for p, (extraction, embedding) in enumerate(
        zip(out_j["extraction"], out_j["embedding"])
    ):
        # Extract the natural tensor of this weight and channel (symmetric traceless)
        X = torch.einsum(extraction["rule"], extraction["numerical"], T)

        # Embed it back: the weight-j, channel-p part of T
        T_prime = torch.einsum(embedding["rule"], embedding["numerical"], X)
        all_T_prime.append(T_prime)

        print('-'*10 + f"j={j}, p={p}"+'-'*10)
        print(f"X=\n{X}")
        print(f"T_prime=\n{T_prime}")

# Sum of all embedded components should reconstruct the original tensor
T_sum = torch.stack(all_T_prime).sum(dim=0)
print(f"Original T=\n{T}")
print(f"Sum of T_prime=\n{T_sum}")

assert torch.allclose(T_sum, T, atol=1e-6)
```

See [`example/get_natural_tensor.py`](example/get_natural_tensor.py) for a complete working example.

## Generating projectors

`natto` can be used to generate three types of projectors: unit vector projectors, tensor product projectors, and decomposition/reconstruction projectors for physical Cartesian tensors.
See the `generate_*.py` scripts in the [`example/`](example/) directory and [`example/README.md`](example/README.md) for details.


## Citation
Chen, Q., Pattamatta, A.S.L., Wang, B., Srolovitz, D.J. and Wen, M., 2026. Atomistic Machine Learning with Irreducible Cartesian Natural Tensors. arXiv preprint arXiv:2510.04015.

```latex
@article{chen2026atomistic,
  title   = {Atomistic Machine Learning with Irreducible Cartesian Natural Tensors},
  author  = {Chen, Qun and Pattamatta, ASL and Wang, Boyu and Srolovitz, David J and Wen, 
  Mingjian},
  journal = {arXiv preprint arXiv:2510.04015},
  year    = {2025},
  doi     = {10.48550/arXiv.2510.04015},
}
```
