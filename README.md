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

The operators depend only on a tensor's rank and its intrinsic symmetry, not on
any particular tensor, so they are built once and then applied to every tensor
of that class. Each arrives with the einsum rule that applies it.

```python
import torch

from natto.mappings import get_reduction
from natto.sym import symmetrize

# The elastic tensor: rank 4, minor and major symmetry.
operators = get_reduction(rank=4, symmetry="ijkl=jikl=klij")

# {0: 2, 2: 2, 4: 1} -- two channels of weight 0, two of weight 2, one of weight 4
print({weight: len(w["embedding"]) for weight, w in operators.items()})

# a random tensor of that class; averaging by hand does not work, since each
# averaging step breaks the symmetry the previous one established
T = symmetrize(torch.randn(3, 3, 3, 3), "ijkl=jikl=klij")

parts = []
for weight, per_weight in operators.items():
    for embedding, extraction in zip(per_weight["embedding"], per_weight["extraction"]):
        # X is the natural tensor of this weight and channel: symmetric and traceless
        X = torch.einsum(extraction["rule"], extraction["numerical"], T)

        # and this is the part of T that it accounts for
        parts.append(torch.einsum(embedding["rule"], embedding["numerical"], X))

# the parts sum back to the tensor they came from
assert torch.allclose(torch.stack(parts).sum(dim=0), T, atol=1e-5)
```

Every rule carries a leading ellipsis, so a batch of tensors of shape
`(..., 3, 3, 3, 3)` works exactly as one does.

Each operator also carries its exact symbolic form under `"symbolic"`, as a
combination of Kronecker deltas and Levi-Civita symbols with rational
coefficients. Nothing in the construction is numerical until it is evaluated.

See [`example/reduce_and_reconstruct.py`](example/reduce_and_reconstruct.py)
for a complete working example.

### An orthonormal basis

`basis="orthonormal"` returns the mappings rotated so that they are orthonormal
under the Cartesian inner product. Those are self-dual: one array both extracts
and embeds, and only the einsum rule tells the two apart.

```python
operators = get_reduction(rank=2, symmetry="ij=ji", basis="orthonormal")

per_weight = operators[2]
embedding, extraction = per_weight["embedding"][0], per_weight["extraction"][0]
assert embedding["numerical"] is extraction["numerical"]
```

The rotation is the inverse square root of the Gram matrix, which is generally
irrational, so these operators have no symbolic form.

### Coupling two natural tensors

`get_coupling_operator` builds the Cartesian counterpart of a Clebsch-Gordan
coefficient, taking natural tensors of weights `l1` and `l2` to the weight-`l3`
part of their product.

```python
from natto.coupling import get_coupling_operator

# two vectors are natural tensors of weight 1; couple them to weight 2
K, rule = get_coupling_operator(l1=1, l2=1, l3=2)

X, Y = torch.randn(3), torch.randn(3)
Z = torch.einsum(rule, K, X, Y)  # shape (3, 3), symmetric and traceless

assert torch.allclose(Z, Z.T, atol=1e-6)
assert abs(Z.trace()) < 1e-6
```

### Cartesian harmonics

`get_harmonic_operator` builds the operator taking the polyadic of a unit vector
to the Cartesian harmonic of a given weight, the Cartesian counterpart of a
spherical harmonic.

```python
from natto.harmonics import get_harmonic_operator

H, rule = get_harmonic_operator(weight=2)

a = torch.randn(3)
a = a / a.norm()
V = torch.einsum(rule, H, a, a)  # symmetric and traceless

# contracting with a second direction gives the Legendre polynomial of the angle
b = torch.randn(3)
b = b / b.norm()
value = torch.einsum("ab,a,b->", V, b, b)
assert torch.allclose(value, torch.special.legendre_polynomial_p(a @ b, 2), atol=1e-6)
```

## Generating operator files

`natto` can generate the coupling operators and the reduction operators for
physical Cartesian tensors, as YAML files.
See the `generate_*.py` scripts in the [`example/`](example/) directory and
[`example/README.md`](example/README.md) for details.


## Citation

`natto` implements the constructions of the following paper; please cite it if you
use the package.

Wen, M., 2026. Reusable Operators for Irreducible Cartesian Tensor Decomposition and Coupling. arXiv preprint arXiv:2609.05971.

```latex
@article{wen2026reusable,
  title   = {Reusable Operators for Irreducible Cartesian Tensor Decomposition and Coupling},
  author  = {Wen, Mingjian},
  journal = {arXiv preprint arXiv:2609.05971},
  year    = {2026},
  doi     = {10.48550/arXiv.2609.05971},
}
```

For the machine-learning models built on these tensors:

Chen, Q., Pattamatta, A.S.L., Wang, B., Srolovitz, D.J. and Wen, M., 2025. Atomistic Machine Learning with Irreducible Cartesian Natural Tensors. arXiv preprint arXiv:2510.04015.

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
