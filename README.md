# natto - natural tensor operations


`natto` is a Python package for irreducible Cartesian tensor operations.
It provides reusable operators for extracting irreducible Cartesian tensors (ICTs,
also known as **natural tensors**), from Cartesian tensors.
It also provides operators to embed ICTs back into Cartesian tensor space.

Some other features:
1. dealing with physical tensors with intrinsic symmetry
2. coupling two ICTs into a third
3. building Cartesian harmonics

## Installation

It is recommended to create a virtual environment first (e.g. using `conda` ) and
then do:

```bash
git clone https://github.com/wengroup/natto.git
cd natto
pip install -e .
```

## Example

A rank-2 Cartesian tensor splits into three ICTs, of weight 0, 1 and 2 -- a
scalar, a vector, and a symmetric traceless matrix:

```python
import textwrap

import numpy as np

from natto.mappings import get_reduction

T = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 12.0]
])

operators = get_reduction(rank=2)
for weight, data in operators.items():
    extraction = data["extraction"][0]
    X = np.einsum(extraction["rule"], extraction["numerical"], T)
    print(f"weight {weight}:")
    print(f"  operator: {extraction['symbolic']}")
    print(f"  X{weight}:")
    print(textwrap.indent(str(X), "    "))
```

```
weight 0:
  operator: +1/3 δ_AB
  X0:
    6.0
weight 1:
  operator: +1/2 ε_ABa
  X1:
    [-1.  2. -1.]
weight 2:
  operator: +1/2 δ_Aa δ_Bb  +1/2 δ_Ab δ_Ba  -1/3 δ_AB δ_ab
  X2:
    [[-5.  3.  5.]
     [ 3. -1.  7.]
     [ 5.  7.  6.]]
```

Each operator is printed beside its result, as `natto` built it. `1/3 δ_AB`
takes a third of the trace, `18 / 3`; `1/2 ε_ABa` the antisymmetric part; the
three terms of `X2` symmetrize `T` and subtract that trace.

### Intrinsic symmetry

Declaring a symmetry removes the weights the class cannot carry. A symmetric
rank-2 tensor has no antisymmetric part, so weight 1 is gone, leaving two ICTs:

```python
T = (T + T.T) / 2

operators = get_reduction(rank=2, symmetry="ij=ji")
for weight, data in operators.items():
    extraction = data["extraction"][0]
    X = np.einsum(extraction["rule"], extraction["numerical"], T)
    print(f"weight {weight}:")
    print(f"  operator: {extraction['symbolic']}")
    print(f"  X{weight}:")
    print(textwrap.indent(str(X), "    "))
```

```
weight 0:
  operator: +1/3 δ_AB
  X0:
    6.0
weight 2:
  operator: +1/2 δ_Aa δ_Bb  +1/2 δ_Ab δ_Ba  -1/3 δ_AB δ_ab
  X2:
    [[-5.  3.  5.]
     [ 3. -1.  7.]
     [ 5.  7.  6.]]
```

Weight 1 is absent from the reduction, not present and zero. `X0` and `X2` are
unchanged: symmetrizing removed only what weight 1 carried.

For more examples, see the [`example/`](example/) directory.

## Citation

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
