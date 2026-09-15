# natto - natural tensor operations


`natto` is a Python package for irreducible Cartesian tensor operations.
It provides reusable operators for extracting irreducible Cartesian tensors (ICTs,
also known as **natural tensors**), from Cartesian tensors.
It also provides operators to embed ICTs back into Cartesian tensor space.

Some other features:
1. dealing with physical tensors with intrinsic symmetry
2. coupling two ICTs into a third
3. building Cartesian harmonics

**Documentation**: https://wengroup.github.io/natto/

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

from natto import act, get_extraction_operators

T = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 12.0]
])

# the extraction operators of every rank-2 tensor, keyed by (weight, channel);
# a weight can occur more than once, each occurrence a channel, and at rank 2
# every weight has exactly one
operators = get_extraction_operators(n=2)

for (weight, p), extraction in operators.items():
    # contract T with the operator to get the ICT of this weight and channel
    X = act(extraction, T)

    print(f"weight {weight}, channel {p}:")
    print(f"  operator: {extraction}")
    print(f"  X{weight}:")
    print(textwrap.indent(str(X), "    "))
    print()
```

```
weight 0, channel 1:
  operator: +1/3 δ_AB
  X0:
    6.0

weight 1, channel 1:
  operator: +1/2 ε_ABa
  X1:
    [-1.  2. -1.]

weight 2, channel 1:
  operator: +1/2 δ_Aa δ_Bb  +1/2 δ_Ab δ_Ba  -1/3 δ_AB δ_ab
  X2:
    [[-5.  3.  5.]
     [ 3. -1.  7.]
     [ 5.  7.  6.]]

```

`X0` is a third of the trace, `X1` the antisymmetric part, and `X2` the symmetric
traceless part.

### Intrinsic symmetry

Declaring a symmetry removes the weights the class cannot carry. A symmetric
rank-2 tensor has no antisymmetric part, so weight 1 is gone, leaving two ICTs:

```python
# a symmetric tensor of the same class
T = (T + T.T) / 2

operators = get_extraction_operators(n=2, symmetry="ij=ji")

for (weight, p), extraction in operators.items():
    X = act(extraction, T)

    print(f"weight {weight}, channel {p}:")
    print(f"  operator: {extraction}")
    print(f"  X{weight}:")
    print(textwrap.indent(str(X), "    "))
    print()
```

```
weight 0, channel 1:
  operator: +1/3 δ_AB
  X0:
    6.0

weight 2, channel 1:
  operator: +1/2 δ_Aa δ_Bb  +1/2 δ_Ab δ_Ba  -1/3 δ_AB δ_ab
  X2:
    [[-5.  3.  5.]
     [ 3. -1.  7.]
     [ 5.  7.  6.]]

```

Weight 1 is absent from the reduction, not present and zero. `X0` and `X2` are
unchanged: symmetrizing removed only what weight 1 carried.

For more, see the [documentation](https://wengroup.github.io/natto/).

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
