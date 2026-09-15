# Overview

`natto` builds reusable operators for irreducible Cartesian tensors (ICTs, also known
as natural tensors) [@Wen2026Reusable]:

- extraction operators, taking a Cartesian tensor to its ICTs, and embedding
  operators, taking each ICT back;
- the same for physical tensors with intrinsic symmetry;
- coupling operators, coupling two ICTs into a third;
- harmonic operators, building Cartesian harmonics.

Every operator is exact, a sum of products of Kronecker deltas and Levi-Civita symbols
with rational coefficients, and depends only on the rank and symmetry of the tensor,
so it is built once and applied to any tensor of that class.

## Installation

```shell
git clone https://github.com/wengroup/natto.git
cd natto
pip install -e .
```

## Contents

- [](extraction_and_embedding.md): extraction and embedding operators, in the dual
  and orthonormal bases and with intrinsic symmetry.
- [](harmonics.md): Cartesian harmonics of a direction.
- [](coupling.md): coupling two ICTs into a third.
- [](reduction.md): the extraction and embedding operators together.
- [](implementation.md): how `natto` computes the operators of [@Wen2026Reusable].
- [](dev_guide.md): installing from source, code style, and building the docs.
- [](api.md): the public functions.
- [](changelog.md): notable changes.

Other functions, not shown in those pages:

- `get_composed_operators`: each embedding operator
  [composed](#eq-composed-collapse) with its extraction operator, taking a tensor
  straight to its part of a weight and channel.
- `get_gram_matrices`: the exact [Gram matrix](#eq-objects) of the mapping tensors of
  each weight.
- `get_natural_projector`: the [natural projector](#eq-projector-matchings) of a
  weight.
