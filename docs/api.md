# API

Everything below is imported from the top-level package, e.g.
`from natto import get_extraction_operators`.

## Operators

Every operator builder returns exact operators: sums of products of Kronecker deltas
and Levi-Civita symbols with rational coefficients, which print in that notation. The
one exception is the orthonormal basis, whose coefficients are floats.
`get_gram_matrices` returns exact rational matrices instead. The reduction builders
key their operators by `(ell, p)`, the weight and the channel of that weight.

| function | returns | see |
|---|---|---|
| `get_extraction_operators(n, ell=None, symmetry=None, basis="dual")` | the operators taking a rank-`n` tensor to its ICTs | [](extraction_and_embedding.md) |
| `get_embedding_operators(n, ell=None, symmetry=None, basis="dual")` | the operators taking each ICT back to its part of the tensor | [](extraction_and_embedding.md) |
| `get_reduction(n, ell=None, symmetry=None, basis="dual")` | both, as `{(ell, p): {"embedding": ..., "extraction": ...}}` | [](reduction.md) |
| `get_composed_operators(n, ell=None, symmetry=None)` | the two composed, a rank-`2n` operator per channel | [composed operator](#eq-composed-collapse) |
| `get_gram_matrices(n, ell=None, symmetry=None)` | the exact Gram matrix of each weight, keyed by `ell` | [Gram matrix](#eq-objects) |
| `get_natural_projector(ell)` | the natural projector of weight `ell` | [natural projector](#eq-projector-matchings) |
| `get_coupling_operator(l1, l2, l3, normalize="legendre")` | the coupling operator of a weight triple | [](coupling.md) |
| `get_harmonic_operator(n, normalize="legendre")` | the harmonic operator of rank `n` | [](harmonics.md) |

The arguments shared by the reduction builders:

- `n`: the rank of the Cartesian tensor.
- `ell`: the one weight to build, or `None` for every weight.
- `symmetry`: the intrinsic symmetry, as index equalities such as `"ijkl=jikl=klij"`,
  or `None`.
- `basis`: `"dual"` for the mappings and their exact duals, or `"orthonormal"` for the
  self-dual basis.

## Applying operators

| function | returns | see |
|---|---|---|
| `evaluate(operator)` | `(array, rule)`: the operator as an array, with the einsum rule that applies it | [arrays and einsum rules](#arrays-and-rules) |
| `act(operator, *arrays)` | the operator applied to one array per input group | [arrays and einsum rules](#arrays-and-rules) |

The array has its output indices first, and the rule contracts one array into each
input group of the operator, with a leading ellipsis on every operand for batches:
`act(operator, *arrays)` is `numpy.einsum(rule, array, *arrays)`. The inputs are the
tensor of an extraction operator and of the natural projector, the ICT of an embedding
operator, `X` and `Y` of a coupling operator, and each copy of the unit vector of a
harmonic operator.
