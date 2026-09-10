# TODO

_The symbolic--numerical boundary question is settled; see "Exactness" below._

## Exactness

Everything that can be exact is, and is:

- the mapping tensors, which are built from Kronecker deltas and Levi-Civita
  symbols with `Fraction` coefficients;
- the Gram matrix and its inverse, contracted and inverted over the rationals;
- the dual mapping tensors, being rational combinations of the mappings;
- the mixing matrices of the intrinsic-symmetry generators, contracted
  symbolically;
- the null space those mixing matrices define, by Gaussian elimination over the
  rationals, and hence the symmetry-adapted mapping tensors.

So the whole reduction, up to and including the symmetry adaptation, carries no
numerical tolerance.

Two places are irreducibly numerical:

- **The orthonormal mapping tensors.** They are `g^(-1/2) G`, and an inverse
  square root is irrational in general, so no exact representation exists.
  Computed in float64 by eigendecomposition.
- **The selection of an independent set.** `qr` decides linear independence by a
  residual norm against a tolerance, on tensors evaluated numerically. Making
  this exact would mean rank-revealing elimination over the rationals; it is
  open, and tracked below.

Evaluated operators are float32 by default, following `torch.get_default_dtype`,
and float64 where a caller asks. That is storage rather than construction: the
symbolic form above is exact whatever it is later evaluated into.

## Open

1. Make the selection of an independent set exact, or decide deliberately not
   to. It is the last tolerance in the construction: `qr.find_independent_tensors`
   applies a residual-norm threshold to the embeddings of one fixed-seed random
   natural tensor. Two questions, in order: whether independence should be
   decided on the mapping tensors themselves rather than on their action on a
   random probe, and whether the decision should be made over the rationals.
   Note that changing either changes which subset is canonical, and so changes
   every stored operator downstream.

2. Establish accuracy criteria for float32 storage. The evaluated operators are
   float32 unless a caller asks otherwise, which is enough for the reductions
   but has never been checked against a stated bound.
