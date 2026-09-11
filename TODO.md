# TODO

_The symbolic--numerical boundary question is settled; see "Exactness" below._

## Exactness

Everything that can be exact is, and is:

- the mapping tensors, which are built from Kronecker deltas and Levi-Civita
  symbols with `Fraction` coefficients;
- the Gram matrix and its inverse, contracted and inverted over the rationals;
- the selection of an independent set of mappings, by
  `qr.select_independent_mappings`: it borders the kept set's Gram matrix with each
  candidate in turn and asks whether the result is singular. That is the residual
  test of Gram-Schmidt made exact, so the selection is a pure function of the
  weight and the rank -- no tolerance, and no random tensor to probe the mappings
  with. `qr` keeps two numerical schemes alongside it, one on the mappings' full
  components and one on their action on a probe, as cross-checks;
- the dual mapping tensors, being rational combinations of the mappings;
- the mixing matrices of the intrinsic-symmetry generators, contracted
  symbolically;
- the null space those mixing matrices define, by Gaussian elimination over the
  rationals, and hence the symmetry-adapted mapping tensors.

So the whole reduction, up to and including the symmetry adaptation, carries no
numerical tolerance.

One place is irreducibly numerical:

- **The orthonormal mapping tensors.** They are `g^(-1/2) G`, and an inverse
  square root is irrational in general, so no exact representation exists.
  Computed in float64 by eigendecomposition.

Evaluated operators are float64. That is storage rather than construction: the
symbolic form above is exact whatever it is later evaluated into.

## Open

Nothing outstanding.
