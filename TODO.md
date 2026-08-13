# TODO

1. Determine the symbolic--numerical boundary for mapping-tensor normalization.
   Check which quantities can be represented exactly with `Fraction`, including the
   mapping tensors, Gram matrices, matrix inverses, dual tensors, and
   symmetry-adapted embedding tensors. Identify where inverse square roots introduce
   irrational coefficients that cannot be represented by `Fraction`. For quantities
   that must be evaluated numerically, define which operations should use `float64`
   (especially Gram-matrix construction, eigendecomposition, inverse square roots,
   and orthonormality checks) and which results may safely be converted to `float32`
   for storage or runtime use. Establish accuracy criteria and tests for both dtypes.
