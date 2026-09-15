---
kernelspec:
  name: python3
  display_name: Python 3
---

# Reduction

`get_reduction` builds the operators of [](extraction_and_embedding.md) for every
weight and channel in one call. For each `(ell, p)` it gives the embedding operator
$G$ under `"embedding"` and the extraction operator $\widetilde G$ under
`"extraction"`, and it takes the same `ell`, `symmetry` and `basis` arguments.

Reducing a rank-2 tensor extracts each ICT and embeds it back into the part of the
tensor it carries:

```{code-cell} python
import numpy as np

from natto import act, evaluate, get_reduction

T = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 12.0]])

reduction = get_reduction(n=2)

for (ell, p), operators in reduction.items():
    G = operators["embedding"]
    G_tilde = operators["extraction"]

    X = act(G_tilde, T)
    T_part = act(G, X)

    print(f"--- weight {ell}, channel {p} ---")
    print(f"G~: {G_tilde}")
    print(f"G:  {G}")
    print(f"X, the ICT:\n{X}")
    print(f"G · X, the part of T:\n{T_part}\n")
```

The three parts are $\operatorname{tr}\mathbf T/3$ times the identity, the
antisymmetric part and the symmetric traceless part, and they sum to `T`.

The same reduction with [arrays and einsum rules](#arrays-and-rules), each operator
evaluated once:

```{code-cell} python
for (ell, p), operators in reduction.items():
    G_array, G_rule = evaluate(operators["embedding"])
    G_tilde_array, G_tilde_rule = evaluate(operators["extraction"])

    X = np.einsum(G_tilde_rule, G_tilde_array, T)
    T_part = np.einsum(G_rule, G_array, X)

    print(f"--- weight {ell}, channel {p} ---")
    print(f"rule of G~: {G_tilde_rule}")
    print(f"rule of G:  {G_rule}")
    print(f"G · X, the part of T:\n{T_part}\n")
```
