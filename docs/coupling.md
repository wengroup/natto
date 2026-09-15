---
kernelspec:
  name: python3
  display_name: Python 3
---

# Coupling

The coupling operator $K$ takes an ICT $\mathbf X$ of weight $\ell_1$ and an ICT
$\mathbf Y$ of weight $\ell_2$ to the weight-$\ell_3$ part $\mathbf Z$ of their
product, the Cartesian counterpart of a Clebsch-Gordan coefficient. $K$ takes
$\mathbf X$ and $\mathbf Y$ as its two inputs, so $\mathbf Z$ is `act(K, X, Y)`.

Two vectors, ICTs of weight 1, couple to weight 2, a symmetric traceless matrix:

```{code-cell} python
import numpy as np

from natto import act, get_coupling_operator

X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])

K = get_coupling_operator(1, 1, 2)
Z = act(K, X, Y)

print(f"K from weights 1 and 1 to 2: {K}\n")
print(f"Z = K · X Y:\n{Z}")
```

As with the [other operators](#arrays-and-rules), `act` evaluates $K$ into an array and contracts it with
its inputs by `numpy.einsum`. `evaluate` gives the array and the rule, with one operand
for $\mathbf X$ and one for $\mathbf Y$:

```{code-cell} python
from natto import evaluate

K_array, rule = evaluate(K)
Z = np.einsum(rule, K_array, X, Y)

print(f"einsum rule: {rule}\n")
print(f"Z = K · X Y:\n{Z}")
```

By default, `normalize="legendre"`, the scale of $K$ is fixed so that, for even
$\ell_1 + \ell_2 + \ell_3$, the [Cartesian harmonics](harmonics.md) of one direction
couple to the harmonic of that direction. `normalize="none"` leaves $K$ unscaled.
