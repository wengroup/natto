---
kernelspec:
  name: python3
  display_name: Python 3
---

# Cartesian harmonics

The harmonic operator $H$ of rank $n$ takes a unit vector $\mathbf a$, tensored with
itself $n$ times, to its Cartesian harmonic, the Cartesian counterpart of a spherical
harmonic. $H$ takes one copy of $\mathbf a$ for each of its Cartesian indices, so the
weight-2 harmonic is `act(H, a, a)`:

```{code-cell} python
import numpy as np

from natto import act, get_harmonic_operator

a = np.array([0.0, 0.0, 1.0])

H = get_harmonic_operator(2)
Y = act(H, a, a)

print(f"H of weight 2: {H}\n")
print(f"H · a a, the weight-2 harmonic of a:\n{Y}")
```

As with the [other operators](#arrays-and-rules), `act` evaluates $H$ into an array and contracts it with
its inputs by `numpy.einsum`. `evaluate` gives the array and the rule, with one operand
for each copy of $\mathbf a$:

```{code-cell} python
from natto import evaluate

H_array, rule = evaluate(H)
Y = np.einsum(rule, H_array, a, a)

print(f"einsum rule: {rule}\n")
print(f"the weight-2 harmonic of a:\n{Y}")
```

By default, `normalize="legendre"`, $H$ is scaled so that the harmonic of one unit
vector contracted with the tensored copies of another gives the Legendre polynomial
of the angle between them. `normalize="none"` leaves it unscaled.
