---
kernelspec:
  name: python3
  display_name: Python 3
---

# Extraction and embedding

A Cartesian tensor $\mathbf T$ of rank $n$ can be split into irreducible Cartesian
tensors (ICTs) [@Wen2026Reusable]. An ICT is a tensor that is symmetric and traceless.
An ICT of rank $\ell$, called its *weight*, has $2\ell + 1$ independent components: a
scalar for $\ell = 0$, a vector for $\ell = 1$, a symmetric traceless matrix for
$\ell = 2$, and so on. The weights of the ICTs of $\mathbf T$ run from $0$ to $n$, and
a weight can occur more than once; each occurrence is a *channel*, numbered
$p = 1, \dots, N_\ell$. The tensor is a sum of parts, one for each ICT
$\mathbf X_\ell^p$, and two operators move between the tensor and its ICTs:

- the extraction operator $\widetilde G$ takes the tensor to an ICT,
  $\mathbf X = \widetilde G \cdot \mathbf T$;
- the embedding operator $G$ takes the ICT back to the part of the tensor it carries,
  $G \cdot \mathbf X$.

Summed over all weights and channels, the parts give back $\mathbf T$. `natto` builds
both operators for every weight and channel, keyed by `(ell, p)`.

## Dual basis

By default $G$ are the [mapping tensors](#eq-objects) of the paper and $\widetilde G$
their duals, both exact. We start with a rank-2 tensor and build its operators:

```{code-cell} python
import numpy as np

from natto import act, get_embedding_operators, get_extraction_operators

T = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 12.0]])

G = get_embedding_operators(n=2)
G_tilde = get_extraction_operators(n=2)

print(f"channels (ell, p): {list(G)}")
```

(channels-rank-2)=
`G` and `G_tilde` are dictionaries with one operator per ICT, keyed by `(ell, p)`:
`ell` is the weight of the ICT, and `p` numbers the channels of that weight, counting
from 1. A rank-2 tensor has $3^2 = 9$ components, which split as $1 + 3 + 5$: one
scalar, one vector and one symmetric traceless matrix. So each of the weights 0, 1 and
2 occurs once, and `p` is always 1. At higher rank a weight can occur several times,
as shown [below](#higher-rank).

Each operator prints as a sum of Kronecker deltas and Levi-Civita symbols. Lower-case
letters are the indices of the ICT, and upper-case letters those of the tensor. For
weight 1:

```{code-cell} python
print(f"G~ of weight 1: {G_tilde[1, 1]}")
print(f"G  of weight 1: {G[1, 1]}")
```

(apply-dual)=
`act` applies an operator to its input. $\widetilde G$ takes `T` to its weight-1 ICT, a
vector, and $G$ takes that vector back to the weight-1 part of `T`, its antisymmetric
part:

```{code-cell} python
X = act(G_tilde[1, 1], T)
T_1 = act(G[1, 1], X)

print(f"X, the weight-1 ICT of T:\n{X}\n")
print(f"G · X, the weight-1 part of T:\n{T_1}")
```

(arrays-and-rules)=
### Arrays and einsum rules

Internally, `act` turns the operator into a numerical array and contracts it with the
input by `numpy.einsum`. `evaluate` gives you both pieces: the array, and the einsum
rule that contracts it with the input. That is useful to store the array, or to apply
one operator to many tensors without evaluating it again.

```{code-cell} python
from natto import evaluate

G_tilde_array, rule = evaluate(G_tilde[1, 1])
X = np.einsum(rule, G_tilde_array, T)

print(f"einsum rule: {rule}\n")
print(f"G~ of weight 1 as an array, shape {G_tilde_array.shape}:\n{G_tilde_array}\n")
print(f"X, the weight-1 ICT of T:\n{X}")
```

`X` is the same ICT as [with `act`](#apply-dual). In the rule, `a` is the index of the
ICT and `A`, `B` those of the tensor, in the order of the array's axes. The `...` lets `T` carry leading batch axes: a stack of
tensors of shape `(m, 3, 3)` gives a stack of `m` ICTs.

(higher-rank)=
### Several channels of one weight

A rank-3 tensor has $3^3 = 27$ components, which split as
$1 + 3 \times 3 + 2 \times 5 + 7$: one ICT of weight 0, three of weight 1, two of
weight 2 and one of weight 3. Weight 1 therefore has three channels, `p = 1, 2, 3`, and
weight 2 has two:

```{code-cell} python
G_tilde = get_extraction_operators(n=3)

print(f"channels (ell, p): {list(G_tilde)}")
```

The three channels of weight 1 are three different vectors a rank-3 tensor carries,
roughly one for each pair of its indices that can be contracted away, and each has its
own extraction and embedding operator. Every builder also takes `ell`, to build the
operators of one weight only:

```{code-cell} python
G_tilde = get_extraction_operators(n=3, ell=1)

print(f"channels (ell, p): {list(G_tilde)}")
```

(channel-order)=
### Which mapping is channel `p`

Each candidate mapping contracts some slots of the tensor away, with Kronecker deltas
and, when $n - \ell$ is odd, one Levi-Civita symbol. The candidates are ordered by
their delta pairs, then their Levi-Civita slots, both ascending, and the channels are
the earliest ones that are independent of those before them. For a rank-4 tensor,
channel 1 of weight 2 traces slots 0 and 1 and keeps the ICT on slots 2 and 3.

Within a repeated weight, the choice of channels is a convention. The total
weight-$\ell$ part, $\sum_p \mathbf S^{\ell, p}$, is the same for any choice; the
individual channels are not. [](conventions.md) states the conventions, with the
rank-4 channels of every weight.

## Orthonormal basis

With `basis="orthonormal"`, the mapping tensors of each weight are orthonormalized,
$\widehat G = g^{-1/2} G$ with $g$ their [Gram matrix](#eq-objects). An orthonormal mapping tensor is
its own dual, so the same $\widehat G$ both extracts and embeds; the extraction and
embedding operators of a channel share one array and differ only in their rule. Each
is exact: rational coefficients times an overall factor $\sqrt{s}$, with $s$ an
integer that is 1 when the factor is rational:

```{code-cell} python
G_hat = get_embedding_operators(n=2, basis="orthonormal")
G_hat_tilde = get_extraction_operators(n=2, basis="orthonormal")

X = act(G_hat_tilde[1, 1], T)
T_1 = act(G_hat[1, 1], X)

print(f"X in the orthonormal basis:\n{X}\n")
print(f"G^ · X, the weight-1 part of T:\n{T_1}")
```

Compare with [the dual basis](#apply-dual): the ICT is $\sqrt 2$ times the dual one,
$[-1, 2, -1]$, since the two bases scale the mapping tensor differently, but the part
of `T` it carries is the same antisymmetric matrix.

## Intrinsic symmetry

A tensor with an intrinsic symmetry carries fewer channels, and `symmetry` builds the
operators of its class. The symmetry is written as index equalities: `"ij=ji"` is a
symmetric rank-2 tensor, `"ij=-ji"` an antisymmetric one, and `"ijkl=jikl=klij"` the
elasticity tensor.

A symmetric rank-2 tensor has no antisymmetric part, so of the channels of
[a general rank-2 tensor](#channels-rank-2), weight 1 is gone:

```{code-cell} python
S = (T + T.T) / 2

G_tilde = get_extraction_operators(n=2, symmetry="ij=ji")

print(f"channels (ell, p): {list(G_tilde)}\n")
print(f"G~ of weight 0: {G_tilde[0, 1]}")
print(f"X, the weight-0 ICT of S: {act(G_tilde[0, 1], S)}")
```
