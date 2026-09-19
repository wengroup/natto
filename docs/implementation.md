# Implementation notes

`natto` computes the operators of [@Wen2026Reusable], but not always by the paper's
formulas taken literally. This page collects the simplifications and rewritings of the
paper's equations that it uses instead. Equations of the paper are written (W15) for
Eq. 15 of [@Wen2026Reusable].

## Notation

$n$ is the rank of the Cartesian tensor and $\ell$ the weight of an ICT. The
reduction is built from the rank-lowering tensors $F^{(p)}$ (W2)–(W5) and the natural
projector $\mathsf E^{(\ell)}$ (W7), which give the mappings, their Gram matrix and
their duals,

```{math}
:label: eq-objects
G^{(p)} = \mathsf E^{(\ell)} F^{(p)},
\qquad
g_{pq} = \frac{1}{2\ell+1} \langle G^{(p)}, G^{(q)} \rangle,
\qquad
\widetilde G^{(p)} = \sum_q (g^{-1})_{pq}\, G^{(q)} ,
```

(W13), (W15) and (W16), where $\langle \cdot, \cdot \rangle$ contracts every index.
The composed operator $S^{(p)} = G^{(p)} \cdot \widetilde G^{(p)}$ contracts the ICT
indices of a mapping and its dual.

## The natural projector as a sum over matchings

Each term of (W7) is one perfect matching $m$ of the projector's $2\ell$ indices, and
its coefficient depends only on the number $t(m)$ of pairs within each index group:

```{math}
:label: eq-projector-matchings
\mathsf E^{(\ell)} = \sum_{m} \frac{c_{t(m)}}{N_{t(m)}}\, D_m,
\qquad
c_t = -\frac{(\ell-2t+2)(\ell-2t+1)}{2t\,(2\ell-2t+1)}\, c_{t-1},
\quad c_0 = 1,
```

with $D_m$ the product of the matching's deltas and $N_t$ the number of matchings
with that $t$. The recursion is (W S27). The projector has $(2\ell-1)!!$ terms, so a
product of two projectors has $[(2\ell-1)!!]^2$ term pairs, about $10^8$ at
$\ell = 6$. The rewritings below avoid forming such products.

## One projector suffices

The projector is symmetric and idempotent, so one of the two projectors in an inner
product of mappings can be left out:

```{math}
:label: eq-gram-collapse
\langle G^{(p)}, G^{(q)} \rangle = \langle F^{(p)}, G^{(q)} \rangle,
\qquad
g_{pq} = \frac{1}{2\ell+1} \langle F^{(p)}, G^{(q)} \rangle .
```

The same holds for the composed operator, whose tensor indices stay free:

```{math}
:label: eq-composed-collapse
S^{(p)}_{\boldsymbol A \boldsymbol B}
= \sum_q (g^{-1})_{pq}\, F^{(p)}_{\boldsymbol\sigma \boldsymbol A}\, G^{(q)}_{\boldsymbol\sigma \boldsymbol B} .
```

Both cost $(2\ell-1)!!$ products instead of $[(2\ell-1)!!]^2$.

## Mappings as vectors

Every mapping of a weight, whether a candidate, a dual or a symmetry-adapted mapping,
is the projector applied to a rational combination of the same rank-lowering tensors
$F_0, \dots, F_{N-1}$, so it is stored as a coefficient vector
$\boldsymbol c \in \mathbb Q^N$. All inner products then come from one table,
computed once per weight and rank:

```{math}
:label: eq-bilinear-forms
L_{ij} = \langle F_i, \mathsf E\, F_j \rangle,
\qquad
\bigl[\langle \cdot, \cdot \rangle\bigr] = C_1\, L\, C_2^{\mathsf T},
```

where the rows of $C_1$ and $C_2$ are the coefficient vectors of two lists of
mappings. The Gram matrix of any mappings, the adapted Gram matrix
$g^Q = M g M^{\mathsf T}$, and the orthonormal mappings $\widehat G = g^{-1/2} G$ (W21),
computed as in [](#eq-ldl-orthonormal), are all matrix algebra on these vectors.

## Permuted mappings

Permuting the tensor indices of a rank-lowering tensor gives another rank-lowering
tensor, up to the sign of reordering its Levi-Civita symbol:

```{math}
:label: eq-signed-closure
P\, F_i = s_i\, F_{\pi(i)},
\qquad s_i = \pm 1 .
```

So a permuted mapping is another coefficient vector, and the mixing matrix of a
symmetry generator (W30),

```{math}
:label: eq-mixing-matrix
M = g^{-1} O,
\qquad
O_{pq} = \frac{1}{2\ell+1} \langle G^{(p)}, P\, G^{(q)} \rangle ,
```

is read from the table [](#eq-bilinear-forms) with no contraction.

## Contracting products of deltas and Levi-Civita symbols

Every operator is a sum of products of Kronecker deltas and Levi-Civita symbols, so
contracting operators reduces to contracting such products term by term, with no
arrays. Following the repeated indices, a chain of deltas collapses to one delta, a
closed chain is a factor of 3, and a Levi-Civita symbol with a repeated index
vanishes:

```{math}
:label: eq-delta-chain
\delta_{a i_1}\, \delta_{i_1 i_2} \cdots \delta_{i_k b} = \delta_{ab},
\qquad
\delta_{i_1 i_2} \cdots \delta_{i_k i_1} = 3,
\qquad
\varepsilon_{i i c} = 0 .
```

Two Levi-Civita symbols joined by a repeated index are expanded into deltas by their
determinant identity, and the chains are followed again:

```{math}
:label: eq-epsilon-determinant
\varepsilon_{abc}\, \varepsilon_{def}
= \sum_{\pi \in S_3} \operatorname{sgn}(\pi)\, \delta_{a\pi(d)}\, \delta_{b\pi(e)}\, \delta_{c\pi(f)} .
```

Two symbols with no index in common are kept as they are, since expanding them would
turn one product into six.

## Full contraction is cycle counting

Every entry of the table is a full contraction, a number rather than an operator. A
product of deltas in which every index occurs twice is a union of cycles, each a
factor of 3, and with one Levi-Civita symbol on each side the paths between the two
symbols give the sign:

```{math}
:label: eq-cycle-count
\prod_k \delta_{u_k v_k} = 3^{\operatorname{cyc}},
\qquad
\varepsilon_{i_1 i_2 i_3}\, \varepsilon_{j_1 j_2 j_3} \prod_k \delta_{u_k v_k}
= 6\, \operatorname{sgn}(\pi)\, 3^{\operatorname{cyc}} ,
```

where $\pi$ is the bijection the delta paths make between the slots of the two
symbols, and the product is zero if a path joins two slots of the same symbol. The
table is filled by counting cycles, without forming any terms.

## Selecting independent mappings

The number of independent mappings of weight $\ell$ in a rank-$n$ tensor, Table II of
[@Wen2026Reusable], follows from $j \otimes 1 = (j-1) \oplus j \oplus (j+1)$:

```{math}
:label: eq-multiplicity-recursion
N(n, \ell) = N(n-1, \ell+1) + [\ell \ge 1]\, \bigl(N(n-1, \ell-1) + N(n-1, \ell)\bigr),
\qquad N(0, \ell) = \delta_{\ell 0} .
```

The candidates are scanned in order, and the scan stops once $N(n, \ell)$ are kept; the
order, and why some order has to be chosen, are in [](conventions.md). A
candidate with Gram entries $b$ against the kept mappings and $d$ with itself is kept
exactly when

```{math}
:label: eq-schur-residual
d - b^{\mathsf T} g^{-1} b \neq 0 ,
```

with $g^{-1}$ extended by the bordered inverse as candidates are kept. The decision is
exact, where the paper's Algorithm 1 decides by a pivoted QR against a tolerance.

## Orthonormal mappings

The paper orthonormalizes the mappings of a weight with the symmetric inverse square
root of their Gram matrix, $\widehat G = g^{-1/2} G$ (W21). Its entries come from the
eigenvalues of $g$, which are irrational in general, so the orthonormal mappings
could only be held in floating point. `natto` factors the Gram matrix exactly
instead, $g = L D L^{\mathsf T}$ with $L$ unit lower triangular and
$D = \operatorname{diag}(d_1, \dots, d_N)$, both rational, and takes

```{math}
:label: eq-ldl-orthonormal
\widehat G = D^{-1/2} L^{-1} G,
\qquad
\widehat G^p = \frac{1}{\sqrt{d_p}} \sum_{q \le p} (L^{-1})_{pq}\, G^q ,
```

which is Gram-Schmidt in the order of the mappings: $\widehat G^1$ is $G^1$
normalized, $\widehat G^2$ is $G^2$ with its part along $G^1$ removed, and so on.
$L^{-1} G$ is rational, and the only irrational number left is one factor
$1/\sqrt{d_p}$ per channel. With $d_p = a/b$ in lowest terms,
$1/\sqrt{d_p} = \sqrt{ab}/a = (k/a)\sqrt{s}$, where $ab = k^2 s$ and $s$ is
square-free, so $k/a$ joins the rational coefficients and the operator carries
$\sqrt{s}$ as its radicand. For example, $d = 8/3$ gives $\tfrac14 \sqrt 6$.

This is a change of basis, not of the result. The two constructions span the same
space and are related by an orthogonal matrix, so they agree on everything that does
not depend on the basis: the total weight-$\ell$ part, the reconstruction,
$\sum_p \lVert \mathbf X^p_\ell \rVert^2$, and every weight that occurs once. Only
how a repeated weight is divided into channels differs; [](conventions.md) states the
order that fixes it.
