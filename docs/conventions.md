# Conventions

When a weight occurs more than once in a tensor, the reduction has to decide which
mappings serve as that weight's channels, and in which order they are numbered as
`(ell, p)`. The mathematics does not decide it: any basis of the weight's mapping
space extracts the same total weight-$\ell$ part, and in general no basis is
preferred over the others. `natto` therefore fixes the choice by conventions, one for
each of the four steps that build the channels, and this page states them. Because
every step follows a stated rule, the same call gives the same channels on every run
and every machine, and a reader can say in advance which mapping is channel `p`.

The four steps are:

1. [order the candidate mappings](#conv-order),
2. [select the independent ones](#conv-select),
3. [adapt them to an intrinsic symmetry](#conv-symmetry), if one is given,
4. [orthonormalize them](#conv-orthonormal), for `basis="orthonormal"`.

Each step is fixed by the output of the one before, so the conventions together
determine every channel. The examples use slots numbered from 0.

(conv-order)=
## 1. Ordering the candidates

Each candidate mapping is the natural projector applied to a rank-lowering tensor,
which contracts some slots of the tensor away: pairs of slots with Kronecker deltas
and, when $n - \ell$ is odd, two slots with a Levi-Civita symbol
$\varepsilon_{\tau ij}$ whose $\tau$ index goes to the ICT. The candidates are
ordered

> by their delta pairs, then their Levi-Civita slots, both ascending,

so the first candidate contracts the earliest slots. In the code this is the tuple
order of `LoweringLabel`, `(deltas, epsilon)`, and `get_lowering_labels` returns the
labels sorted by it. For a rank-4 tensor:

| weight | candidates, in order |
|---|---|
| 0 | $\delta_{01}\delta_{23}$, $\delta_{02}\delta_{13}$, $\delta_{03}\delta_{12}$ |
| 1 | $\delta_{01}\varepsilon_{\tau 23}$, $\delta_{02}\varepsilon_{\tau 13}$, $\delta_{03}\varepsilon_{\tau 12}$, $\delta_{12}\varepsilon_{\tau 03}$, $\delta_{13}\varepsilon_{\tau 02}$, $\delta_{23}\varepsilon_{\tau 01}$ |
| 2 | $\delta_{01}$, $\delta_{02}$, $\delta_{03}$, $\delta_{12}$, $\delta_{13}$, $\delta_{23}$ |
| 3 | $\varepsilon_{\tau 01}$, $\varepsilon_{\tau 02}$, $\varepsilon_{\tau 03}$, $\varepsilon_{\tau 12}$, $\varepsilon_{\tau 13}$, $\varepsilon_{\tau 23}$ |
| 4 | nothing contracted |

At weight 2 the first candidate traces slots 0 and 1 and keeps the ICT on slots 2
and 3. Every candidate is enumerated before any is selected, so the rule costs one
sort, and nothing depends on the order in which the labels happen to be generated.

Each candidate is taken with coefficient 1, and its Levi-Civita symbol with the $\tau$
index first and the two slots increasing, $\varepsilon_{\tau ij}$ with $i < j$; at
weight 0, where the symbol takes three slots, $\varepsilon_{ijk}$ with $i < j < k$.
The opposite orientation would flip the sign of the candidate, and of every channel
built from it.

(conv-select)=
## 2. Selecting the independent mappings

The candidates of a weight are generally not independent. At weight 2 and rank 3
there are three, with the Levi-Civita symbol on slots $(0,1)$, $(0,2)$ and $(1,2)$,
and they satisfy

$$
G^1 - G^2 + G^3 = 0 .
$$

The candidates are scanned in order, and each is kept when it is independent of those
already kept; the scan stops at the multiplicity of the weight. Independence is
decided exactly, over the rationals, by the Schur complement of
[the bordered Gram matrix](#eq-schur-residual). The channels are the kept candidates,
in the order they were kept, so `(ell, p)` is the $p$-th of them.

Independent sets form a matroid, so scanning a fixed total order keeps its
lexicographically first basis: a definite set, whichever exact test decides
independence along the scan. For a rank-4 tensor, weight 3 is the only weight with a
dependent candidate:

| weight | kept | dropped |
|---|---|---|
| 0, 1, 2 | all | none |
| 3 | $\varepsilon_{\tau 01}$, $\varepsilon_{\tau 02}$, $\varepsilon_{\tau 03}$ | $\varepsilon_{\tau 12}$, $\varepsilon_{\tau 13}$, $\varepsilon_{\tau 23}$ |

Dropping a dependent candidate discards no information: it lies in the span of those
kept, so every combination of all the candidates is a combination of the kept ones.
At weight 2 and rank 3, $a G^1 + b G^2 + c G^3 = (a - c) G^1 + (b + c) G^2$.

Some convention cannot be avoided. Combining all the candidates could make the basis
treat the slots evenly, but at weight 2 and rank 3 no basis does: permuting the three
slots permutes the three candidates up to sign, and on their two-dimensional span this
is the two-dimensional irreducible representation of $S_3$, which fixes no basis. The
conventions only decide where the arbitrariness goes.

The numerical selections, `selection="qr"`, `"components"` and `"embeddings"`, scan
the same ordered candidates but may keep a different subset. Only the default,
`"symbolic"`, defines the channels described here.

(conv-symmetry)=
## 3. Adapting to an intrinsic symmetry

With a `symmetry`, the kept mappings $G^1, \dots, G^N$ are mixed into the
symmetry-adapted mappings $Q^k = \sum_q c^k_q G^q$. Each generator of the symmetry
acts on the $G^q$ through a [mixing matrix](#eq-mixing-matrix) $M^a$, and the
coefficient vectors $c$ are the null space of the stacked rows of $M^a - \eta_a I$,
with $\eta_a$ the generator's sign.

That null space is the set of combinations the symmetry leaves invariant, so it
depends on the symmetry, not on the generators that happen to express it or their
order. Its basis is read off a full reduced row echelon form over the rationals, which
is unique for a given null space and column order, the columns being the $G^q$ in the
order of step 2. The $Q^k$ are then ordered

> by the mappings $G^q$ they contain, compared in turn: the first, then the second,
> and so on,

so $G^1 + G^2 + G^5$ comes before $G^1 + G^3 + G^4$. Only which mappings appear
decides the order, not their coefficients, and no two $Q^k$ contain the same ones, so
there are no ties. Four examples:

| tensor | weight | $G$, in order | $Q$ |
|---|---|---|---|
| elastic, `ijkl=jikl=klij` | 0 | $\delta_{01}\delta_{23}$, $\delta_{02}\delta_{13}$, $\delta_{03}\delta_{12}$ | $Q^1 = G^1$, $Q^2 = G^2 + G^3$ |
| piezoelectric, `ijk=ikj` | 1 | $\delta_{01}$, $\delta_{02}$, $\delta_{12}$ | $Q^1 = G^1 + G^2$, $Q^2 = G^3$ |
| elastic, `ijkl=jikl=klij` | 2 | $\delta_{01}$, $\delta_{02}$, $\delta_{03}$, $\delta_{12}$, $\delta_{13}$, $\delta_{23}$ | $Q^1 = G^1 + G^6$, $Q^2 = G^2 + G^3 + G^4 + G^5$ |
| photoelastic, `ijkl=jikl` | 3 | $\varepsilon_{\tau 01}$, $\varepsilon_{\tau 02}$, $\varepsilon_{\tau 03}$ | $Q^1 = G^2 - \tfrac12 G^1$, $Q^2 = G^3 - \tfrac12 G^1$ |

In the first two, the minor symmetry exchanges $G^2$ and $G^3$, and $G^1$ and $G^2$,
respectively. In the last, both $Q^k$ contain $G^1$, and the second mapping they
contain, $G^2$ or $G^3$, decides.

Each $Q^k$ is scaled so that the last mapping it contains has coefficient 1, as the
reduced row echelon form gives it: $Q^1 = G^2 - \tfrac12 G^1$ above has coefficient 1
on $G^2$. Another scale would span the same space, but the channel $\mathbf X^k$ of
the dual basis would change by the inverse factor.

(conv-orthonormal)=
## 4. Orthonormalizing

With `basis="orthonormal"`, the mappings of step 2 or 3 are orthonormalized. The
paper writes this as $\widehat G = g^{-1/2} G$ (W21). For a weight that occurs once
that fixes the channel up to sign, but for a repeated weight it is one orthonormal
basis among many, and `natto` takes the one that follows the order of steps 1 to 3:

> channel 1 is the first mapping normalized, channel 2 is the second with its part
> along channel 1 removed, normalized, and so on,

which is Gram-Schmidt in the order of the mappings; the implementation notes give it
as [an exact factorization of $g$](#eq-ldl-orthonormal). Both bases span the same space
and are related by a rotation. For the piezoelectric tensor at weight 1, with the
Gram matrix of $Q^1, Q^2$ being $\begin{pmatrix} 8 & 2 \\ 2 & 3 \end{pmatrix}$,

$$
\widehat G^1 = \frac{1}{\sqrt 8}\, Q^1,
\qquad
\widehat G^2 = \sqrt{\tfrac{2}{5}}\, \bigl( Q^2 - \tfrac14 Q^1 \bigr) .
$$

Each $\widehat G^p$ has a positive coefficient on its own $Q^p$, as in the example,
which fixes its sign. Reordering the mappings changes the result, which is why the
order of steps 1 to 3 matters, but rescaling one only flips the sign of its channel if
the factor is negative.

## What the conventions affect

Only how a repeated weight is divided into channels. These do not depend on any of
them:

- the total weight-$\ell$ part $\sum_p \mathbf S^{\ell, p}$, and so the reconstruction
  of $\mathbf T$,
- $\sum_p \lVert \mathbf X^p_\ell \rVert^2$ in the orthonormal basis,
- the part $\mathbf S^{\ell, 1}$ of a weight that occurs once, whose mapping is fixed
  up to scale.

The individual channels $\mathbf X^p_\ell$ and $\mathbf S^{\ell, p}$ of a repeated
weight do depend on them, and so does $\sum_p \lVert \mathbf X^p_\ell \rVert^2$ in the
dual basis, which is not an isometry. So does the ICT $\mathbf X^1_\ell$ of a weight
that occurs once, through the scale and sign conventions: by a factor in the dual
basis, and by a sign in the orthonormal one.
