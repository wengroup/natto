# Notation

This file is the correspondence between the paper and the package. It is
binding: no identifier enters natto that is not in the tables below, and a name
here means the same thing everywhere it appears. Entries marked *pending* are
agreed but not yet applied.

The paper is *Reduction of Cartesian tensors with intrinsic symmetry into
irreducible Cartesian tensors*; equation and section numbers refer to it.

## The rule: words outside, symbols inside

Two registers, and which one applies depends on who reads the name.

**Public names are descriptive words**: function names, parameters, dictionary
keys, and the keys of the generated YAML files. They are read in user code with
no equation nearby, so they have to carry their own meaning. The symbols are
unambiguous in the paper, where the surrounding equation supplies the context.
As bare Python identifiers they carry none, and the package has already been
bitten by that: `H` was at various points the dual mapping tensor, the coupling
operator, and the harmonic operator, and `_tilde` was used for the orthonormal
mappings, which the paper writes with a hat, while the tilde is its dual.

Short symbol aliases such as `get_E` alongside `get_natural_projector` may be
added later. They are deliberately absent for now: a second name for everything
is easy to introduce once the vocabulary has settled and awkward to withdraw
after it has been published.

**Local variables inside a short mathematical routine are the paper's symbols.**
Such a routine is a transcription of an equation and the surrounding lines
supply the context, so `for c, G in zip(row, embedding)` reads as
$\widetilde{\mathbf{G}}^p = \sum_q (\mathbf{g}^{-1})_{pq} \mathbf{G}^q$ while
spelling it out does not. `G_tilde` is the
right local name for a dual: the paper's tilde is the dual, and the confusion it
caused before was that the package used the tilde for the hat.

The paper's symbol also belongs in the docstring, next to the equation that
defines it.

## Operators

| Paper | Meaning | natto |
|---|---|---|
| $\mathbf{E}_{(\ell\mid\ell)}$, Eq. (12) | Natural projector, $\mathcal{T}^\ell \to \mathcal{X}^\ell$ | `get_natural_projector` |
| $\mathbf{F}^p_{n \to \ell}$, Eqs. (14), (16) | Rank-lowering tensor, $\mathcal{T}^n \to \mathcal{T}^\ell$ | not exposed; built inside the embedding operators |
| $\mathbf{G}^p_{(\ell\mid n)}$, Eq. (19) | Mapping tensor, embeds $\mathcal{X}^\ell \to \mathcal{T}^n$ | `embedding` |
| $\widetilde{\mathbf{G}}^p_{(\ell\mid n)}$, Eq. (22) | Dual mapping tensor, extracts $\mathcal{T}^n \to \mathcal{X}^\ell$ | `extraction` |
| $\widehat{\mathbf{G}}^p_{(\ell\mid n)}$, Eq. (26) | Orthonormal mapping tensor, self-dual | `basis="orthonormal"`, not a separate function |
| $\mathbf{Q}^p_{(\ell\mid n)}$, Eq. (36) | Symmetry-adapted mapping tensor | `symmetry=...`, not a separate function |
| $\mathbf{S}^{\ell,p}_n$, Eq. (25) | Embedding of $\mathbf{X}^p_\ell$ in $\mathcal{T}^n$ | `decomposition`; see the note below |
| $\mathbf{K}_{(\ell_3\mid\ell_1,\ell_2)}$, Eqs. (47), (48) | Coupling operator | `get_coupling_operator` |
| $\mathbf{H}_{(n\mid n)}$, Eq. (40) | Harmonic operator | not implemented |
| $\mathbf{V}_n$, Eq. (41) | Cartesian harmonic of weight $n$ | not implemented |

$\mathbf{Q}$ and $\widehat{\mathbf{G}}$ are arguments rather than functions on purpose. Section IV of the
paper makes the argument itself: the Gram matrix, the duals, extraction,
embedding and orthonormalization all carry over "with the $\mathbf{Q}^p$ in
place of the $\mathbf{G}^p$". A symmetry-adapted mapping is a mapping under a constraint, and an
orthonormal one is the same construction in a different basis. Which of them
came back is a fact about the result, not something a caller must know before
calling.

$\mathbf{S}$ needs care: the paper's $\mathbf{S}^{\ell,p}_n$ is a rank-$n$
*tensor*, the weight-$\ell$ part of one particular $\mathbf{T}$, while natto's
is the rank-$2n$ *operator* that produces it,
$\mathbf{S} = \mathbf{G} \odot^\ell \widetilde{\mathbf{G}}$. Same letter, one
the map and one its output.

## Quantities

| Paper | Meaning | natto |
|---|---|---|
| $\ell$ | Weight | `weight` |
| $p$ | Multiplicity index, the paper's "channel" | `p`; the older term "seniority" is retired |
| $n$ | Rank of the Cartesian tensor | `rank` |
| $\mathbf{g}$, $g_{pq}$, Eq. (21) | Gram matrix of the mappings | `gram` |
| $\mathbf{g}^{-1}$ | Its inverse | `gram_inverse` |
| $\mathbf{g}^{-1/2}$ | Its symmetric inverse square root | `gram_inverse_sqrt` |
| $C$, Eq. (49), even $L$ | Coupling normalization | `coeff_C_even` |
| $C$, Eq. (50), odd $L$ | Coupling normalization | `coeff_C_odd` |
| $\mathbf{M}^a$, Eq. (37) | Mixing matrix of generator `a` | `action` matrix |
| $\Pi_a$, $\eta_a$, Eq. (35) | Index permutation and its sign | `permutation`, `sign` |
| $\mathcal{S}$, Sec. IV | Intrinsic symmetry class | `symmetry`, given as `"ijkl=jikl=klij"` |
| $N^{\mathrm{c}}_\ell$ | Number of candidate mappings | not named |
| $N_\ell$ | Multiplicity, the number of channels | `multiplicity` |
| $N^{\mathcal{S}}_\ell$ | Multiplicity within a symmetry class | `multiplicity` |

The paper writes both coupling constants as $C$, its value differing by the
parity of $L$; $D$ appears only in a LaTeX label. `coeff_C_even` and
`coeff_C_odd` follow the text rather than the label.

## Index letters

Within the symbolic layer, indices are letters, lower case for the natural
tensor space and upper case for the Cartesian tensor space.

- `t`, `s`: general tensor, no assumed symmetry
- `u`, `v`: symmetric tensor
- `x`, `y`, `z`: natural tensor, symmetric and traceless
- `a`, `b`, `c`: rank-1 tensors, vectors

## Operations

| Paper | Meaning |
|---|---|
| $\otimes$ | Tensor product |
| $\otimes^n$ | $n$-fold tensor product |
| $\odot^n$ | $n$-fold contraction |
| $\cong$ | Equality under stated conditions |
| $\langle\,\cdot\,\rangle$ | Average over index permutations |

## Retired names

Names that meant something else, or several things, and must not come back.

| Retired | Was | Now |
|---|---|---|
| `H` | the dual mapping tensor, in `GHS` | `get_extraction_operators` |
| `H` | the coupling operator, in `H_tp` | `get_coupling_operator` |
| `H` | an index-ordering flag, `mode="H"` | `mode="extraction"` |
| `H_tp` | module name | `coupling` |
| `GHS` | module name | `mappings`, plus `orthonormal` and `symmetry_adapted` |
| `G_tilde`, `Q_tilde` | the *orthonormal* mappings, i.e. the paper's hat | `basis="orthonormal"`. `G_tilde` is free again, and correct, for a dual |
| `coeff_D` | the odd-parity coupling constant | `coeff_C_odd` |
| `seniority` | the multiplicity index | `p`, "channel" |
| `projector` | the key for `S` | `decomposition` |

## The three operators

The reduction returns three operators per weight and channel, under these keys:

| Key | Paper | Maps |
|---|---|---|
| `embedding` | $\mathbf{G}$ | $\mathbf{X} \to \mathbf{T}'$ |
| `extraction` | $\widetilde{\mathbf{G}}$ | $\mathbf{T} \to \mathbf{X}$ |
| `decomposition` | $\mathbf{S} = \mathbf{G} \odot^\ell \widetilde{\mathbf{G}}$ | $\mathbf{T} \to \mathbf{T}'$, in one step |

`decomposition` is the awkward one. It names the whole reduction in ordinary
use, while each of these is one channel of it, so a set of them reads as several
decompositions. Alternatives were weighed and lost for worse reasons:
`projector` collides with the natural projector, `component` and `part` are too
general, `composition` reads as stoichiometry in a materials context, and
`filter` shadows a builtin.
