# Notation

This file is the correspondence between the paper and the package. It is
binding: no identifier enters natto that is not in the tables below, and a name
here means the same thing everywhere it appears.

The paper is *Reduction of Cartesian tensors with intrinsic symmetry into
irreducible Cartesian tensors*; equation and section numbers refer to it.

## The rule: words, not symbols

Public names are descriptive words, not the paper's symbols. The symbols are
unambiguous in the paper, where the surrounding equation supplies the context.
As bare Python identifiers they carry none, and the package has already been
bitten by that: `H` was at various points the dual mapping tensor, the coupling
operator, and the harmonic operator, and `_tilde` was used for the orthonormal
mappings, which the paper writes with a hat, while the tilde is its dual.

Short symbol aliases such as `get_E` alongside `get_natural_projector` may be
added later. They are deliberately absent for now: a second name for everything
is easy to introduce once the vocabulary has settled and awkward to withdraw
after it has been published.

The paper's symbol belongs in the docstring, next to the equation that defines
it, where the context is restored.

## Operators

| Paper | Meaning | natto |
|---|---|---|
| `E_(l\|l)`, Eq. (12) | Natural projector, `T^l -> X^l` | `get_natural_projector` |
| `F^p_(n->l)`, Eqs. (14), (16) | Rank-lowering tensor, `T^n -> T^l` | not exposed; built inside the embedding operators |
| `G^p_(l\|n)`, Eq. (19) | Mapping tensor, embeds `X^l -> T^n` | `get_embedding_operators` |
| `G~^p_(l\|n)`, Eq. (22) | Dual mapping tensor, extracts `T^n -> X^l` | `get_extraction_operators` |
| `G^^p_(l\|n)`, Eq. (26) | Orthonormal mapping tensor, self-dual | `basis="orthonormal"`, not a separate function |
| `Q^p_(l\|n)`, Eq. (36) | Symmetry-adapted mapping tensor | `symmetry=...`, not a separate function |
| `S^(l,p)_n`, Eq. (25) | Embedding of `X^p_l` in `T^n` | see the note below |
| `K_(l3\|l1,l2)`, Eqs. (47), (48) | Coupling operator | `get_coupling_operator` |
| `H_(n\|n)`, Eq. (40) | Harmonic operator | not implemented |
| `V_n`, Eq. (41) | Cartesian harmonic of weight `n` | not implemented |

`Q` and `G^^` are arguments rather than functions on purpose. Section IV of the
paper makes the argument itself: the Gram matrix, the duals, extraction,
embedding and orthonormalization all carry over "with the `Q^p` in place of the
`G^p`". A symmetry-adapted mapping is a mapping under a constraint, and an
orthonormal one is the same construction in a different basis. Which of them
came back is a fact about the result, not something a caller must know before
calling.

`S` needs care: the paper's `S^(l,p)_n` is a rank-`n` *tensor*, the weight-`l`
part of one particular `T`, while natto's `S` is the rank-`2n` *operator* that
produces it, `S = G . G~`. Same letter, one the map and one its output.

## Quantities

| Paper | Meaning | natto |
|---|---|---|
| `l` | Weight | `weight` |
| `p` | Multiplicity index, the paper's "channel" | `p`; the older term "seniority" is retired |
| `n` | Rank of the Cartesian tensor | `rank` |
| `g`, `g_pq`, Eq. (21) | Gram matrix of the mappings | `gram` |
| `g^-1` | Its inverse | `gram_inverse` |
| `g^(-1/2)` | Its symmetric inverse square root | `gram_inverse_sqrt` |
| `C`, Eq. (49) even `L` | Coupling normalization | `coeff_C_even` |
| `C`, Eq. (50) odd `L` | Coupling normalization | `coeff_C_odd` |
| `M^a`, Eq. (37) | Mixing matrix of generator `a` | `action` matrix |
| `Pi_a`, `eta_a`, Eq. (35) | Index permutation and its sign | `permutation`, `sign` |
| `S` (calligraphic), Sec. IV | Intrinsic symmetry class | `symmetry`, given as `"ijkl=jikl=klij"` |
| `N^c_l` | Number of candidate mappings | not named |
| `N_l` | Multiplicity, the number of channels | `multiplicity` |
| `N^S_l` | Multiplicity within a symmetry class | `multiplicity` |

The paper writes both coupling constants as `C`, its value differing by the
parity of `L`; `D` appears only in a LaTeX label. `coeff_C_even` and
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
| `(x)` | Tensor product |
| `(x)^n` | `n`-fold tensor product |
| `.^n` | `n`-fold contraction |
| `~=` | Equality under stated conditions |
| `<.>` | Average over index permutations |

## Retired names

Names that meant something else, or several things, and must not come back.

| Retired | Was | Now |
|---|---|---|
| `H` | the dual mapping tensor, in `GHS` | `get_extraction_operators` |
| `H` | the coupling operator, in `H_tp` | `get_coupling_operator` |
| `H_tp` | module name | `coupling` |
| `G_tilde`, `Q_tilde` | the orthonormal mappings, the paper's hat | `basis="orthonormal"` |
| `coeff_D` | the odd-parity coupling constant | `coeff_C_odd` |
| `seniority` | the multiplicity index | `p`, "channel" |
