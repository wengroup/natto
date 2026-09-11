# Notation

This file holds the naming decisions that the code cannot hold itself: the rule
that governs which register a name is written in, the names that have been
retired and must not come back, and the one place where the paper's symbol and
natto's object are not the same thing.

Everything else lives next to the implementation. Each operator's docstring
names the equation that defines it, so a table of those here would be a second
copy to keep in step -- and this file has already been caught drifting, having
cited a working title the paper no longer carries.

The paper is *Reusable Operators for Irreducible Cartesian Tensor Decomposition
and Coupling* (arXiv:2609.05971); equation and section numbers throughout the
package refer to it.

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

## The one trap

$\mathbf{S}$ needs care: the paper's $\mathbf{S}^{\ell,p}_n$ is a rank-$n$
*tensor*, the weight-$\ell$ part of one particular $\mathbf{T}$, while natto's
is the rank-$2n$ *operator* that produces it,
$\mathbf{S} = \mathbf{G} \odot^\ell \widetilde{\mathbf{G}}$. Same letter, one
the map and one its output.

## Retired names

Names that meant something else, or several things, and must not come back.

| Retired | Was | Now |
|---|---|---|
| `H` | the dual mapping tensor | `extraction`, `get_extraction_operators` |
| `H` | the coupling operator | `get_coupling_operator` |
| `H` | an index-ordering flag, `mode="H"` | `mode="extraction"` |
| `G_tilde`, `Q_tilde` | the *orthonormal* mappings, i.e. the paper's hat | `basis="orthonormal"`. `G_tilde` is free again, and correct, for a dual |
| `coeff_D` | the odd-parity coupling constant | `coeff_C_odd` |
| `seniority` | the multiplicity index | `p`, "channel" |
| `projector` | the key for `S` | `decomposition` |
| `get_G_H_S` | the whole reduction | `get_reduction` |
| `_of_j` | the suffix for one weight | `_of_weight`; `j` is `weight` and `n` is `rank` throughout |
| `H_tp` | module | `coupling` |
| `GHS` | module | `mappings`, plus `orthonormal` and `symmetry_adaptation` |
| `EGH` | module | `lowering`, `natural_projector`, `mappings` and `gram` |
| `operators` | module holding four separate things | split, as above |
| `ops` | module | `algebra` |
| `sym` | module | `intrinsic_symmetry`; the permutation symmetry a tensor already has |
| `symmetry_adapted` | module | `symmetry_adaptation`, so it no longer reads as a half of `intrinsic_symmetry` |
| `symmetrize` | module | `symmetric_traceless`; the numerical route to an ICT |
| `symmetrize` | *two* functions of that name, in those two modules | `intrinsic_symmetry.impose_symmetry` and `symmetric_traceless.symmetrize` |
| `matrix` | module | `rational`; it is exact arithmetic, not numpy |
| `get_E_rules` | index patterns of the natural projector | `get_projector_rules` |
| `get_G_rules_*` | index patterns of the rank-lowering tensors | `get_lowering_rules_*` |
| `get_G_even`, `get_G_odd` | the candidate mapping tensors | `get_mappings_even`, `get_mappings_odd` |
| `get_S` | the decomposition operators | `get_decomposition_operators` |
| `contract_G` | contraction of two mappings | `contract_mappings` |

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
