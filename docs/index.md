# natto

`natto` builds reusable operators for irreducible Cartesian tensors (ICTs, also known
as natural tensors) [@Wen2026Reusable]:

- extraction operators, taking a Cartesian tensor to its ICTs, and embedding
  operators, taking each ICT back;
- the same for physical tensors with intrinsic symmetry;
- coupling operators, coupling two ICTs into a third;
- harmonic operators, building Cartesian harmonics.

Every operator is exact, a sum of products of Kronecker deltas and Levi-Civita symbols
with rational coefficients, and depends only on the rank and symmetry of the tensor,
so it is built once and applied to any tensor of that class.

To get started, see the [](overview.md) for installation and a guide to the pages.

## Citation

```bibtex
@article{wen2026reusable,
  title   = {Reusable Operators for Irreducible Cartesian Tensor Decomposition and Coupling},
  author  = {Wen, Mingjian},
  journal = {arXiv preprint arXiv:2609.05971},
  year    = {2026},
  doi     = {10.48550/arXiv.2609.05971},
}
```
