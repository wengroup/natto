# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Operator.radicand`: an operator may carry an overall factor $\sqrt{s}$, with $s$ a
  square-free integer.

### Changed

- The channels of a repeated weight follow stated conventions; see
  [](conventions.md). This renumbers some channels.
  - Candidate mappings are ordered by their delta pairs, then their Levi-Civita
    slots.
  - Symmetry-adapted mappings are ordered by the independent mappings they contain.
- The orthonormal basis is exact: it is built by an $LDL^{\mathsf T}$ factorization of
  the Gram matrix rather than by $g^{-1/2}$. The channels of a repeated weight are a
  rotation of the previous ones; totals and reconstructions are unchanged.

### Removed

- `natto.orthonormal.OrthonormalOperator` and `get_inverse_square_root`. Orthonormal
  operators are `Operator`s.

## [0.0.1] - yyyy-mm-dd

### Added

- Added feature

### Changed

- Changed API

### Fixed

- Fixed bug
