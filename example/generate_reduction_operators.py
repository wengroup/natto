"""Precompute the reduction operators of named physical tensor classes.

Writes `reduction_operators.yaml` holding, for each class below, the embedding
and extraction operators of every weight and channel: the exact
symbolic form of each, the einsum rule that applies it, and its evaluated
values. The operators depend only on rank and symmetry, so a consumer that
always works with the same handful of physical tensors can build this file once
and load it instead of constructing anything.

Edit `physical_tensors` at the bottom to change which classes are written. The
symbolic forms are written with `d` for the Kronecker delta and `e` for the
Levi-Civita symbol, so the file stays plain ASCII.
"""

from pathlib import Path

from natto import evaluate, get_gram_matrices, get_reduction
from natto.rational import float_matrix, fraction_matrix, matrix_inverse
from natto.utils import yaml_dump


def get_operator_entry(operator) -> dict:
    """One operator as the file stores it: symbolic form, einsum rule and values."""
    array, rule = evaluate(operator)

    return {
        "symbolic": operator.to_string(ascii=True),
        "rule": rule,
        "numerical": array.tolist(),
    }


def get_reduction_entries(rank: int, symmetry: str) -> dict:
    """The operators of one class, grouped by weight as the file lays them out.

    Each weight holds a list of embedding and a list of extraction operators, one
    per channel in channel order, and its Gram matrix with the inverse.
    """
    entries = {}
    for weight, gram in get_gram_matrices(rank, symmetry=symmetry).items():
        gram_inverse = matrix_inverse(gram)
        entries[weight] = {
            "embedding": [],
            "extraction": [],
            "gram": {
                "symbolic": fraction_matrix(gram),
                "numerical": float_matrix(gram),
            },
            "gram_inverse": {
                "symbolic": fraction_matrix(gram_inverse),
                "numerical": float_matrix(gram_inverse),
            },
        }

    for (weight, _), operators in get_reduction(rank, symmetry=symmetry).items():
        for key in ("embedding", "extraction"):
            entries[weight][key].append(get_operator_entry(operators[key]))

    return entries


if __name__ == "__main__":
    physical_tensors = {
        "nuclear_shielding": {
            "rank": 2,
            "symmetry": "ij",
        },
        "polarizability": {
            "rank": 2,
            "symmetry": "ij=ji",
        },
        "piezoelectricity": {
            "rank": 3,
            "symmetry": "ijk=ikj",
        },
        "elasticity": {
            "rank": 4,
            "symmetry": "ijkl=jikl=klij",
        },
    }

    results = {}
    for name, info in physical_tensors.items():
        info["operators"] = get_reduction_entries(info["rank"], info["symmetry"])
        results[name] = info

    # Same to yaml
    filename = Path("./reduction_operators.yaml")
    yaml_dump(results, filename)
