"""Precompute the reduction operators of named physical tensor classes.

Writes `reduction_operators.yaml` holding, for each class below, the embedding,
extraction and decomposition operators of every weight and channel: the exact
symbolic form of each, the einsum rule that applies it, and its evaluated
values. The operators depend only on rank and symmetry, so a consumer that
always works with the same handful of physical tensors can build this file once
and load it instead of constructing anything.

Edit `physical_tensors` at the bottom to change which classes are written. The
symbolic forms are written with `d` for the Kronecker delta and `e` for the
Levi-Civita symbol, so the file stays plain ASCII.
"""

from pathlib import Path

import torch

from natto.mappings import get_reduction
from natto.utils import yaml_dump

torch.set_default_dtype(torch.float64)


def convert_to_list(tensor):
    """Make one operator entry YAML-safe, recursively.

    Tensors become nested lists, and a symbolic form becomes its string with the
    delta and epsilon characters spelled out in ASCII.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.tolist()
    elif isinstance(tensor, dict):
        if "symbolic" in tensor:
            for k, v in tensor.items():
                if k == "symbolic":
                    tensor[k] = str(v).replace("\u03b4", "d").replace("\u03b5", "e")
                else:
                    tensor[k] = convert_to_list(v)
            return tensor
        else:
            return {k: convert_to_list(v) for k, v in tensor.items()}
    elif isinstance(tensor, list):
        return [convert_to_list(v) for v in tensor]
    else:
        return tensor


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
        out = get_reduction(info["rank"], info["symmetry"], numerical=True)
        info["operators"] = convert_to_list(out)
        results[name] = info

    # Same to yaml
    filename = Path("./reduction_operators.yaml")
    yaml_dump(results, filename)
