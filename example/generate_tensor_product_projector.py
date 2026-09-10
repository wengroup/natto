r"""
Generate the coupling operator K and its einsum rule, for Z = X \otimes Y.

Z_l3 can be calculated as Z = einsum(rule, K, X, Y).
"""

from pathlib import Path

import torch
from torch import Tensor

from natto.coupling import get_coupling_operator, get_coupling_symbolic
from natto.ops import simplify_linear_combination
from natto.utils import yaml_dump


def generate_coupling_operators(
    max_l1: int, max_l2: int, max_l3: int, dtype=torch.float64
) -> dict[str, dict[str, Tensor]]:
    """
    Generate coupling operators and the corresponding einsum rules.

    Args:
        max_l1: Maximum weight of X.
        max_l2: Maximum weight of Y.
        max_l3: Maximum weight of Z.
        dtype: The data type of the generated tensors. Default is torch.float64 for
            higher precision.

    Return:
        Coupling operators and rules,
        {l1-l2-l3-normalize: {'rule': rule, 'symbolic': ..., 'numerical': ...}}
    """
    torch.set_default_dtype(dtype)

    if not max_l1 + max_l2 >= max_l3:
        raise ValueError("l1 + l2 must be greater than or equal to l3")

    # TODO, deal with l1=0 or l2=0 cases
    all_K = {}
    for l1 in range(max_l1 + 1):
        for l2 in range(max_l2 + 1):
            for l3 in range(abs(l1 - l2), min(l1 + l2 + 1, max_l3 + 1)):
                for normalize in ["unity", "none"]:
                    K_symbolic, _, _, _ = get_coupling_symbolic(l1, l2, l3)
                    K, rule = get_coupling_operator(l1, l2, l3, normalize)

                    K_symbolic = simplify_linear_combination(K_symbolic)
                    # replace δ (delta) by d
                    # replace ε (epsilon) by e
                    K_symbolic = str(K_symbolic).replace("δ", "d").replace("ε", "e")

                    # Convert to numpy to save it
                    K = K.tolist()
                    key = f"{l1}-{l2}-{l3}-{normalize}"
                    all_K[key] = {
                        "rule": rule,
                        "symbolic": K_symbolic,
                        "numerical": K,
                    }

    return all_K


if __name__ == "__main__":
    # Generate coupling operators and rules
    all_K = generate_coupling_operators(max_l1=4, max_l2=4, max_l3=4)

    # Save to yaml
    filename = Path("./tensor_product_projector.yaml")
    yaml_dump(all_K, filename, compress=True)
