r"""Precompute the coupling operators of every admissible weight triple.

Writes `coupling_operators.yaml` holding, for each triple of weights and each
normalization, the coupling operator K -- the Cartesian counterpart of a
Clebsch-Gordan coefficient -- as its exact symbolic form and its evaluated
values, with the einsum rule that applies it:

    Z = einsum(rule, K, X, Y)

taking natural tensors X and Y of weights l1 and l2 to the weight-l3 part of
their product.

Edit `max_l1`, `max_l2` and `max_l3` at the bottom to change how far the sweep
runs; the cost grows quickly with the weights. The symbolic forms are written
with `d` for the Kronecker delta and `e` for the Levi-Civita symbol, so the file
stays plain ASCII.
"""

from pathlib import Path

import numpy as np

from natto.algebra import simplify_linear_combination
from natto.coupling import get_coupling_operator, get_coupling_symbolic
from natto.utils import yaml_dump


def generate_coupling_operators(
    max_l1: int, max_l2: int, max_l3: int, dtype=np.float64
) -> dict[str, dict[str, np.ndarray]]:
    """
    Generate coupling operators and the corresponding einsum rules.

    Args:
        max_l1: Maximum weight of X.
        max_l2: Maximum weight of Y.
        max_l3: Maximum weight of Z.
        dtype: The data type of the generated arrays.

    Return:
        Coupling operators and rules,
        {l1-l2-l3-normalize: {'rule': rule, 'symbolic': ..., 'numerical': ...}}
    """
    if not max_l1 + max_l2 >= max_l3:
        raise ValueError("l1 + l2 must be greater than or equal to l3")

    # TODO, deal with l1=0 or l2=0 cases
    all_K = {}
    for l1 in range(max_l1 + 1):
        for l2 in range(max_l2 + 1):
            for l3 in range(abs(l1 - l2), min(l1 + l2 + 1, max_l3 + 1)):
                for normalize in ["unity", "none"]:
                    K_symbolic, _, _, _ = get_coupling_symbolic(l1, l2, l3)
                    K, rule = get_coupling_operator(l1, l2, l3, normalize, dtype=dtype)

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
    filename = Path("./coupling_operators.yaml")
    yaml_dump(all_K, filename, compress=True)
