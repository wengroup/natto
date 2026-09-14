r"""Precompute the harmonic operators up to a given weight.

Writes `harmonic_operators.yaml` holding, for each weight and each
normalization, the harmonic operator H as its exact symbolic form and its
evaluated values, with the einsum rule that applies it:

    V = einsum(rule, H, *[a] * weight)

taking a unit vector a to the Cartesian harmonic of that weight, the Cartesian
counterpart of a spherical harmonic.

Edit `max_weight` at the bottom to change how far the sweep runs. Under the
`legendre` normalization the weight-fold contraction of the harmonic with a unit
vector is the Legendre polynomial of the angle between the two; under `none` the
operator is the bare natural projector. The symbolic forms are written with `d`
for the Kronecker delta and `e` for the Levi-Civita symbol, so the file stays
plain ASCII.
"""

from pathlib import Path

import numpy as np

from natto.harmonics import get_harmonic_operator
from natto.natural_projector import get_natural_projector
from natto.utils import yaml_dump


def generate_harmonic_operators(max_weight: int) -> dict[str, dict[str, np.ndarray]]:
    """
    Generate harmonic operators and the corresponding einsum rules.

    Args:
        max_weight: Maximum weight of the harmonic.

    Return:
        Harmonic operators and rules,
        {weight-normalize: {'rule': rule, 'symbolic': ..., 'numerical': ...}}
    """
    all_H = {}
    for weight in range(max_weight + 1):
        for normalize in ["legendre", "none"]:
            # the unnormalized harmonic operator is the natural projector
            H_symbolic = get_natural_projector(weight)
            H, rule = get_harmonic_operator(weight, normalize)

            # `d` for the Kronecker delta and `e` for the Levi-Civita symbol
            H_symbolic = H_symbolic.to_string(ascii=True)

            all_H[f"{weight}-{normalize}"] = {
                "rule": rule,
                "symbolic": H_symbolic,
                "numerical": H.tolist(),
            }

    return all_H


if __name__ == "__main__":
    # Generate harmonic operators and rules
    all_H = generate_harmonic_operators(max_weight=4)

    # Save to yaml
    filename = Path("./harmonic_operators.yaml")
    yaml_dump(all_H, filename, compress=True)
