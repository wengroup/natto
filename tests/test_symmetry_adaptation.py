"""Symmetry-adapted mappings.

Their order is part of the result: it numbers the channels of a weight. The test runs
over the Table III classes that have an intrinsic symmetry.
"""

import pytest

from tests.test_reduction import (
    PHYSICAL_TENSOR_CLASSES,
    TensorClass,
    get_independent_mappings_cached,
    get_tensor_class_params,
)


@pytest.mark.parametrize(
    "tensor_class",
    get_tensor_class_params(
        [tc for tc in PHYSICAL_TENSOR_CLASSES if tc.symmetry is not None]
    ),
)
def test_symmetry_adapted_order(tensor_class: TensorClass):
    """Symmetry-adapted mappings are ordered by the mappings they contain, no ties.

    The kept mappings sit at increasing candidate numbers, so the support over the
    candidates orders them as the support over the kept mappings does.
    """
    for weight in tensor_class.multiplicity:
        Q, _ = get_independent_mappings_cached(
            weight, tensor_class.rank, tensor_class.symmetry
        )
        supports = [tuple(i for i, c in enumerate(Q_p.coefficients) if c) for Q_p in Q]

        assert all(a < b for a, b in zip(supports, supports[1:]))
