"""Index letters, and the renaming of them.

Every operator in this package is built as a symbolic expression over named
indices, and then contracted through an ``einsum`` rule written in those same
letters. Two kinds of bookkeeping recur often enough to live together here:

- naming a fresh run of letters for a tensor of a given rank, where Roman
  letters carry the indices of the generic tensor :math:`\\mathbf T_n` and Greek
  ones -- spelled upper case here -- the indices of the ICT;
- renaming the letters of an expression already built, either by a shift or by
  an explicit map, so that two expressions can be contracted without their
  indices colliding.

Nothing here knows what the operators mean; it is the plumbing they share.
"""

import string

from natto.symbolic import CartesianTensor, LinearCombination, TensorProduct


def letter_index(n: int, start: int = 0, upper_case: bool = False) -> str:
    """
    Get a list of letters 'abc...' of length n.

    Args:
        n: the length of the letters
        start: the starting index
        upper_case: whether to use upper case letters
    """
    if upper_case:
        return string.ascii_uppercase[start : start + n]
    else:
        return string.ascii_lowercase[start : start + n]


def double_index(n: int, start: int = 0, upper_case: bool = False) -> list[str]:
    """
    Get multiple double indices, like ['ab', 'cd', 'ef'].

    Args:
        n: the number of double indices
        start: the starting index
        upper_case: whether to use upper case letters

    Examples:
        >>> double_index(2)
        ['ab', 'cd']
        >>> double_index(3, start=1)
        ['bc', 'cd', 'de']
    """
    indices = letter_index(2 * n, start, upper_case)
    return [indices[i : i + 2] for i in range(0, 2 * n, 2)]


def repeat_double_index(n: int, start: int = 0, upper_case: bool = False) -> list[str]:
    """
    Get multiple repeated double indices, like ['aa', 'bb', 'cc'].

    Args:
        n: the number of double indices
        start: the starting index
        upper_case: whether to use upper case letters

    Examples:
        >>> repeat_double_index(2)
        ['aa', 'bb']
        >>> repeat_double_index(3, start=1)
        ['bb', 'cc', 'dd']
    """
    indices = letter_index(n, start, upper_case)

    return [s + s for s in indices]


def shift_index(
    tensor: CartesianTensor | TensorProduct, shift: int, letters: str = None
) -> CartesianTensor | TensorProduct:
    """
    Shift the index of a tensor by a certain amount.

    For example, for T_ijk, and shift=1, the new tensor is T_jkl.

    Args:
        tensor: The tensor to shift the index.
        shift: The amount to shift the index.
        letters: The letters signifying the indices to shift. If None, shift all the
            indices. For example, if T_ijAB, and letters = 'AB', then the new tensor
            would be T_ijCD, where C and D are the new letters.

    Returns:
        The new tensor with the shifted index.
    """

    # The zero tensor has no meaningful indices to shift. In particular, a zero
    # TensorProduct stores a Zero component whose constructor does not accept the
    # general CartesianTensor arguments used below.
    if tensor.factor == 0:
        return tensor

    def _shift(t: CartesianTensor):
        if letters is None:
            indices = "".join([chr(ord(i) + shift) for i in t.indices])
        else:
            indices = "".join(
                [chr(ord(i) + shift) if i in letters else i for i in t.indices]
            )
        return t.__class__(indices, factor=t.factor, symbol=t.symbol)

    if isinstance(tensor, CartesianTensor):
        return _shift(tensor)

    elif isinstance(tensor, TensorProduct):
        components = [_shift(t) for t in tensor]
        return tensor.__class__(*components, factor=tensor.factor)

    else:
        raise ValueError(f"Unknown tensor type: {type(tensor)}")


def relabel_indices(
    tensor: CartesianTensor | TensorProduct, mapping: dict[str, str]
) -> CartesianTensor | TensorProduct:
    """Rename the indices of a tensor according to `mapping`.

    The substitution is simultaneous, so a mapping may permute letters among
    themselves. Applying it one letter at a time would chain the replacements,
    turning a transposition into a collapse.

    Args:
        tensor: The tensor whose indices to rename.
        mapping: Old index letter to new. Letters absent from it are left alone.

    Returns:
        The tensor with its indices renamed.
    """
    # A zero tensor carries no meaningful indices, and a zero TensorProduct holds
    # a component whose constructor does not take the arguments used below.
    if tensor.factor == 0:
        return tensor

    def _relabel(t: CartesianTensor):
        indices = "".join(mapping.get(index, index) for index in t.indices)

        return t.__class__(indices, factor=t.factor, symbol=t.symbol)

    if isinstance(tensor, CartesianTensor):
        return _relabel(tensor)

    if isinstance(tensor, TensorProduct):
        return tensor.__class__(*[_relabel(t) for t in tensor], factor=tensor.factor)

    raise ValueError(f"Unknown tensor type: {type(tensor)}")


def relabel_indices_2(
    tensor: LinearCombination, mapping: dict[str, str]
) -> LinearCombination:
    """Rename the indices of every term of a linear combination.

    Args:
        tensor: The linear combination whose indices to rename.
        mapping: Old index letter to new, applied simultaneously.

    Returns:
        The linear combination with its indices renamed.
    """
    return LinearCombination(*[relabel_indices(t, mapping) for t in tensor])


def shift_index_2(
    tensor: LinearCombination, shift: int, letters: str = None
) -> LinearCombination:
    """
    Shift all the index of a Tensors object by a certain amount.

    Returns:
        The new tensor with the shifted index.
    """
    components = [shift_index(t, shift, letters) for t in tensor]
    return LinearCombination(*components)
