from typing import Callable, Generator, Iterator, Optional, Tuple, Type, Union

import numpy as np
import torch

from temporal_policies.utils import typing


def map_structure(
    func: Callable,
    *args,
    atom_type: Union[Type, Tuple[Type, ...]] = (
        torch.Tensor,
        np.ndarray,
        *typing.scalars,
        type(None),
    ),
    skip_type: Optional[Union[Type, Tuple[Type, ...]]] = None,
):
    """Applies the function over the nested structure atoms.

    Works like tensorflow.nest.map_structure():
    https://www.tensorflow.org/api_docs/python/tf/nest/map_structure

    Args:
        func: Function applied to the atoms of *args.
        *args: Nested structure arguments of `func`.
        atom_type: Types considered to be atoms in the nested structure.
        skip_type: Types to be skipped and returned as-is in the nested structure.

    Returns:
        Results of func(*args_atoms) in the same nested structure as *args.
    """
    arg_0 = args[0]
    if isinstance(arg_0, atom_type):
        return func(*args)
    elif skip_type is not None and isinstance(arg_0, skip_type):
        return arg_0 if len(args) == 1 else args
    elif isinstance(arg_0, dict):
        return {
            key: map_structure(
                func,
                *(arg[key] for arg in args),
                atom_type=atom_type,
                skip_type=skip_type,
            )
            for key in arg_0
        }
    elif hasattr(arg_0, "__iter__"):
        iterable_class = type(arg_0)
        return iterable_class(
            map_structure(func, *args_i, atom_type=atom_type, skip_type=skip_type)
            for args_i in zip(*args)
        )
    else:
        return arg_0 if len(args) == 1 else args


def structure_iterator(
    structure,
    atom_type: Union[Type, Tuple[Type, ...]] = (
        torch.Tensor,
        np.ndarray,
        *typing.scalars,
        type(None),
    ),
    skip_type: Optional[Union[Type, Tuple[Type, ...]]] = None,
) -> Iterator:
    """Provides an iterator over the atom values in the flattened nested structure.

    Args:
        structure: Nested structure.
        atom_type: Types considered to be atoms in the nested structure.
        skip_type: Types to be skipped and returned as-is in the nested structure.

    Returns:
        Iterator over the atom values in the flattened nested structure.
    """

    def iterate_structure(
        structure,
    ) -> Generator:
        if isinstance(structure, atom_type):
            yield structure
        elif skip_type is not None and isinstance(structure, skip_type):
            pass
        elif isinstance(structure, dict):
            for val in structure.values():
                for elem in iterate_structure(val):
                    yield elem
        elif hasattr(structure, "__iter__"):
            for val in structure:
                for elem in iterate_structure(val):
                    yield elem
        else:
            pass

    return iter(iterate_structure(structure))
