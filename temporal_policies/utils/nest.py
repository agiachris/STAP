from typing import (
    Any,
    Dict,
    Callable,
    Generator,
    Iterator,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np

# mypy doesn't support recursive types, so we define two levels of recursion.
StructureAtom = Any
Structure = Union[StructureAtom, Dict[str, StructureAtom], Sequence[StructureAtom]]
NestedStructure = Union[Structure, Dict[str, Structure], Sequence[Structure]]


def map_structure(
    func: Callable,
    *args: NestedStructure,
    atom_type: Union[Type, Tuple[Type, ...]] = (
        np.ndarray,
        float,
        int,
        bool,
        str,
        type(None),
    )
) -> NestedStructure:
    """Applies the function over the nested structure atoms.

    Works like tensorflow.nest.map_structure():
    https://www.tensorflow.org/api_docs/python/tf/nest/map_structure

    Args:
        func: Function applied to the atoms of *args.
        *args: Nested structure arguments of `func`.
        atom_type: Types considered to be atoms in the nested structure.

    Returns:
        Results of func(*args_atoms) in the same nested structure as *args.
    """
    arg_0 = args[0]
    if isinstance(arg_0, atom_type):
        return func(*args)
    elif isinstance(arg_0, dict):
        return {key: map_structure(func, *(arg[key] for arg in args)) for key in arg_0}  # type: ignore
    else:
        iterable_class = type(arg_0)
        return iterable_class(map_structure(func, *args_i) for args_i in zip(*args))  # type: ignore


def structure_iterator(
    structure: NestedStructure,
    atom_type: Union[Type, Tuple[Type, ...]] = (
        np.ndarray,
        float,
        int,
        bool,
        str,
        type(None),
    ),
) -> Iterator:
    """Provides an iterator over the atom values in the flattened nested structure.

    Args:
        structure: Nested structure.
        atom_type: Types considered to be atoms in the nested structure.

    Returns:
        Iterator over the atom values in the flattened nested structure.
    """

    def iterate_structure(
        structure: NestedStructure,
    ) -> Generator:
        if isinstance(structure, atom_type):
            yield structure
        elif isinstance(structure, dict):
            for val in structure.values():
                for elem in iterate_structure(val):
                    yield elem
        else:
            for val in structure:
                for elem in iterate_structure(val):
                    yield elem

    return iter(iterate_structure(structure))
