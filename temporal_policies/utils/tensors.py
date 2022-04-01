from typing import Any, Callable, Iterator, List, Optional, Tuple, Type, Union

import gym  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore

from temporal_policies.utils import nest


def null_tensor(
    space: gym.spaces.Space, batch_shape: Tuple[int, ...] = tuple()
) -> torch.Tensor:
    """Constructs a null tensor from the given space.

    Args:
        space: Gym space.
        batch_shape: Batch shape.
    """

    def null_value(dtype: np.number) -> Union[float, int]:
        if isinstance(dtype, np.floating):
            return float("nan")
        else:
            return 0

    return torch.full(
        (*batch_shape, *space.shape), null_value(space.dtype), dtype=space.dtype
    )


def map_structure(
    func: Callable,
    *args: nest.NestedStructure,
    atom_type: Optional[Union[Type, Tuple[Type, ...]]] = (torch.Tensor, np.ndarray),
) -> nest.NestedStructure:
    """Maps the function over the structure containing either Torch tensors or Numpy
    arrays.

    Args:
        func: Function to be mapped.
        *args: Nested structure arguments of `func`.
        atom_type: Type to which the function should be applied.
    """
    return nest.map_structure(
        func,
        *args,
        atom_type=atom_type,
        skip_type=(np.ndarray, torch.Tensor, float, int, bool, str, type(None)),
    )


def iterate_structure(
    structure: nest.NestedStructure,
    atom_type: Optional[Union[Type, Tuple[Type, ...]]] = (torch.Tensor, np.ndarray),
) -> Iterator:
    """Provides an iterator over the Torch tensors or Numpy arrays in the nested
    structure.

    Args:
        structure: Nested structure
        atom_type: Types considered to be atoms in the nested structure.

    Returns:
        Iterator over the atom values in the flattened nested structure.
    """
    return nest.iterate_structure(
        structure,
        atom_type=atom_type,
        skip_type=(np.ndarray, torch.Tensor, float, int, bool, str, type(None)),
    )


def to(structure: nest.NestedStructure, device: torch.device) -> nest.NestedStructure:
    """Moves the nested structure to the given device.

    Numpy arrays are converted to Torch tensors first.

    Args:
        structure: Nested structure.
        device: Torch device.

    Returns:
        Transferred structure.
    """

    def _to(x: nest.StructureAtom) -> Union[nest.StructureAtom, torch.Tensor]:
        if isinstance(x, torch.Tensor):
            return x.to(device)
        else:
            return torch.from_numpy(x).to(device)

    return map_structure(_to, structure)


def numpy(structure: nest.NestedStructure) -> nest.NestedStructure:
    """Converts the nested structure to Numpy arrays.

    Args:
        structure: Nested structure.

    Returns:
        Numpy structure.
    """
    return map_structure(
        lambda x: x.cpu().detach().numpy(), structure, atom_type=torch.Tensor
    )


def from_numpy(structure: nest.NestedStructure) -> nest.NestedStructure:
    """Converts the nested structure to Torch tensors.

    Args:
        structure: Nested structure.

    Returns:
        Tensor structure.
    """
    return map_structure(lambda x: torch.from_numpy(x), structure, atom_type=np.ndarray)


def numpy_wrap(func: Callable) -> Callable:
    """Decorator that creates a wrapper around Torch functions to be compatible
    with Numpy inputs and outputs.

    Args:
        func: Torch function.

    Returns:
        Function compatible with Torch or Numpy.
    """

    def numpy_func(*args, **kwargs):
        is_numpy = any(iterate_structure((args, kwargs), atom_type=np.ndarray))
        if is_numpy:
            args, kwargs = from_numpy((args, kwargs))

        result = func(*args, **kwargs)

        if is_numpy:
            result = numpy(result)

        return result

    return numpy_func


def torch_wrap(func: Callable) -> Callable:
    """Decorator that creates a wrapper around Numpy functions to be compatible
    with Torch inputs and outputs.

    Args:
        func: Numpy function.

    Returns:
        Function compatible with Torch or Numpy.
    """

    def torch_func(*args, **kwargs):
        is_tensor = False
        for tensor in iterate_structure((args, kwargs)):
            is_tensor = True
            device = tensor.device
            break

        result = func(*args, **kwargs)

        if is_tensor:
            result = to(result, device)

        return result


def vmap(dim: int) -> Callable:
    """Decorator that vectorizes functions.

    Args:
        dim: Numer of dimensions of the first function input.
        func: Function to vectorize.

    Returns:
        Vectorized function.
    """

    def append(x: Any, xs: List[Any]) -> None:
        xs.append(x)

    def stack(x: Any, xs: List[Any]) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(x, torch.Tensor):
            return torch.stack(xs, dim=0)
        else:
            return np.array(xs)

    def _vmap(func: Callable) -> Callable:
        def vectorized_func(*args, **kwargs):
            is_batch = False
            for arr in iterate_structure((args, kwargs)):
                is_batch = arr.dim != dim
                batch_shape = arr.shape[:-dim] if dim > 0 else arr.shape
                break

            if not is_batch:
                return func(*args, **kwargs)

            # Flatten batch dims.
            batch_size = np.prod(batch_shape)
            args, kwargs = map_structure(
                lambda x: x.reshape(batch_size, *x.shape[-dim:]), (args, kwargs)
            )

            # Call func for each batch element.
            results = None
            for i in range(batch_size):
                args_i, kwargs_i = map_structure(lambda x: x[i], (args, kwargs))
                result = func(*args_i, **kwargs_i)
                if i == 0:
                    results = map_structure(lambda x: [x], result)
                else:
                    map_structure(append, result, results)
            results = map_structure(stack, result, results)

            # Restore batch dims.
            if dim == 0:
                results = map_structure(lambda x: x.reshape(*batch_shape), results)
            else:
                results = map_structure(
                    lambda x: x.reshape(*batch_shape, *x.shape[-dim:]), results
                )

            return results

        return vectorized_func

    return _vmap
