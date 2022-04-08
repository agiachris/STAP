from typing import Any, Callable, Iterator, List, Sequence, Tuple, Type, Union

import numpy as np  # type: ignore
import torch  # type: ignore

from temporal_policies.utils import nest


def device(device: Union[str, torch.device] = "auto") -> torch.device:
    """Gets the torch device.

    Args:
        device: "cpu", "gpu", or "auto".

    Returns:
        Torch device.
    """
    if isinstance(device, torch.device):
        return device

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def to_tensor(
    x: Union[torch.Tensor, np.ndarray, Sequence[float], Sequence[int], float, int]
) -> torch.Tensor:
    """Converts the scalar or array to a tensor.

    Args:
        x: Scalar or array.

    Returns:
        Tensor with at least 1 dimension (non-scalar).
    """
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, (float, int)):
        return torch.tensor([x])
    else:
        return torch.tensor(x)


def map_structure(
    func: Callable,
    *args: nest.NestedStructure,
    atom_type: Union[Type, Tuple[Type, ...]] = (torch.Tensor, np.ndarray),
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


def structure_iterator(
    structure: nest.NestedStructure,
    atom_type: Union[Type, Tuple[Type, ...]] = (torch.Tensor, np.ndarray),
) -> Iterator:
    """Provides an iterator over the Torch tensors or Numpy arrays in the nested
    structure.

    Args:
        structure: Nested structure
        atom_type: Types considered to be atoms in the nested structure.

    Returns:
        Iterator over the atom values in the flattened nested structure.
    """
    return nest.structure_iterator(
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


# TODO: Handle device.
def numpy_wrap(func: Callable) -> Callable:
    """Decorator that creates a wrapper around Torch functions to be compatible
    with Numpy inputs and outputs.

    Args:
        func: Torch function.

    Returns:
        Function compatible with Torch or Numpy.
    """

    def numpy_func(*args, **kwargs):
        is_numpy = any(structure_iterator((args, kwargs), atom_type=np.ndarray))
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
        try:
            tensor = next(structure_iterator((args, kwargs)))
        except StopIteration:
            device = None
        else:
            device = tensor.device
            args, kwargs = numpy((args, kwargs))

        result = func(*args, **kwargs)

        if device is not None:
            result = to(result, device)

        return result

    return torch_func


def vmap(dims: int) -> Callable:
    """Decorator that vectorizes functions.

    Args:
        dims: Number of dimensions of the first function input.
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
            try:
                arr = next(structure_iterator((args, kwargs)))
            except StopIteration:
                is_batch = False
            else:
                arr_dim = arr.dim() if isinstance(arr, torch.Tensor) else arr.ndim
                is_batch = arr_dim != dims

            if not is_batch:
                return func(*args, **kwargs)

            # Flatten batch dims.
            batch_shape = arr.shape[:-dims] if dims > 0 else arr.shape
            batch_dims = len(batch_shape)
            batch_size = np.prod(batch_shape)
            args, kwargs = map_structure(
                lambda x: x.reshape(batch_size, *x.shape[batch_dims:]), (args, kwargs)
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
            results = map_structure(
                lambda x: x.reshape(*batch_shape, *x.shape[1:]), results
            )

            return results

        return vectorized_func

    return _vmap
