import math
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch

from temporal_policies.utils import nest, typing


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
    x: Union[torch.Tensor, np.ndarray, Sequence[float], Sequence[int], typing.Scalar]
) -> torch.Tensor:
    """Converts the scalar or array to a tensor.

    Args:
        x: Scalar or array.

    Returns:
        Tensor.
    """
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        return torch.tensor(x)


def dim(
    x: Union[torch.Tensor, np.ndarray, Sequence[float], Sequence[int], typing.Scalar]
) -> int:
    """Gets the number of dimensions of x.

    Args:
        x: Scalar or array.

    Returns:
        Number of dimensions.
    """
    if isinstance(x, torch.Tensor):
        return x.dim()
    elif isinstance(x, np.ndarray):
        return x.ndim
    elif isinstance(x, (float, int)):
        return 0
    else:
        return 1


def map_structure(
    func: Callable,
    *args,
    atom_type: Union[Type, Tuple[Type, ...]] = (torch.Tensor, np.ndarray)
):
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
        skip_type=(np.ndarray, torch.Tensor, *typing.scalars, str, type(None)),
    )


def structure_iterator(
    structure, atom_type: Union[Type, Tuple[Type, ...]] = (torch.Tensor, np.ndarray)
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
        skip_type=(np.ndarray, torch.Tensor, *typing.scalars, str, type(None)),
    )


def to(structure, device: torch.device):
    """Moves the nested structure to the given device.

    Numpy arrays are converted to Torch tensors first.

    Args:
        structure: Nested structure.
        device: Torch device.

    Returns:
        Transferred structure.
    """

    def _to(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)

        try:
            return torch.from_numpy(x).to(device)
        except TypeError:
            return x

    return map_structure(_to, structure)


def numpy(structure):
    """Converts the nested structure to Numpy arrays.

    Args:
        structure: Nested structure.

    Returns:
        Numpy structure.
    """
    return map_structure(
        lambda x: x.cpu().detach().numpy(), structure, atom_type=torch.Tensor
    )


def from_numpy(structure, device: Optional[torch.device] = None):
    """Converts the nested structure to Torch tensors.

    Args:
        structure: Nested structure.

    Returns:
        Tensor structure.
    """
    if device is None:
        return map_structure(
            lambda x: torch.from_numpy(x), structure, atom_type=np.ndarray
        )
    return map_structure(
        lambda x: torch.from_numpy(x).to(device), structure, atom_type=np.ndarray
    )


def unsqueeze(structure, dim: int):
    def _unsqueeze(
        x: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(x, np.ndarray):
            return np.expand_dims(x, dim)
        elif isinstance(x, torch.Tensor):
            return x.unsqueeze(dim)
        return x

    return map_structure(_unsqueeze, structure)


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
        try:
            next(structure_iterator((args, kwargs), atom_type=np.ndarray))
        except StopIteration:
            is_numpy = False
        else:
            is_numpy = True
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
        dims: Number of dimensions of the first tensor function input.
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


def batch(dims: int) -> Callable:
    """Decorator that ensures function inputs have at least one batch dimension.

    If original arguments are not batched, returned results will also not have a batch.

    Args:
        dims: Number of dimensions of the first tensor function input.
        func: Function to vectorize.

    Returns:
        Flexible batch function.
    """

    def _vmap(func: Callable) -> Callable:
        def batched_func(*args, **kwargs):
            try:
                arr = next(structure_iterator((args, kwargs)))
            except StopIteration:
                is_batch = False
            else:
                arr_dim = arr.dim() if isinstance(arr, torch.Tensor) else arr.ndim
                is_batch = arr_dim != dims

                if is_batch:
                    return func(*args, **kwargs)

                args, kwargs = map_structure(lambda x: x.unsqueeze(0), (args, kwargs))
                results = func(*args, **kwargs)
                results = map_structure(lambda x: x.squeeze(0), results)

                return results

        return batched_func

    return _vmap


def rgb_to_cnn(img_rgb: torch.Tensor, contiguous: bool = False) -> torch.Tensor:
    if contiguous:
        return img_rgb.moveaxis(-1, -3).contiguous().float() / 255
    else:
        return img_rgb.moveaxis(-1, -3).float() / 255


def cnn_to_rgb(img_cnn: torch.Tensor, contiguous: bool = False) -> torch.Tensor:
    img_rgb = (255 * img_cnn.clip(0, 1).moveaxis(-3, -1) + 0.5).to(torch.uint8)
    if contiguous:
        return img_rgb.contiguous()
    else:
        return img_rgb


def get_num_free_bytes() -> int:
    cuda_device = torch.cuda.current_device()
    num_unreserved_bytes: int = torch.cuda.mem_get_info(cuda_device)[0]  # type: ignore
    num_reserved_bytes = torch.cuda.memory_reserved(cuda_device)
    num_allocated_bytes = torch.cuda.memory_allocated(cuda_device)
    num_free_bytes = num_unreserved_bytes + num_reserved_bytes - num_allocated_bytes

    return num_free_bytes


def compute_minibatch(batch_size: int, element_size: int) -> Tuple[int, int]:
    num_free_bytes = get_num_free_bytes()
    max_minibatch_size = int(num_free_bytes / (2 * element_size))

    # Redistribute batch size equally across all iterations.
    num_batches = int(math.ceil(batch_size / max_minibatch_size) + 0.5)
    minibatch_size = int(math.ceil(batch_size / num_batches) + 0.5)

    return minibatch_size, num_batches
