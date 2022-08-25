import pathlib
import subprocess
import types
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, Union

import yaml


T = TypeVar("T")


def save_git_hash(path: pathlib.Path) -> None:
    """Saves the current git hash to the given path.

    Args:
        path: Path to save git hash.
    """
    process = subprocess.Popen(
        ["git", "rev-parse", "HEAD"], shell=False, stdout=subprocess.PIPE
    )
    git_head_hash = process.communicate()[0].strip()
    with open(path / "git_hash.txt", "wb") as f:
        f.write(git_head_hash)


def get_class(classname: Union[str, Type[T]], module: types.ModuleType) -> Type[T]:
    """Gets the class from the given module.

    Returns classname directly if it is already a class.

    Args:
        classname: Class name with '.' denoting submodules.
        module: Python module to search.

    Returns:
        Class.
    """

    def _get_submodule(module, submodules: List[str]) -> Type[T]:
        if not submodules:
            return module
        return _get_submodule(vars(module)[submodules[0]], submodules[1:])

    if isinstance(classname, str):
        submodules = classname.split(".")

        try:
            return _get_submodule(module, submodules)
        except KeyError as e:
            raise KeyError(f"Cannot find {classname} in {module.__name__}:\n{e}")
    else:
        return classname


def get_instance(
    classname: Union[str, T], kwargs: Dict[str, Any], module: types.ModuleType
) -> T:
    """Creates an instance of the given class with kwargs.

    Returns classname directly if it is already an instance.

    Args:
        classname: Class name with '.' denoting submodules.
        kwargs: Class constructor kwargs .
        module: Python module to search.

    Returns:
        Class instance.
    """
    if isinstance(classname, str):
        cls: Type[T] = get_class(classname, module)
        return cls(**kwargs)
    else:
        return classname


def parse_class(config: Dict[str, Any], key: str, module: types.ModuleType) -> Type:
    """Parses the class from a config.

    Args:
        config: Config dict.
        key: Dict key containing class name as its value.
        module: Python module to search.

    Returns:
        Class.
    """
    if key not in config:
        raise KeyError(f"{key} missing from config")
    return get_class(config[key], module)


def parse_kwargs(config: Dict[str, Any], key: str) -> Dict:
    """Parses the kwargs from a config.

    Args:
        config: Config dict.
        key: Dict key containing kwargs as its value.

    Returns:
        Kwargs or empty dict.
    """
    try:
        kwargs = config[key]
    except KeyError:
        return {}
    return {} if kwargs is None else kwargs


def load_config(
    path: Union[str, pathlib.Path], config_prefix: Optional[str] = None
) -> Dict[str, Any]:
    """Loads a config from path.

    Args:
        path: Path to the config, config directory, or checkpoint.
        config_prefix: Prefix of config file to search: "{config_prefix}_config.yaml".

    Returns:
        Config dict.
    """
    if isinstance(path, str):
        path = pathlib.Path(path)

    if path.suffix == ".yaml":
        config_path = path
    else:
        if path.suffix == ".pt":
            path = path.parent

        config_name = "config.yaml"
        if config_prefix is not None:
            config_name = f"{config_prefix}_{config_name}"

        config_path = path / config_name

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


class Factory(Generic[T]):
    """Base factory class."""

    def __init__(
        self,
        config: Union[str, pathlib.Path, Dict[str, Any]],
        key: str,
        module: types.ModuleType,
    ):
        """Parses the config.

        Args:
            config: Config path or dict.
            key: Key of class definition in the config dict.
            module: Python module of class.
        """
        if not isinstance(config, dict):
            with open(config, "r") as f:
                config = yaml.safe_load(f)
        assert isinstance(config, dict)

        self._config = config
        self._cls = parse_class(config, key, module)
        self._kwargs = dict(parse_kwargs(config, f"{key}_kwargs"))
        self._key = key
        self._last_instance: Optional[T] = None
        self._post_hooks: List[Callable[[T], None]] = []

    @property
    def config(self) -> Dict[str, Any]:
        """Loaded config dict."""
        return self._config

    @property
    def cls(self) -> Type:
        """Parsed class name."""
        return self._cls

    @property
    def kwargs(self) -> Dict[str, Any]:
        """Parsed class kwargs."""
        return self._kwargs

    @property
    def last_instance(self) -> Optional[T]:
        """Last created instance."""
        return self._last_instance

    def save_config(self, path: Union[str, pathlib.Path]) -> None:
        """Saves the config to path.

        Args:
            path: Directory where config will be saved.
        """
        path = pathlib.Path(path)
        with open(path / f"{self._key}_config.yaml", "w") as f:
            yaml.dump(self.config, f)

    def get_instance(self, *args, **kwargs) -> T:
        """Gets the last created instance or creates a new one with the given args.

        Args:
            *args: Constructor args.
            **kwargs: Constructor kwargs.

        Returns:
            Last created instance.
        """
        if self.last_instance is None:
            self.__call__(*args, **kwargs)
        assert self.last_instance is not None
        return self.last_instance

    def add_post_hook(self, post_hook: Callable[[T], Any]) -> None:
        """Adds a callback function to call when this factory is called.

        Args:
            post_hook: Function to call.
        """
        self._post_hooks.append(post_hook)

    def run_post_hooks(self, instance: T) -> None:
        """Runs the post hooks.

        Args:
            instance: Instance to pass to the post hooks.
        """
        self._last_instance = instance
        for post_hook in self._post_hooks:
            post_hook(instance)

    def __call__(self, *args, **kwargs) -> T:
        """Creates an instance of the class.

        Args:
            *args: Constructor args.
            **kwargs: Constructor kwargs.

        Returns:
            Class instance.
        """
        merged_kwargs = dict(self.kwargs)
        merged_kwargs.update(kwargs)
        instance = self.cls(*args, **merged_kwargs)

        self.run_post_hooks(instance)

        return instance
