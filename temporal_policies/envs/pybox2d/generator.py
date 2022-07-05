from collections import OrderedDict
from copy import deepcopy
import os
from typing import Any, Dict, Optional

from Box2D import b2FixtureDef, b2PolygonShape, b2World
import numpy as np
import yaml

from .utils import GeometryHandler, sample_random_params
from .constants import COLORS


class Generator(GeometryHandler):
    def __init__(
        self,
        config,
        env: Optional[Dict[str, Any]] = None,
        world: Optional[b2World] = None,
        env_params={},
        rand_params={},
        mode="init",
        **kwargs,
    ):
        """PyBox2D environment generator."""
        super().__init__(**kwargs)
        with open(os.path.realpath(config), "r") as file:
            self._config = yaml.load(file, yaml.Loader)
        self._env_params = env_params
        self._rand_params = rand_params

        # Public attributes: Box2D world and environment parameters
        self.env = env
        self.world = world
        if mode == "init":
            next(self)
        elif mode == "load":
            assert env is not None and world is not None
        elif mode == "clone":
            self._clone()
        else:
            raise ValueError(f"Unsupported Generator instantiation mode {mode}")

    def __iter__(self):
        """Declare class as an iterator."""
        return self

    def __next__(self):
        """Randomly sample a Box2D environment, setting up the public self.env
        dictionary and a b2World instance as self.world
        """
        self._setup_env()
        self._setup_world()

    def _setup_env(self):
        """Setup ordered dictionary of environment objects and their attributes."""
        env = dict(self._config.copy(), **self._env_params.copy())
        env = OrderedDict(sorted(env.items()))
        assert all(
            object_data["class"] in GeometryHandler._VALID_CLASSES
            for object_data in env.values()
        )

        # Light domain randomization (shape_kwargs)
        for object_name, rand_params in self._rand_params.items():
            for k, v in rand_params["shape_kwargs"].items():
                env[object_name]["shape_kwargs"][k] = sample_random_params(v)

        self.env = env
        self.vectorize(self.env)

    def _setup_world(self):
        """Setup Box2D world by constructing rigid bodies."""
        self.world = b2World()
        for object_name, object_data in self.env.items():
            create_shape_fn = getattr(self, object_data["class"])
            object_data["shapes"] = create_shape_fn(
                object_name, **object_data["shape_kwargs"]
            )
            object_data["bodies"] = {}

            # Light domain randomization (body_kwargs)
            rand_params = {}
            if (
                object_name in self._rand_params
                and "body_kwargs" in self._rand_params[object_name]
            ):
                for k, v in self._rand_params[object_name]["body_kwargs"].items():
                    rand_params[k] = sample_random_params(v)
                object_data["body_kwargs"].update(rand_params)

            for k, v in object_data["shapes"].items():
                if object_data["type"] == "static":
                    body = self._create_static(userData=k, **v)
                elif object_data["type"] == "dynamic":
                    body = self._create_dynamic(
                        userData=k, **v, **object_data["body_kwargs"]
                    )
                else:
                    raise NotImplementedError(
                        f"Cannot create rigid body of type {object_data['type']}"
                    )
                object_data["bodies"][k] = body
            self.env[object_name] = object_data

    def _clone(self):
        """Clone environment."""
        self._clone_env()
        self._clone_world()

    def _clone_env(self):
        """Clone ordered dictionary of environment objects and their attributes."""
        env = OrderedDict()
        for object_name, object_data in self.env.items():
            instance_keys = ["body_kwargs", "bodies"]
            _object_data = {
                k: deepcopy(v) for k, v in object_data.items() if k not in instance_keys
            }

            if object_data["type"] == "dynamic":
                _object_data["body_kwargs"] = object_data["body_kwargs"]

                assert (
                    len(object_data["bodies"]) == 1
                ), "Only support cloning of rigid bodies with one fixture"
                for body_name, body in object_data["bodies"].items():
                    _object_data["shapes"][body_name]["position"] = np.array(
                        body.position.copy(), dtype=np.float32
                    )
                    _body_kwargs = {
                        "angle": body.angle,
                        "linearVelocity": body.linearVelocity.copy(),
                        "angularVelocity": body.angularVelocity,
                        "linearDamping": body.linearDamping,
                        "angularDamping": body.angularDamping,
                        "awake": body.awake,
                        "fixedRotation": body.fixedRotation,
                        "bullet": body.bullet,
                        "active": body.active,
                        "gravityScale": body.gravityScale,
                    }
                _object_data["body_kwargs"].update(_body_kwargs)
            env[object_name] = _object_data

        self.env = env

    def _clone_world(self):
        """Clone Box2D world by constructing rigid bodies."""
        self.world = b2World()
        for object_name, object_data in self.env.items():
            object_data["bodies"] = {}

            for k, v in object_data["shapes"].items():
                if object_data["type"] == "static":
                    body = self._create_static(userData=k, **v)
                elif object_data["type"] == "dynamic":
                    body = self._create_dynamic(
                        userData=k, **v, **object_data["body_kwargs"]
                    )
                else:
                    raise NotImplementedError(
                        f"Cannot create rigid body of type {object_data['type']}"
                    )
                object_data["bodies"][k] = body
            self.env[object_name] = object_data

    def _create_static(self, position, box, userData=None):
        """Add static body to world.

        args:
            position: centroid position in world reference (m) -- np.array (2,)
            box: half_w, half_h box shape parameters (m) -- np.array (2,)
        """
        body = self.world.CreateStaticBody(
            position=position.astype(np.float64),
            shapes=b2PolygonShape(box=box.astype(np.float64)),
            userData=userData,
        )
        return body

    def _create_dynamic(
        self,
        position,
        box,
        density=1,
        friction=0.1,
        restitution=0.1,
        userData=None,
        **kwargs,
    ):
        """Add static body to world.

        args:
            position: centroid position in world reference (m) -- np.array (2,)
            box: half_w, half_h box shape parameters (m) -- np.array (2,)
            density: rigid body density (kg / m^2)
            friction: Coulumb friction coefficient
            restitution: rigid body restitution
            user_data: pointer to user specified data
            kwargs: additional key word arguments for b2BodyDef
        """
        body = self.world.CreateDynamicBody(
            position=position.astype(np.float64),
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(box=box.astype(np.float64)),
                density=density,
                friction=friction,
                restitution=restitution,
            ),
            userData=userData,
            **kwargs,
        )
        body.sleepingAllowed = False
        return body

    @staticmethod
    def _get_body_name(object_name, part_name):
        return f"{object_name}_{part_name}"

    def _get_class(self, object_name):
        """Return class of the rigid body object_name."""
        return self.env[object_name]["class"]

    def _get_type(self, object_name):
        """Return type of rigid body object_name."""
        return self.env[object_name]["type"]

    def _get_bodies(self, object_name):
        """Return rigid bodies attached to object_name."""
        return self.env[object_name]["bodies"]

    def _get_body(self, object_name, part_name):
        """Return part_name rigid body attached to object_name."""
        if "_" not in part_name:
            part_name = self._get_body_name(object_name, part_name)
        return self.env[object_name]["bodies"][part_name]

    def _get_shapes(self, object_name):
        """Return shapes attached to object_name."""
        return self.env[object_name]["shapes"]

    def _get_shape(self, object_name, part_name):
        """Return shape_name shape attached to object_name."""
        if "_" not in part_name:
            part_name = self._get_body_name(object_name, part_name)
        return self.env[object_name]["shapes"][part_name]

    def _get_shape_kwargs(self, object_name):
        """Return shape_kwargs used to construct object_name."""
        return self.env[object_name]["shape_kwargs"]

    def _get_color(self, object_name):
        """Return color for rendering rigid body object_name. Black is returned
        for rigid bodies with unspecified rendering parameters.
        """
        try:
            color = COLORS[self.env[object_name]["render_kwargs"]["color"]]
        except KeyError:
            color = COLORS["black"]
        return color
