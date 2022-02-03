from Box2D import *
import numpy as np
from collections import OrderedDict

from .utils import GeometryHandler
from .constants import *


class Generator(GeometryHandler):

    def __init__(self, 
                 env=None,
                 world=None,
                 env_params={}, 
                 rand_params={}, 
                 geometry_params={}):
        """PyBox2D environment generator.
        """
        super().__init__(**geometry_params)
        self._env_params = env_params
        self._rand_params = rand_params
        self._geometry_params = geometry_params
        
        # Public attributes: Box2D world and environment parameters
        self.env = env
        self.world = world
        if env is None:
            assert world is None
            next(self)

    def __iter__(self):
        """Declare class as an iterator.
        """
        return self

    def __next__(self):
        """Return a randomly sampled 2D environment.
        """
        self._setup_env()
        self._setup_world()

    def _setup_env(self):
        """Setup ordered dictionary of environment objects and their attributes.
        """
        env = dict(ENV_OBJECTS.copy(), **self._env_params.copy())
        env = OrderedDict(sorted(env.items()))
        for object_name in env.keys():
            assert env[object_name]["class"] in GeometryHandler._VALID_CLASSES

        # Light domain randomization
        for object_name, rand_params in self._rand_params.items():
            for k, v in rand_params["shape_kwargs"].items():
                if isinstance(v, list) and isinstance(sum(v), int):
                    sample = np.random.choice(v)
                elif isinstance(v, list) and isinstance(sum(v), float):
                    sample = np.random.uniform(v[0], v[1])
                elif isinstance(v, list) and isinstance(v[0], list):
                    sample = [np.random.uniform(_v[0], _v[1]) for _v in zip(v)]
                else:
                    raise ValueError("Incorrect specification of randomization bounds.")
                env[object_name]["shape_kwargs"][k] = sample

        self.env = env
        self.vectorize(self.env)
    
    def _setup_world(self):
        """Setup Box2D world by constructing rigid bodies.
        """
        self.world = b2World()
        for object_name, object_data in self.env.items():
            create_shape_fn = getattr(self, object_data["class"])
            object_data["shapes"] = create_shape_fn(object_name, **object_data["shape_kwargs"])
            object_data["bodies"] = {}
            for k, v in object_data["shapes"].items():
                if object_data["type"] == "static":
                    bodies = self._create_static(userData=k, **v)
                elif object_data["type"] == "dynamic":
                    bodies = self._create_dynamic(userData=k, **v, **object_data["body_kwargs"])
                else:
                    raise NotImplementedError("Cannot create rigid body of type {}".format(object_data["type"]))
                object_data["bodies"][k] = bodies
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
            userData=userData
        )
        return body
    
    def _create_dynamic(self, position, box, 
                       density=1,
                       friction=0.1,
                       restitution=0.1,
                       userData=None
                       ):
        """Add static body to world.
        
        args: 
            position: centroid position in world reference (m) -- np.array (2,)
            box: half_w, half_h box shape parameters (m) -- np.array (2,)
            density: rigid body density (kg / m^2)
            friction: Coulumb friction coefficient
            restitution: rigid body restitution
            user_data: pointer to user specified data
        """
        body = self.world.CreateDynamicBody(
            position=position.astype(np.float64),
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(box=box.astype(np.float64)),
                density=density,
                friction=friction,
                restitution=restitution
            ),
            userData=userData,
        )
        return body    

    @staticmethod
    def _get_body_name(object_name, part_name):
        return "{}_{}".format(object_name, part_name)

    def _get_body(self, object_name, part_name):
        """Return part_name rigid body attached to object_name.
        """
        body_name = self._get_body_name(object_name, part_name)
        return self.env[object_name]["bodies"][body_name]

    def _get_shape(self, object_name, part_name):
        """Return shape_name shape attached to object_name.
        """
        shape_name = self._get_body_name(object_name, part_name)
        return self.env[object_name]["shapes"][shape_name]

    def _get_shape_kwargs(self, object_name):
        """Return shape_kwargs used to construct object_name.
        """
        return self.env[object_name]["shape_kwargs"]
    