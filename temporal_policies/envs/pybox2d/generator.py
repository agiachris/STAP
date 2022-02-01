from collections import OrderedDict
from Box2D import *

from .utils import GeometryHandler
from .constants import *


class Generator(GeometryHandler):

    def __init__(self, env_params={}, rand_params={}, geometry_params={}):
        """PyBox2D environment generator.
        """
        super().__init__(**geometry_params)
        self._env_params = env_params
        self._rand_params = rand_params
        self._geometry_params = geometry_params
        
        # Public attributes: Box2D world and environment parameters
        self.world = None
        self.env = None
        self.__next__()

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
        # TODO: Domain randomization takes place here.
        env = dict(ENV_OBJECTS.copy(), **self._env_params)
        self.env = OrderedDict(sorted(env.items()))
        for o in self.env.keys():
            assert self.env[o]["class"] in GeometryHandler._VALID_CLASSES
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
                create_body_fn = getattr(self, "_create_{}".format(object_data["type"]))
                object_data["bodies"][k] = create_body_fn(userData=k, **v, **object_data["body_kwargs"])
            self.env[object_name] = object_data

    def _create_static(self, position, box, userData=None):
        """Add static body to world.
        
        args: 
            position: tuple(x, y) centroid position in world reference (m)
            box: tuple(half_w, half_h) box shape parameters
        """
        body = self.world.CreateStaticBody(
            position=position,
            shapes=b2PolygonShape(box=box),
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
            position: tuple(x, y) centroid position in world reference (m)
            box: tuple(half_w, half_h) box shape parameters
            user_data: pointer to user specified data
        """
        body = self.world.CreateDynamicBody(
            position=position,
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(box=box),
                density=density,
                friction=friction,
                restitution=restitution
            ),
            userData=userData
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
    