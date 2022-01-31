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
        self.env = OrderedDict(ENV_OBJECTS.copy(), **self._env_params)
        for o in self.env.keys():
            assert self.env[o]["class"] in GeometryHandler._VALID_CLASSES
        self.vectorize(self.env)
    
    def _setup_world(self):
        """Setup Box2D world by constructing rigid bodies.
        """
        self.world = b2World()
        for o in self.env:
            obj = self.env[o]
            obj["shapes"] = getattr(self, obj["class"])(**obj["shape_kwargs"])
            obj["bodies"] = {}
            for k, v in obj["shapes"].items():
                obj["bodies"][k] = getattr(self, "_create_{}".format(obj["type"]))(**v, **obj["body_kwargs"])
            self.env[o] = obj

    def _create_static(self, position, box):
        """Add static body to world.
        
        args: 
            position: tuple(x, y) centroid position in world reference (m)
            box: tuple(half_w, half_h) box shape parameters
        """
        body = self.world.CreateStaticBody(
            position=position,
            shapes=b2PolygonShape(box=box)
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
