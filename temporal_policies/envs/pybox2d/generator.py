from Box2D import *
from .utils import GeometryHandler
from .constants import *


class Generator(GeometryHandler):

    def __init__(self, env_objects={}, rand_params={}, geometry_params={}):
        """PyBox2D environment generator.
        """
        super().__init__(**geometry_params)

        self.world = None
        self.env_objects = dict(ENV_OBJECTS.copy(), **env_objects)
        for o in self.env_objects.keys():
            assert self.env_objects[o]["class"] in GeometryHandler._VALID_CLASSES
        self.vectorize(self.env_objects)
        self._rand_params = rand_params
        self._geometry_params = geometry_params

    def __iter__(self):
        """Declare class as an iterator.
        """
        return self

    def __next__(self):
        """Return a randomly sampled 2D environment.
        """
        self.world = b2World()
        for o in self.env_objects:
            obj = self.env_objects[o]
            obj["shapes"] = getattr(self, obj["class"])(**obj["shape_kwargs"])
            obj["bodies"] = {}
            for k, v in obj["shapes"].items():
                obj["bodies"][k] = getattr(self, "create_{}".format(obj["type"]))(**v, **obj["body_kwargs"])
            self.env_objects[o] = obj

    def create_static(self, position, box):
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
    
    def create_dynamic(self, position, box, 
                       density=1,
                       friction=0.1,
                       restitution=0.5,
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
