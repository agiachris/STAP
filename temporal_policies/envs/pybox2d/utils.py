import numpy as np


class GeometryHandler(object):

    _VALID_CLASSES = ["workspace", "receptacle", "block"]

    def __init__(self, global_x=0, global_y=0):
        """Class for constructing Box2D shape parameters from user-specified configs.
        Below are the currently supported shapes.
            - workspace: ground, left wall, right wall
            - receptacle: left or right wall, and ceiling
            - block: solid rectangle

        args:
            global_x: x-dim workspace frame translation w.r.t global (m)
            global_y: y-dim workspace frame translation w.r.t global (m)
        """
        self._t_global = np.array([global_x, global_y], dtype=np.float32)

    def vectorize(self, env):
        """Convert shape "size" configuration tuples into numpy arrays."""
        for object_name, object_data in env.items():
            size = object_data["shape_kwargs"]["size"]
            object_data["shape_kwargs"]["size"] = np.array(size, dtype=np.float32)
            # Set default global reference frame to workspace bottom left
            if env[object_name]["class"] == "workspace" and np.all(
                self._t_global == np.zeros_like(self._t_global)
            ):
                self._t_global = np.array([size[0] * 0.5, 0], dtype=np.float32)

    def transform_global(self, shapes):
        """Transform shape parameters into global reference frame."""
        for s in shapes.keys():
            shapes[s]["position"] += self._t_global

    def workspace(self, name, size, t=0.1):
        """Compute shape parameters for the environment workspace.

        args:
            size: workspace size w x h (m) -- np.array (2,)
            t: polygon thickness (m) (default: 0.1)
        returns:
            shapes: polygon shape parameters
        """
        (h_w, h_h), h_t = size * 0.5, t * 0.5
        shapes = {
            "{}_ground".format(name): {
                "position": np.array([0, -h_t], dtype=np.float32),
                "box": np.array([h_w + t, h_t], dtype=np.float32),
            },
            "{}_left_wall".format(name): {
                "position": np.array([-(h_w + h_t), h_h], dtype=np.float32),
                "box": np.array([h_t, h_h], dtype=np.float32),
            },
            "{}_right_wall".format(name): {
                "position": np.array([h_w + h_t, h_h], dtype=np.float32),
                "box": np.array([h_t, h_h], dtype=np.float32),
            },
        }
        self.transform_global(shapes)
        return shapes

    def receptacle(self, name, size, config=-1, t=0.1, dx=0.0):
        """Compute shape parameters for the receptacle container.
        args:
            size: receptacle size w x h (m) -- np.array (2,)
            config: wall configuration, 1 = right wall, -1 = left wall (default: -1)
            t: polygon thickness (m) (default: 0.1)
            dx: workspace x-axis offset (default: 0.0)
        returns:
            shapes: polygon shape parameters
        """
        w, h = size
        (h_w, h_h), h_t = size * 0.5, t * 0.5
        shapes = {
            "{}_ceiling".format(name): {
                "position": np.array([0 + dx, h + h_t], dtype=np.float32),
                "box": np.array([h_w, h_t], dtype=np.float32),
            },
            "{}_wall".format(name): {
                "position": np.array(
                    [config * (h_w + h_t) + dx, h_h + h_t], dtype=np.float32
                ),
                "box": np.array([h_t, h_h + h_t], dtype=np.float32),
            },
        }
        self.transform_global(shapes)
        return shapes

    def block(self, name, size, dx=0.0, dy=0.0):
        """Compute shape parameters of a block.
        args:
            size: block size w x h (m) -- np.array(2,)
            dx: workspace x-axis offset (default: 0.0)
            dy: workspace y-axis offset (default: 0.0)
        returns:
            shapes: polygon shape parameters
        """
        h_w, h_h = size * 0.5
        shapes = {
            "{}_block".format(name): {
                "position": np.array([dx, h_h + dy], dtype=np.float32),
                "box": np.array([h_w, h_h], dtype=np.float32),
            }
        }
        self.transform_global(shapes)
        return shapes


def rigid_body_2d(theta, tx, ty, r=1):
    """Construct a 2D rigid body homogeneous transform.
    All parameters are expressed in world coordinates.
    args:
        theta: rotation between frames
        tx: translation between frames
        ty: translation between frames
        r: resolution scaling
    returns:
        transform: 2D rigid body transform -- np.array (3, 3)
    """
    transform = np.eye(3)
    transform[:2, :2] = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    transform[:2, 2] = np.array([tx, ty]) * r
    return transform.astype(np.float32)


def to_homogenous(points):
    """Convert 2D or 3D coordinates to augmented coordinates such that they
    can be used for a homogenous transform.

    args:
        points: 2D or 3D coordinates -- np.array (N, D)
    returns:
        augmented_points: 2D or 3D homogenous coordinates -- np.array (N, D+1)
    """
    is_transposed = False
    if points.shape[0] == 2:
        is_transposed = True
        points = points.T
    ones = np.ones((points.shape[0], 1))
    augmented_points = np.concatenate((points, ones), axis=1)
    if is_transposed:
        augmented_points = augmented_points.T
    return augmented_points.astype(np.float32)


def shape_to_vertices(position, box):
    """Computes rectangle vertices from shape parameters.
    args:
        position: 2D shape centroid (m) -- np.array (2,)
        box: half width and half height of the shape (m) -- np.array (2,)
    returns:
        vertices: np.array of rectangle vertices in CCW order -- np.array (4, 2)
    """
    v1 = np.array([position[0] - box[0], position[1] - box[1]])
    v2 = np.array([position[0] + box[0], position[1] - box[1]])
    v3 = np.array([position[0] + box[0], position[1] + box[1]])
    v4 = np.array([position[0] - box[0], position[1] + box[1]])
    vertices = np.array([v1, v2, v3, v4])
    return vertices.astype(np.float32)


def sample_random_params(v):
    """Sample from discrete or continuous distribution uniformly based
    based on parameters v.
    """
    if isinstance(v, list) and isinstance(v[0], list):
        # multi-variate discrete or continuous sampling distribution
        v = np.array([sample_random_params(_v) for _v in v])
    elif isinstance(v, list) and isinstance(sum(v), int):
        # discrete sampling distribution
        v = np.random.choice(v)
    elif isinstance(v, list) and isinstance(sum(v), float):
        # continuous uniform sampling distribution
        assert len(v) == 2
        v = np.random.uniform(v[0], v[1])
    else:
        raise ValueError("Incorrect specification of randomization bounds.")
    return v
