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
        self._t_global = np.array([global_x, global_y])
        
    @staticmethod
    def vectorize(objects):
        """Convert shape "size" configuration tuples into numpy arrays.
        """
        for o in objects.keys():
            objects[o]["shape_kwargs"]["size"] = np.array(objects[o]["shape_kwargs"]["size"])

    def transform_global(self, shapes):
        """Transform shape parameters into global reference frame. 
        """
        for s in shapes.keys():
            shapes[s]["position"] += self._t_global

    def workspace(self, size, t=0.1):
        """Compute shape parameters for the environment workspace.

        args:
            size: workspace size w x h (m)
            t: polygon thickness (m) (default: 0.1)
        returns: 
            shapes: polygon shape parameters
        """
        (h_w, h_h), h_t = size / 2, t / 2
        shapes = {
            "ground": {
                "position": np.array([0, -h_t]),
                "box": np.array([h_w + t, h_t])
            },
            "left_wall": {
                "position": np.array([-(h_w + h_t), h_h]),
                "box": np.array([h_t, h_h])
            },
            "right_wall": {
                "position": np.array([h_w + h_t, h_h]),
                "box": np.array([h_t, h_h])
            }
        }
        self.transform_global(shapes)
        return shapes

    def receptacle(self, size, config=-1, t=0.1, dx=0.0):
        """Compute shape parameters for the receptacle container.
        args:
            size: receptacle size w x h (m)
            config: wall configuration, 1 = right wall, -1 = left wall (default: -1)
            t: polygon thickness (m) (default: 0.1)
            dx: workspace x-axis offset (default: 0.0)
        returns: 
            shapes: polygon shape parameters
        """
        w, h = size
        (h_w, h_h), h_t = size / 2, t / 2
        shapes = {
            "ceiling": {
                "position": np.array([0 + dx, h + h_t]),
                "box": np.array([h_w, h_t])
            },
            "wall": {
                "position": np.array([config * (h_w + h_t) + dx, h_h + h_t]),
                "box": np.array([h_t, h_h + h_t])
            }
        }
        self.transform_global(shapes)
        return shapes

    def block(self, size, dx=0.0, dy=0.0):
        """Compute shape parameters of a block.
        args:
            size: block size w x h (m)
            dx: workspace x-axis offset (default: 0.0)
            dy: workspace y-axis offset (default: 0.0)
        returns: 
            shapes: polygon shape parameters
        """
        h_w, h_h = size / 2
        shapes = {
            "block": {
                "position": np.array([dx, h_h + dy]),
                "box": np.array([h_w, h_h])
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
        transform: 2D rigid body transform
    """
    transform = np.eye(3, dtype=float)
    transform[:2, :2] = np.array([[np.cos(theta), -np.sin(theta)],
                                  [np.sin(theta), np.cos(theta)]])
    transform[:2, 2] = np.array([tx, ty]) * r
    return transform


def shape_to_vertices(position, box):
    """Computes rectangle vertices from shape parameters.
    args:
        position: 2D shape centroid (m)
        box: half width and half height of the shape (m)
    returns:
        vertices: np.array of rectangle vertices in CCW order
    """
    v1 = np.array([position[0] - box[0], position[1] - box[1]])
    v2 = np.array([position[0] + box[0], position[1] - box[1]])
    v3 = np.array([position[0] + box[0], position[1] + box[1]])
    v4 = np.array([position[0] - box[0], position[1] + box[1]])
    vertices = np.array([v1, v2, v3, v4]).astype(float)
    return vertices
