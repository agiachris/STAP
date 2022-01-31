import numpy as np


class GeometryHandler(object):

    _VALID_CLASSES = ["workspace", "receptacle", "block"]

    def __init__(self, global_x=0, global_y=0):
        self._t_global = np.array([global_x, global_y])
        
    @staticmethod
    def vectorize(objects):
        for o in objects.keys():
            objects[o]["shape_kwargs"]["size"] = np.array(objects[o]["shape_kwargs"]["size"])

    def transform_global(self, shapes):
        for s in shapes.keys():
            shapes[s]["position"] += self._t_global

    def workspace(self, size, t=0.1):
        """Compute shape parameters for the environment workspace.
        args:
            size: workspace size w x h (m)
            t: polygon thickness (m) (default: 0.1)
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
