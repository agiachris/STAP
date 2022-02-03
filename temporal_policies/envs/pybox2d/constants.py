ENV_OBJECTS = {
    "playground": {
        "class": "workspace",
        "type": "static",
        "shape_kwargs": {
            "size": (12, 8),
            "t": 0.1
        },
        "render_kwargs": {
            "color": "black"
        }
    },
    "box": {
        "class": "receptacle",
        "type": "static",
        "shape_kwargs": {
            "size": (3, 3),
            "t": 0.1,
            "config": -1,
            "dx": -3.5
        },
        "render_kwargs": {
            "color": "brown"
        }
    },
    "obstacle": {
        "class": "block",
        "type": "static",
        "shape_kwargs": {
            "size": (1, 1.5),
            "dx": 3
        },
        "render_kwargs": {
            "color": "red"
        }
    },
    "item": {
        "class": "block",
        "type": "dynamic",
        "shape_kwargs": {
            "size": (1, 1.5),
            "dy": 5
        },
        "body_kwargs": {
            "density": 1,
            "friction": 0.5,
            "restitution": 0.0
        },
        "render_kwargs": {
            "color": "emerald"
        }
    },
}


ENV_RANDOM_PARAMS = {
    "h_receptacle": [0.5, 1.5],     # h (m) receptacle height scale range
    "c_receptacle": {-1, 1},        # receptacle opening, c = -1 (right), c = 1 (left)
    "h_obstacle": [0.5, 1.5],       # h (m) obstacle height scale range
    "w_block": [0.5, 1.5],          # w (m) block width scale range
    "h_block": [0.5, 1.5]           # h (m) block height scale range
}


COLORS = {
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "rouge": (128, 0, 0),
    "green": (0, 255, 0),
    "emerald": (0, 128, 0),
    "blue": (0, 0, 255),
    "navy": (0, 0, 128),
    "brown": (100, 42, 42), 
    "black": (0, 0, 0)
}
