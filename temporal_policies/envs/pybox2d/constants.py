ENV_OBJECTS = {
    "playground": {
        "class": "workspace",
        "type": "static",
        "shape_kwargs": {
            "size": (8, 6),
            "t": 0.1
        },
        "body_kwargs": {}
    },
    "box": {
        "class": "receptacle",
        "type": "static",
        "shape_kwargs": {
            "size": (2, 2),
            "t": 0.1,
            "config": -1,
            "dx": -2
        },
        "body_kwargs": {}
    },
    "obstacle": {
        "class": "block",
        "type": "static",
        "shape_kwargs": {
            "size": (1, 2),
            "dx": 3
        },
        "body_kwargs": {}
    },
    "item": {
        "class": "block",
        "type": "dynamic",
        "shape_kwargs": {
            "size": (1, 2),
            "dy": 4
        },
        "body_kwargs": {
            "density": 1,
            "friction": 0.1,
            "restitution": 0.25
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
