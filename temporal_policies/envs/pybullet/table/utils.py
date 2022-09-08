from typing import Optional, Tuple

import itertools
import numpy as np
from ctrlutils import eigen
import pybullet as p
from shapely.geometry import Polygon

from temporal_policies.envs.pybullet.table.objects import Object
from temporal_policies.envs.pybullet.sim import body


TABLE_CONSTRAINTS = {
    "table_z_max": 0.00,
    "table_x_min": 0.30,
    "workspace_x_min": 0.40,
    "operational_x_min": 0.50,
    "operational_x_max": 0.60,
    "obstruction_x_min": 0.575,
    "workspace_radius": 0.7,
}


EPSILONS = {"aabb": 0.01, "align": 0.99, "twist": 0.001, "tipping": 0.1}


def is_above(obj_a: Object, obj_b: Object) -> bool:
    """Returns True if the object a is above the object b."""
    min_child_z = obj_a.aabb()[0, 2]
    max_parent_z = obj_b.aabb()[1, 2]
    return min_child_z > max_parent_z - EPSILONS["aabb"]


def is_upright(obj: Object) -> bool:
    """Returns True if the child objects z-axis aligns with the world frame."""
    aa = eigen.AngleAxisd(eigen.Quaterniond(obj.pose().quat))
    return abs(aa.axis.dot(np.array([0.0, 0.0, 1.0]))) >= EPSILONS["align"]


def is_within_distance(
    obj_a: Object, obj_b: Object, distance: float, physics_id: int
) -> bool:
    """Returns True if the closest points between two objects are within distance."""
    return bool(
        p.getClosestPoints(
            obj_a.body_id, obj_b.body_id, distance, physicsClientId=physics_id
        )
    )


def is_moving(obj: Object) -> bool:
    """Returns True if the object is moving."""
    return bool((np.abs(obj.twist()) >= EPSILONS["twist"]).any())


def is_below_table(obj: Object) -> bool:
    """Returns True if the object is below the table."""
    return obj.pose().pos[2] < TABLE_CONSTRAINTS["table_z_max"]


def is_touching(
    body_a: body.Body,
    body_b: body.Body,
    link_id_a: Optional[int] = None,
    link_id_b: Optional[int] = None,
) -> bool:
    """Returns True if there are any contact points between the two bodies."""
    assert body_a.physics_id == body_b.physics_id
    kwargs = {}
    if link_id_a is not None:
        kwargs["linkIndexA"] = link_id_a
    if link_id_b is not None:
        kwargs["linkIndexB"] = link_id_b
    contacts = p.getContactPoints(
        bodyA=body_a.body_id,
        bodyB=body_b.body_id,
        physicsClientId=body_a.physics_id,
        **kwargs,
    )
    return len(contacts) > 0


def is_intersecting(obj_a: Object, obj_b: Object) -> bool:
    """Returns True if object a intersects object b in the world x-y plane."""
    polygons_a = [
        Polygon(hull) for hull in obj_a.convex_hulls(world_frame=True, project_2d=True)
    ]
    polygons_b = [
        Polygon(hull) for hull in obj_b.convex_hulls(world_frame=True, project_2d=True)
    ]

    return any(
        poly_a.intersects(poly_b)
        for poly_a, poly_b in itertools.product(polygons_a, polygons_b)
    )


def is_under(obj_a: Object, obj_b: Object) -> bool:
    """Returns True if object a is underneath object b."""
    if not is_above(obj_a, obj_b) and is_intersecting(obj_a, obj_b):
        return True
    return False


def is_inworkspace(
    obj: Optional[Object] = None,
    obj_pos: Optional[np.ndarray] = None,
    distance: Optional[np.ndarray] = None,
) -> bool:
    """Returns True if the objects is in the workspace."""
    if obj_pos is None or distance is None:
        if obj is None:
            raise ValueError("Must specify obj or obj_pos and distance")
        obj_pos = obj.pose().pos[:2]
        distance = float(np.linalg.norm(obj_pos))
    if not (
        TABLE_CONSTRAINTS["workspace_x_min"] <= obj_pos[0]
        and distance < TABLE_CONSTRAINTS["workspace_radius"]
    ):
        return False

    return True
