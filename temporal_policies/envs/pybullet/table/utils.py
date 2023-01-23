import collections
from typing import Any, Dict, Optional, Union

from ctrlutils import eigen
import itertools
import numpy as np
import pybullet as p
from shapely.geometry import Polygon
import yaml

from temporal_policies.envs.pybullet.sim import body, math
from temporal_policies.envs.pybullet.table.objects import Object
from temporal_policies.envs.pybullet.table.primitives import ACTION_CONSTRAINTS


TABLE_CONSTRAINTS = {
    "table_z_max": 0.00,
    "table_x_min": 0.28,
    "table_y_min": -0.45,
    "table_y_max": 0.45,
    "workspace_x_min": 0.40,
    "operational_x_min": 0.50,
    "operational_x_max": 0.60,
    "obstruction_x_min": 0.575,
    "workspace_radius": 0.75,
}


EPSILONS = {"aabb": 0.05, "align": 0.99, "twist": 0.001, "tipping": 0.1}


def compute_margins(obj: Object, sim: bool = True) -> np.ndarray:
    """Compute the x-y margins of the object in the world frame."""
    aabb = obj.aabb(sim=sim)[:, :2]
    margin = 0.5 * (aabb[1] - aabb[0])
    return margin


def compute_object_pose(obj: Object, theta: float, sim: bool = True) -> math.Pose:
    """Computes a new pose for the object with the given theta."""
    aa = eigen.AngleAxisd(theta, np.array([0.0, 0.0, 1.0]))
    quat = eigen.Quaterniond(aa)
    pose = math.Pose(pos=obj.pose(sim=sim).pos, quat=quat.coeffs)
    return pose


def is_above(obj_a: Object, obj_b: Object, sim: bool = True) -> bool:
    """Returns True if the object a is above the object b."""
    min_child_z = obj_a.aabb(sim=sim)[0, 2]
    max_parent_z = obj_b.aabb(sim=sim)[1, 2]
    return min_child_z > max_parent_z - EPSILONS["aabb"]


def is_upright(obj: Object, sim: bool = True) -> bool:
    """Returns True if the child objects z-axis aligns with the world frame."""
    aa = eigen.AngleAxisd(eigen.Quaterniond(obj.pose(sim=sim).quat))
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


TWIST_HISTORY: Dict[str, Dict[Object, np.ndarray]] = collections.defaultdict(dict)


def is_moving(obj: Object, use_history: Optional[str] = None) -> bool:
    """Returns True if the object is moving.

    Args:
        obj: Object.
        use_history: A unique user-provided key that if set, will average the
            current velocity with the previous velocity from when this function
            was last called with the same key to decide whether the object is
            moving. This helps avoid reporting the object as moving when it is
            simply vibrating due to Pybullet instability. The unique key helps
            avoid interference between different functions calling
            `is_moving()`.
    """
    global TWIST_HISTORY
    twist = obj.twist()
    if use_history is not None:
        try:
            old_twist = TWIST_HISTORY[use_history][obj]
        except KeyError:
            old_twist = twist

        TWIST_HISTORY[use_history][obj] = twist
        twist = 0.5 * (twist + old_twist)

    return bool((np.abs(twist) >= EPSILONS["twist"]).any())


def is_below_table(obj: Object, sim: bool = True) -> bool:
    """Returns True if the object is below the table."""
    return obj.pose(sim=sim).pos[2] < TABLE_CONSTRAINTS["table_z_max"]


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


def is_intersecting(obj_a: Object, obj_b: Object, sim: bool = True) -> bool:
    """Returns True if object a intersects object b in the world x-y plane."""
    polygons_a = [
        Polygon(hull)
        for hull in obj_a.convex_hulls(world_frame=True, project_2d=True, sim=sim)
    ]
    polygons_b = [
        Polygon(hull)
        for hull in obj_b.convex_hulls(world_frame=True, project_2d=True, sim=sim)
    ]

    return any(
        poly_a.intersects(poly_b)
        for poly_a, poly_b in itertools.product(polygons_a, polygons_b)
    )


def is_under(obj_a: Object, obj_b: Object, sim: bool = True) -> bool:
    """Returns True if object a is underneath object b.

    Returns False if object b is not the rack.

    Otherwise, check if object a (that's not the table) is underneath the rack"""
    if "rack" != obj_b.name:
        return False
    if "table" in [obj_a.name, obj_b.name]:
        return False
    if not is_above(obj_a, obj_b, sim=sim) and is_intersecting(obj_a, obj_b, sim=sim):
        return True
    return False


def is_inhand(obj: Object) -> bool:
    """Returns True if the object is in the gripper."""
    z_pos = obj.pose().pos[2]
    z_min = ACTION_CONSTRAINTS["max_lift_height"] - obj.size[2] * 2
    return z_pos > z_min


def is_on(
    obj_a: Object, obj_b: Object, on_distance: float = 0.04, sim: bool = True
) -> bool:
    """Returns True if object a is on top of object b."""
    if (
        is_above(obj_a, obj_b, sim=sim)
        and is_intersecting(obj_a, obj_b, sim=sim)
        and not is_inhand(obj_a, sim=sim)
        and abs(obj_a.aabb(sim=sim)[0, 2] - obj_b.aabb(sim=sim)[1, 2]) < on_distance
    ):
        return True
    return False


def is_inhand(obj: Object, sim: bool = True) -> bool:
    """Returns True if the object is in the gripper."""
    z_pos = obj.pose(sim=sim).pos[2]
    z_min = ACTION_CONSTRAINTS["max_lift_height"] - obj.size[2] * 0.5
    return z_pos > z_min


def is_inworkspace(
    obj: Optional[Object] = None,
    obj_pos: Optional[np.ndarray] = None,
    distance: Optional[float] = None,
    sim: bool = True,
) -> bool:
    """Returns True if the object is in the workspace."""
    if obj_pos is None or distance is None:
        if obj is None:
            raise ValueError("Must specify obj or obj_pos and distance")
        obj_pos = obj.pose(sim=sim).pos[:2]
        distance = float(np.linalg.norm(obj_pos))
    if not (
        TABLE_CONSTRAINTS["workspace_x_min"] <= obj_pos[0]
        and distance < TABLE_CONSTRAINTS["workspace_radius"]
    ):
        return False

    return True


def is_beyondworkspace(
    obj: Optional[Object] = None,
    obj_pos: Optional[np.ndarray] = None,
    distance: Optional[float] = None,
    sim: bool = True,
) -> bool:
    """Returns True if the object is beyond the workspace."""
    if obj_pos is None or distance is None:
        if obj is None:
            raise ValueError("Must specify obj or obj_pos and distance")
        obj_pos = obj.pose(sim=sim).pos[:2]
        distance = float(np.linalg.norm(obj_pos))
    if distance < TABLE_CONSTRAINTS["workspace_radius"]:
        return False

    return True


def load_config(config: Union[str, Any]) -> Any:
    if isinstance(config, str):
        with open(config, "r") as f:
            config = yaml.safe_load(f)
    return config
