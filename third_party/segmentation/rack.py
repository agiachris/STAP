from typing import Tuple, Optional

from ctrlutils import eigen
import cv2  # type: ignore
import numpy as np  # type: ignore

from utils import ObjectInfo, PointCloud, extract_largest_contour, filter_point_cloud

BBOX_RACK = np.array(
    [
        (0.25, 1.0),  # (back, front) from robot's perspective.
        (-0.5, 0.5),  # (right, left).
        (0.05, 0.17),  # (bottom, top).
    ]
)

SIZE_RACK = np.array([0.22, 0.32, 0.16])
DEBUG = False


def segment_rack(
    img_color: np.ndarray, img_depth: np.ndarray, point_cloud: PointCloud
) -> Tuple[np.ndarray, Optional[ObjectInfo]]:
    # Filter color range.
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
    mask_rack = (10 < img_h) & (img_h < 30)
    mask_rack &= (10 < img_s) & (img_s < 220)
    mask_rack &= (30 < img_v) & (img_v < 190)

    # Select points within box.
    mask_rack &= filter_point_cloud(point_cloud, BBOX_RACK)

    if DEBUG:
        mask_rack_color = mask_rack

    ds = img_depth[mask_rack]
    if len(ds) == 0:
        return mask_rack, None
    d_min = ds.min() - 20
    d_max = ds.max() + 20
    mask_depth = (d_min <= img_depth) & (img_depth <= d_max)

    idx_points = point_cloud.flatten_indices(mask_rack)
    xyzs = point_cloud.points[idx_points]
    xyz_min = xyzs.min(axis=0)
    xyz_min[2] = BBOX_RACK[2][0]
    xyz_max = xyzs.max(axis=0)
    xyz_max[2] = BBOX_RACK[2][1]
    idxx_min = (xyz_min[None, :] <= point_cloud.points).all(axis=1)
    idxx_max = (point_cloud.points <= xyz_max[None, :]).all(axis=1)

    mask_xyz = point_cloud.expand_indices(idxx_min & idxx_max)

    mask_rack = mask_depth & mask_xyz

    mask_rack = extract_largest_contour(mask_rack)

    # Get object pose.
    pos = xyzs.mean(axis=0)
    pos[2] = SIZE_RACK[2]
    if pos[0] < 0.7:
        pos[:2] = np.array([0.44, -0.33])
        quat = eigen.Quaterniond(eigen.AngleAxisd(np.pi / 2, np.array([0.0, 0.0, 1.0])))
    else:
        pos[:2] = np.array([0.82, 0.0])
        quat = eigen.Quaterniond.identity()

    if DEBUG:
        cv2.imshow("Color", img_color)
        cv2.imshow("h", img_h)
        cv2.imshow("s", img_s)
        cv2.imshow("v", img_v)
        cv2.imshow("color", mask_rack_color.astype(np.uint8) * 255)
        cv2.imshow("depth", mask_depth.astype(np.uint8) * 255)
        cv2.imshow("xyz", mask_xyz.astype(np.uint8) * 255)
        cv2.imshow("rack", mask_rack.astype(np.uint8) * 255)
        key = cv2.waitKey(0)
        if key is not None and chr(key) == "q":
            exit(1)

    return mask_rack, ObjectInfo(pos=pos, quat=quat, size=SIZE_RACK)
