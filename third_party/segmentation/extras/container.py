from typing import Tuple, Optional

import cv2  # type: ignore
from ctrlutils import eigen  # type: ignore
import numpy as np  # type: ignore

from utils import ObjectInfo, PointCloud, extract_largest_contour

SIZE_CONTAINER = np.array([0.13, 0.13, 0.06])
DEBUG = False


def segment_container(
    img_color: np.ndarray,
    img_depth: np.ndarray,
    point_cloud: PointCloud,
    mask_others: np.ndarray,
) -> Tuple[np.ndarray, Optional[ObjectInfo]]:
    # Filter color range.
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
    mask_container = (15 < img_h) & (img_h < 40)
    mask_container &= (10 < img_s) & (img_s < 230)
    mask_container &= (5 < img_v) & (img_v < 100)
    mask_container &= ~mask_others

    if DEBUG:
        mask_container_color = mask_container

    # Get largest contour.
    while True:
        mask_container_temp = extract_largest_contour(mask_container)
        idx_points = point_cloud.flatten_indices(mask_container_temp)
        xyzs = point_cloud.points[idx_points]
        xyz_min = xyzs.min(axis=0)
        xyz_max = xyzs.max(axis=0)
        if xyz_max[0] - xyz_min[0] < 0.3:
            mask_container = mask_container_temp
            break
        mask_container &= ~mask_container_temp

    # ds = img_depth[mask_container]
    # if len(ds) == 0:
    #     return mask_container, None
    # d_min = ds.min() - 20
    # d_max = ds.max() + 20
    # mask_depth = (d_min <= img_depth) & (img_depth <= d_max)

    # Exclude objects inside container.
    idx_points = point_cloud.flatten_indices(mask_container)
    xyzs = point_cloud.points[idx_points]
    xyz_min = xyzs.min(axis=0)
    xyz_max = xyzs.max(axis=0)  # + np.array([0.01, 0, 0.02])
    idxx_interior = (point_cloud.points > xyz_min + np.array([0.12, 0.04, 0.02])).all(
        axis=1
    )
    idxx_interior &= (point_cloud.points < xyz_max - np.array([0.04, 0.04, 0])).all(
        axis=1
    )
    mask_xyz = point_cloud.expand_indices(~idxx_interior)
    mask_container &= mask_xyz
    # xyz_min[2] = max(xyz_max[2] - 0.24, 0)
    # idxx_min = (xyz_min[None, :] <= point_cloud.points).all(axis=1)
    # idxx_max = (point_cloud.points <= xyz_max[None, :]).all(axis=1)
    # mask_xyz = point_cloud.expand_indices(idxx_min & idxx_max)

    # mask_container = mask_depth & mask_xyz

    # # Get largest contour.
    # mask_container = extract_largest_contour(mask_container)

    # Get box pose.
    pos = xyzs.mean(axis=0)
    pos[2] = xyz_min[2] + SIZE_CONTAINER[2] / 2

    u, s, vt = np.linalg.svd(xyzs[:, :2] - pos[None, :2], full_matrices=False)
    xy_normal = vt[-1]
    theta = np.arctan2(xy_normal[1], xy_normal[0])
    aa = eigen.AngleAxisd(angle=theta, axis=np.array([0, 0, 1]))
    quat = eigen.Quaterniond(aa)
    quat = eigen.Quaterniond.identity()

    if DEBUG:
        cv2.imshow("Color", img_color)
        cv2.imshow("h", img_h)
        cv2.imshow("s", img_s)
        cv2.imshow("v", img_v)
        cv2.imshow("color", mask_container_color.astype(np.uint8) * 255)
        cv2.imshow("container", mask_container.astype(np.uint8) * 255)
        # cv2.imshow("others", mask_others.astype(np.uint8) * 255)
        # cv2.imshow("depth", mask_depth.astype(np.uint8) * 255)
        # cv2.imshow("xyz", mask_xyz.astype(np.uint8) * 255)
        key = cv2.waitKey(0)
        if key is not None and chr(key) == "q":
            exit(1)

    return mask_container, ObjectInfo(pos, quat, SIZE_CONTAINER)
