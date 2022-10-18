from typing import Tuple, Optional

import cv2  # type: ignore
from ctrlutils import eigen  # type: ignore
import numpy as np  # type: ignore

from utils import ObjectInfo, PointCloud, extract_largest_contour

from rack import SIZE_RACK

SIZE_MILK = np.array([0.055, 0.055, 0.08])
DEBUG = False


def segment_milk(
    img_color: np.ndarray,
    img_depth: np.ndarray,
    point_cloud: PointCloud,
    mask_others: np.ndarray,
) -> Tuple[np.ndarray, Optional[ObjectInfo]]:
    # Filter color range.
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
    mask_milk = 165 < img_h  # | (img_h < 5)
    mask_milk &= 30 < img_s
    mask_milk &= (100 < img_v) & (img_v < 245)

    mask_milk &= ~mask_others

    if DEBUG:
        mask_milk_color = mask_milk

    mask_milk = extract_largest_contour(mask_milk)

    ds = img_depth[mask_milk]
    if len(ds) == 0:
        return mask_milk, None
    d_min = ds.min() - 10
    d_max = ds.max()  # + 20
    mask_depth = (d_min <= img_depth) & (img_depth <= d_max)

    idx_points = point_cloud.flatten_indices(mask_milk)
    xyzs = point_cloud.points[idx_points]
    xyz_min = xyzs.min(axis=0) - np.array([0.03, 0.0, 0.0])
    xyz_max = xyzs.max(axis=0) + np.array([0.0, 0.0, 0.03])
    idxx_min = (xyz_min[None, :] <= point_cloud.points).all(axis=1)
    idxx_max = (point_cloud.points <= xyz_max[None, :]).all(axis=1)
    mask_xyz = point_cloud.expand_indices(idxx_min & idxx_max)

    mask_milk = mask_depth & mask_xyz & ~mask_others

    mask_milk = extract_largest_contour(mask_milk)

    # Get box pose.
    pos = xyzs.mean(axis=0)

    u, s, vt = np.linalg.svd(xyzs[:, :2] - pos[None, :2], full_matrices=False)
    # Assume x is always positive.
    xy_normal = vt[-1] if vt[-1, 0] > 0 else -vt[-1]
    theta = np.arctan2(xy_normal[1], xy_normal[0])
    aa = eigen.AngleAxisd(angle=theta, axis=np.array([0, 0, 1]))
    quat = eigen.Quaterniond(aa)

    pos[0] -= SIZE_MILK[0] / 2 + 0.01
    pos[1] -= 0.01
    if pos[2] > SIZE_RACK[2]:
        pos[2] = SIZE_RACK[2] + 0.5 * SIZE_MILK[2]
    else:
        pos[2] = 0.5 * SIZE_MILK[2]

    if DEBUG:
        cv2.imshow("Color", img_color)
        cv2.imshow("h", img_h)
        cv2.imshow("s", img_s)
        cv2.imshow("v", img_v)
        cv2.imshow("color", mask_milk_color.astype(np.uint8) * 255)
        cv2.imshow("depth", mask_depth.astype(np.uint8) * 255)
        cv2.imshow("xyz", mask_xyz.astype(np.uint8) * 255)
        cv2.imshow("milk", mask_milk.astype(np.uint8) * 255)
        key = cv2.waitKey(0)
        if key is not None and chr(key) == "q":
            exit(1)

    return mask_milk, ObjectInfo(pos, quat, SIZE_MILK)
