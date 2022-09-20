from typing import Tuple, Optional

import cv2  # type: ignore
from ctrlutils import eigen  # type: ignore
import numpy as np  # type: ignore

from utils import ObjectInfo, PointCloud, extract_largest_contour

SIZE_BAR = np.array([0.04, 0.15, 0.14])


def segment_bar(
    img_color: np.ndarray,
    img_depth: np.ndarray,
    point_cloud: PointCloud,
    mask_others: np.ndarray,
) -> Tuple[np.ndarray, Optional[ObjectInfo]]:
    # Filter color range.
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
    mask_bar = (10 < img_h) & (img_h < 25)
    mask_bar &= (30 < img_s) & (img_s < 180)
    mask_bar &= (60 < img_v) & (img_v < 170)
    mask_bar &= ~mask_others

    # mask_bar_color = mask_bar

    mask_bar = extract_largest_contour(mask_bar)

    ds = img_depth[mask_bar]
    if len(ds) == 0:
        return mask_bar, None
    d_min = ds.min() - 50
    d_max = ds.max() + 20
    mask_depth = (d_min <= img_depth) & (img_depth <= d_max)

    idx_points = point_cloud.flatten_indices(mask_bar)
    xyzs = point_cloud.points[idx_points]
    xyz_min = xyzs.min(axis=0) - np.array([0.02, 0.0, 0])
    xyz_max = xyzs.max(axis=0) + np.array([0.02, 0.0, 0.07])
    idxx_min = (xyz_min[None, :] <= point_cloud.points).all(axis=1)
    idxx_max = (point_cloud.points <= xyz_max[None, :]).all(axis=1)
    mask_xyz = point_cloud.expand_indices(idxx_min & idxx_max)

    mask_bar = mask_depth & mask_xyz

    mask_bar = extract_largest_contour(mask_bar)

    # Get box pose.
    pos = xyzs.mean(axis=0)
    pos[2] = xyz_min[2] + SIZE_BAR[2] / 2

    u, s, vt = np.linalg.svd(xyzs[:, :2] - pos[None, :2], full_matrices=False)
    xy_normal = vt[-1]
    theta = np.arctan2(xy_normal[1], xy_normal[0])
    aa = eigen.AngleAxisd(angle=theta, axis=np.array([0, 0, 1]))
    quat = eigen.Quaterniond(aa)

    # cv2.imshow("Color", img_color)
    # cv2.imshow("h", img_h)
    # cv2.imshow("s", img_s)
    # cv2.imshow("v", img_v)
    # cv2.imshow("color", mask_bar_color.astype(np.uint8) * 255)
    # cv2.imshow("depth", mask_depth.astype(np.uint8) * 255)
    # cv2.imshow("xyz", mask_xyz.astype(np.uint8) * 255)
    # cv2.imshow("bar", mask_bar.astype(np.uint8) * 255)
    # key = cv2.waitKey(0)
    # if key is not None and chr(key) == "q":
    #     exit(1)

    return mask_bar, ObjectInfo(pos, quat, SIZE_BAR)
