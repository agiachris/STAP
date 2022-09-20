from typing import Tuple, Optional

import cv2  # type: ignore
import numpy as np  # type: ignore

from utils import ObjectInfo, PointCloud, extract_largest_contour

SIZE_CREAM = np.array([0.09, 0.09, 0.12])
DEBUG = False


def segment_cream(
    img_color: np.ndarray,
    img_depth: np.ndarray,
    point_cloud: PointCloud,
    mask_others: np.ndarray,
) -> Tuple[np.ndarray, Optional[ObjectInfo]]:
    # Filter color range.
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
    mask_cream = (165 < img_h) & (img_h < 181)
    mask_cream &= (100 < img_s) & (img_s < 256)
    mask_cream &= (110 < img_v) & (img_v < 200)
    mask_cream &= ~mask_others

    mask_cream = extract_largest_contour(mask_cream)

    if DEBUG:
        mask_cream_color = mask_cream

    ds = img_depth[mask_cream]
    if len(ds) == 0:
        return mask_cream, None
    d_min = ds.min()  # - 20
    d_max = ds.max() + 20
    mask_depth = (d_min <= img_depth) & (img_depth <= d_max)

    idx_points = point_cloud.flatten_indices(mask_cream)
    xyzs = point_cloud.points[idx_points]
    xyz_min = xyzs.min(axis=0) - np.array([0, 0, 0.11])
    xyz_max = xyzs.max(axis=0) + np.array([0.01, 0, 0.02])
    xyz_min[2] = max(0.01, xyz_min[2])
    xyz_min[0] = max(xyz_min[0], xyz_max[0] - SIZE_CREAM[0])
    idxx_min = (xyz_min[None, :] <= point_cloud.points).all(axis=1)
    idxx_max = (point_cloud.points <= xyz_max[None, :]).all(axis=1)
    mask_xyz = point_cloud.expand_indices(idxx_min & idxx_max)

    mask_cream = mask_depth & mask_xyz

    # Get largest contour.
    mask_cream = extract_largest_contour(mask_cream)

    # Get object pose.
    pos = xyzs.mean(axis=0)
    pos[0] = xyz_max[0] - SIZE_CREAM[0] / 2
    pos[2] = xyz_max[2] - SIZE_CREAM[2] / 2

    if DEBUG:
        cv2.imshow("Color", img_color)
        cv2.imshow("h", img_h)
        cv2.imshow("s", img_s)
        cv2.imshow("v", img_v)
        cv2.imshow("color", mask_cream_color.astype(np.uint8) * 255)
        cv2.imshow("depth", mask_depth.astype(np.uint8) * 255)
        cv2.imshow("xyz", mask_xyz.astype(np.uint8) * 255)
        cv2.imshow("cream", mask_cream.astype(np.uint8) * 255)
        key = cv2.waitKey(0)
        if key is not None and chr(key) == "q":
            exit(1)

    return mask_cream, ObjectInfo(pos=pos, size=SIZE_CREAM)
