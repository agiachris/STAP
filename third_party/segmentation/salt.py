from typing import Tuple, Optional

import cv2  # type: ignore
import numpy as np  # type: ignore

from utils import ObjectInfo, PointCloud, extract_largest_contour

from rack import SIZE_RACK

SIZE_SALT = np.array([0.05, 0.05, 0.1])
DEBUG = False


def segment_salt(
    img_color: np.ndarray,
    img_depth: np.ndarray,
    point_cloud: PointCloud,
    mask_others: np.ndarray,
) -> Tuple[np.ndarray, Optional[ObjectInfo]]:
    # Filter color range.
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
    mask_salt = (60 < img_h) & (img_h < 110)
    mask_salt &= (20 < img_s) & (img_s < 100)
    mask_salt &= (20 < img_v) & (img_v < 190)
    mask_salt &= ~mask_others

    if DEBUG:
        mask_salt_color = mask_salt

    mask_salt = extract_largest_contour(mask_salt)

    ds = img_depth[mask_salt]
    if len(ds) == 0:
        return mask_salt, None
    d_min = ds.min() - 40
    d_max = ds.max()  # + 20
    mask_depth = (d_min <= img_depth) & (img_depth <= d_max)

    idx_points = point_cloud.flatten_indices(mask_salt)
    xyzs = point_cloud.points[idx_points]
    xyz_min = xyzs.min(axis=0) - np.array([0.0, 0.0, 0.00])
    xyz_max = xyzs.max(axis=0) + np.array([0.0, 0.0, 0.07])
    # xyz_min[0] = max(xyz_min[0], xyz_max[0] - SIZE_SALT[0])
    # xyz_min[2] = xyz_max[2] - SIZE_SALT[2] - 0.01
    idxx_min = (xyz_min[None, :] <= point_cloud.points).all(axis=1)
    idxx_max = (point_cloud.points <= xyz_max[None, :]).all(axis=1)
    mask_xyz = point_cloud.expand_indices(idxx_min & idxx_max)

    mask_salt = mask_depth & mask_xyz & ~mask_others

    mask_salt = extract_largest_contour(mask_salt)

    # Get box pose.
    pos = xyzs.mean(axis=0)
    pos[0] -= 0.01
    if pos[2] > SIZE_RACK[2]:
        pos[2] = SIZE_RACK[2] + 0.5 * SIZE_SALT[2]
    else:
        pos[2] = 0.5 * SIZE_SALT[2]

    if DEBUG:
        cv2.imshow("Color", img_color)
        cv2.imshow("h", img_h)
        cv2.imshow("s", img_s)
        cv2.imshow("v", img_v)
        cv2.imshow("color", mask_salt_color.astype(np.uint8) * 255)
        cv2.imshow("depth", mask_depth.astype(np.uint8) * 255)
        cv2.imshow("xyz", mask_xyz.astype(np.uint8) * 255)
        cv2.imshow("salt", mask_salt.astype(np.uint8) * 255)
        key = cv2.waitKey(0)
        if key is not None and chr(key) == "q":
            exit(1)

    return mask_salt, ObjectInfo(pos=pos, size=SIZE_SALT)
