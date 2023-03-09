from typing import Tuple, Optional

import cv2  # type: ignore
import numpy as np  # type: ignore

from utils import ObjectInfo, PointCloud, extract_largest_contour

from rack import SIZE_RACK

SIZE_ICECREAM = np.array([0.06, 0.06, 0.065])

DEBUG = False


def segment_icecream(
    img_color: np.ndarray,
    img_depth: np.ndarray,
    point_cloud: PointCloud,
    mask_others: np.ndarray,
) -> Tuple[np.ndarray, Optional[ObjectInfo]]:
    # Filter color range.
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
    mask_icecream = (70 < img_h) & (img_h < 110)
    mask_icecream &= (20 < img_s) & (img_s < 190)
    mask_icecream &= (150 < img_v) & (img_v < 255)
    mask_icecream &= ~mask_others

    mask_icecream_color = mask_icecream

    mask_icecream = extract_largest_contour(mask_icecream)

    ds = img_depth[mask_icecream]
    if len(ds) == 0:
        return mask_icecream, None
    d_min = ds.min()  # - 40
    d_max = ds.max()  # + 30
    mask_depth = (d_min <= img_depth) & (img_depth <= d_max)

    idx_points = point_cloud.flatten_indices(mask_icecream)
    xyzs = point_cloud.points[idx_points]
    xyz_min = xyzs.min(axis=0) + np.array([0.0, 0.0, 0.02])
    xyz_max = xyzs.max(axis=0)  # np.array([0.01, 0.0, 0.0])
    # xyz_min[0] = max(xyz_min[0], xyz_max[0] - SIZE_icecream[1])
    # xyz_min[2] = xyz_max[2] - SIZE_icecream[2] - 0.01
    idxx_min = (xyz_min[None, :] <= point_cloud.points).all(axis=1)
    idxx_max = (point_cloud.points <= xyz_max[None, :]).all(axis=1)
    mask_xyz = point_cloud.expand_indices(idxx_min & idxx_max)

    mask_icecream = mask_depth & mask_xyz & ~mask_others

    mask_icecream = extract_largest_contour(mask_icecream)

    # Get box pose.
    pos = xyzs.mean(axis=0)
    # pos[0] = xyz_max[0] - 0.5 * SIZE_ICECREAM[0]
    # pos[0] -= 0.02
    # pos[0] -= 0.02
    # pos[0] -= 0.01
    pos[0] += 0.015
    if pos[2] > SIZE_RACK[2]:
        pos[2] = SIZE_RACK[2] + 0.035
    else:
        pos[2] = 0.035  # Height in pybullet is 0.07

    if DEBUG:
        cv2.imshow("Color", img_color)
        cv2.imshow("h", img_h)
        cv2.imshow("s", img_s)
        cv2.imshow("v", img_v)
        cv2.imshow("color", mask_icecream_color.astype(np.uint8) * 255)
        # cv2.imshow("depth", mask_depth.astype(np.uint8) * 255)
        # cv2.imshow("xyz", mask_xyz.astype(np.uint8) * 255)
        cv2.imshow("icecream", mask_icecream.astype(np.uint8) * 255)
        key = cv2.waitKey(0)
        if key is not None and chr(key) == "q":
            exit(1)

    return mask_icecream, ObjectInfo(pos=pos, size=SIZE_ICECREAM)
