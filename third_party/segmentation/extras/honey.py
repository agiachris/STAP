from typing import Tuple, Optional

import cv2  # type: ignore
from ctrlutils import eigen  # type: ignore
import numpy as np  # type: ignore

from utils import ObjectInfo, PointCloud, extract_largest_contour

SIZE_HONEY = np.array([0.05, 0.16, 0.11])


def segment_honey(
    img_color: np.ndarray,
    img_depth: np.ndarray,
    point_cloud: PointCloud,
    mask_others: np.ndarray,
) -> Tuple[np.ndarray, Optional[ObjectInfo]]:
    # Filter color range.
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
    mask_honey = (5 < img_h) & (img_h < 20)
    mask_honey &= (80 < img_s) & (img_s < 240)
    mask_honey &= (60 < img_v) & (img_v < 150)
    mask_honey &= ~mask_others

    # mask_honey_color = mask_honey

    mask_honey = extract_largest_contour(mask_honey)

    ds = img_depth[mask_honey]
    if len(ds) == 0:
        return mask_honey, None
    d_min = ds.min() - 30
    d_max = ds.max()  # + 20
    mask_depth = (d_min <= img_depth) & (img_depth <= d_max)

    idx_points = point_cloud.flatten_indices(mask_honey)
    xyzs = point_cloud.points[idx_points]
    xyz_min = xyzs.min(axis=0) - np.array([0.0, 0.0, -0.0])
    xyz_max = xyzs.max(axis=0) + np.array([0.03, 0.0, 0.04])
    idxx_min = (xyz_min[None, :] <= point_cloud.points).all(axis=1)
    idxx_max = (point_cloud.points <= xyz_max[None, :]).all(axis=1)
    mask_xyz = point_cloud.expand_indices(idxx_min & idxx_max)

    mask_honey = mask_depth & mask_xyz & ~mask_others

    mask_honey = extract_largest_contour(mask_honey)

    # Get box pose.
    pos = xyzs.mean(axis=0)
    pos[2] = xyz_min[2] + SIZE_HONEY[2] / 2

    # cv2.imshow("Color", img_color)
    # cv2.imshow("h", img_h)
    # cv2.imshow("s", img_s)
    # cv2.imshow("v", img_v)
    # cv2.imshow("color", mask_honey_color.astype(np.uint8) * 255)
    # cv2.imshow("depth", mask_depth.astype(np.uint8) * 255)
    # cv2.imshow("xyz", mask_xyz.astype(np.uint8) * 255)
    # cv2.imshow("honey", mask_honey.astype(np.uint8) * 255)
    # key = cv2.waitKey(0)
    # if key is not None and chr(key) == "q":
    #     exit(1)

    return mask_honey, ObjectInfo(pos=pos, size=SIZE_HONEY)
