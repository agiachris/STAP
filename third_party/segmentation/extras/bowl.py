from typing import Tuple, Optional

import cv2  # type: ignore
import numpy as np  # type: ignore

from utils import ObjectInfo, PointCloud

SIZE_BOWL = np.array([0.16, 0.16, 0.08])

# BOWL: 0.08 H, 0.09 Base diameter, 0.16 Top diameter


def segment_bowl(
    img_color: np.ndarray, img_depth: np.ndarray, point_cloud: PointCloud
) -> Tuple[np.ndarray, Optional[ObjectInfo]]:
    # Filter color range.
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
    mask_bowl = (130 < img_h) & (img_h < 150)
    mask_bowl &= (20 < img_s) & (img_s < 170)
    mask_bowl &= (40 < img_v) & (img_v < 180)

    idx_points = point_cloud.flatten_indices(mask_bowl)
    if len(idx_points) == 0:
        return mask_bowl, None
    xyzs = point_cloud.points[idx_points]

    pos = np.mean((xyzs.min(axis=0), xyzs.max(axis=0)), axis=0)

    # cv2.imshow("Color", img_color)
    # cv2.imshow("h", img_h)
    # cv2.imshow("s", img_s)
    # cv2.imshow("v", img_v)
    # # cv2.imshow("color", mask_cracker_color.astype(np.uint8) * 255)
    # cv2.imshow("bowl", mask_bowl.astype(np.uint8) * 255)
    # # cv2.imshow("depth", mask_depth.astype(np.uint8) * 255)
    # # cv2.imshow("xyz", mask_xyz.astype(np.uint8) * 255)
    # key = cv2.waitKey(0)
    # if key is not None and chr(key) == "q":
    #     exit(1)

    return mask_bowl, ObjectInfo(pos=pos, size=SIZE_BOWL)
