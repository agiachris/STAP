from typing import Tuple, Optional

import cv2  # type: ignore
from ctrlutils import eigen  # type: ignore
import numpy as np  # type: ignore

from utils import ObjectInfo, PointCloud, extract_largest_contour

SIZE_CRACKER = np.array([0.06, 0.16, 0.21])
DEBUG = False


def segment_cracker(
    img_color: np.ndarray, img_depth: np.ndarray, point_cloud: PointCloud
) -> Tuple[np.ndarray, Optional[ObjectInfo]]:
    # Filter color range.
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    img_h, img_s, _ = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
    mask_cracker = (170 < img_h) | (img_h < 20)
    mask_cracker &= 150 < img_s

    if DEBUG:
        mask_cracker_color = mask_cracker

    # idx_points = point_cloud.flatten_indices(mask_cracker)
    # if (idx_points == 0).all():
    #     return mask_cracker
    # xys = (100 * point_cloud.points[:, :2]).astype(int)
    # xy_min = xys[idx_points].min(axis=0)
    # xys -= xy_min[None, :]
    # xs = xys[idx_points, 0]
    # ys = xys[idx_points, 1]
    # dim = (xs.max() + 1, ys.max() + 1)
    # inds = np.ravel_multi_index((xs, ys), dim)
    # idx_hist = np.zeros_like(idx_points)
    # for val, count in zip(*np.unique(inds, return_counts=True)):
    #     if count < 50:
    #         continue
    #     x, y = np.unravel_index(val, dim)
    #     idx_hist[(xys[:, 0] == x) & (xys[:, 1] == y)] = 1
    # mask_cracker = point_cloud.expand_indices(idx_hist)

    # Get largest contour.
    mask_cracker = extract_largest_contour(mask_cracker)

    ds = img_depth[mask_cracker]
    if len(ds) == 0:
        return mask_cracker, None
    d_min = ds.min() - 20
    d_max = ds.max() + 20
    mask_depth = (d_min <= img_depth) & (img_depth <= d_max)

    idx_points = point_cloud.flatten_indices(mask_cracker)
    xyzs = point_cloud.points[idx_points]
    xyz_min = xyzs.min(axis=0)
    xyz_max = xyzs.max(axis=0) + np.array([0.01, 0, 0.02])
    xyz_min[2] = max(xyz_max[2] - 0.24, 0)
    idxx_min = (xyz_min[None, :] <= point_cloud.points).all(axis=1)
    idxx_max = (point_cloud.points <= xyz_max[None, :]).all(axis=1)
    mask_xyz = point_cloud.expand_indices(idxx_min & idxx_max)

    mask_cracker = mask_depth & mask_xyz

    # Get largest contour.
    mask_cracker = extract_largest_contour(mask_cracker)

    # Get box pose.
    pos = xyzs.mean(axis=0)
    pos[2] = max(xyz_min[2] + SIZE_CRACKER[2] / 2, SIZE_CRACKER[2] / 2)

    u, s, vt = np.linalg.svd(xyzs[:, :2] - pos[None, :2], full_matrices=False)
    xy_normal = vt[-1]
    theta = np.arctan2(xy_normal[1], xy_normal[0])
    aa = eigen.AngleAxisd(angle=theta, axis=np.array([0, 0, 1]))
    quat = eigen.Quaterniond(aa)

    pos[0] += 0.02

    if DEBUG:
        cv2.imshow("Color", img_color)
        cv2.imshow("h", img_h)
        cv2.imshow("s", img_s)
        # cv2.imshow("v", img_v)
        cv2.imshow("color", mask_cracker_color.astype(np.uint8) * 255)
        cv2.imshow("cracker", mask_cracker.astype(np.uint8) * 255)
        cv2.imshow("depth", mask_depth.astype(np.uint8) * 255)
        cv2.imshow("xyz", mask_xyz.astype(np.uint8) * 255)
        key = cv2.waitKey(0)
        if key is not None and chr(key) == "q":
            exit(1)

    return mask_cracker, ObjectInfo(pos, quat, SIZE_CRACKER)
