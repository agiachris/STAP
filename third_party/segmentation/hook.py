from typing import Tuple, Optional

import cv2  # type: ignore
from ctrlutils import eigen  # type: ignore
import numpy as np  # type: ignore

from utils import ObjectInfo, PointCloud, extract_largest_contour

SIZE_HOOK = np.array([0.4, 0.2, 0.04])
DEBUG = False


def segment_hook(
    img_color: np.ndarray,
    img_depth: np.ndarray,
    point_cloud: PointCloud,
    mask_others: np.ndarray,
) -> Tuple[np.ndarray, Optional[ObjectInfo]]:
    # Filter color range.
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
    # mask_hook = (40 < img_h) & (img_h < 110)
    # mask_hook &= (0 < img_s) & (img_s < 30)
    # mask_hook &= (150 < img_v) & (img_v < 250)
    mask_hook = (15 < img_h) & (img_h < 50)
    mask_hook &= (10 < img_s) & (img_s < 230)
    mask_hook &= (5 < img_v) & (img_v < 100)

    if DEBUG:
        mask_hook_color = mask_hook

    mask_hook &= ~mask_others

    mask_hook = extract_largest_contour(mask_hook)

    # Get hook pose.
    idx_points = point_cloud.flatten_indices(mask_hook)
    if len(idx_points) == 0:
        return mask_hook, None
    xyzs = point_cloud.points[idx_points]
    pos = xyzs.mean(axis=0)

    xys_c = xyzs[:, :2] - pos[None, :2]
    # x direction is along first eigenvector.
    u, s, vt = np.linalg.svd(xys_c, full_matrices=False)
    xys_eig = xys_c @ vt.T
    xs_eig = xys_eig[:, 0]
    # Handle side is the side with more distance from the center.
    idxx_handle = xs_eig < 0 if -xs_eig.min() > xs_eig.max() else xs_eig > 0

    idx_handle = np.zeros_like(idx_points)
    idx_handle[idx_points] = idxx_handle
    mask_handle = point_cloud.expand_indices(idx_handle)

    # Find eigenvectors of only handle points.
    xys_handle = xyzs[idxx_handle, :2]
    pos_handle = xys_handle.mean(axis=0)
    xys_handle_c = xys_handle - pos_handle
    u, s, vt = np.linalg.svd(xys_handle_c, full_matrices=False)
    # print(vt)
    # xys_eig = xys_handle_c @ vt.T
    # pos_eig = np.percentile(xys_eig, 10, axis=0) + np.array([0.4, 0.1])

    # pos[:2] = (vt.T @ pos_eig) + pos_handle[:2]
    # pos[:2] = np.percentile(xyzs[:, :2], 10, axis=0) + np.array([0.3, 0.13])
    # pos[2] = max(SIZE_HOOK[2] / 2, pos[2])
    pos[:2] = pos_handle[:2]

    # Assume x is always positive.
    xy_handle = vt[0] if vt[0, 0] > 0 else -vt[0]
    theta = np.arctan2(xy_handle[1], xy_handle[0])
    aa = eigen.AngleAxisd(angle=theta, axis=np.array([0, 0, 1]))
    quat = eigen.Quaterniond(aa)

    xy_head = np.array([-xy_handle[1], xy_handle[0]])
    pos[:2] += 0.06 * xy_handle + 0.1 * xy_head
    pos[0] -= 0.02
    # pos[0] -= 0.01
    # pos[0] += 0.02  # needs to think it's further than it actually is
    # pos[0] += 0.01  # needs to think it's further than it actually is
    if DEBUG:
        cv2.imshow("Color", img_color)
        cv2.imshow("h", img_h)
        cv2.imshow("s", img_s)
        cv2.imshow("v", img_v)
        cv2.imshow("hook", mask_hook.astype(np.uint8) * 255)
        cv2.imshow("color", mask_hook_color.astype(np.uint8) * 255)
        cv2.imshow("handle", mask_handle.astype(np.uint8) * 255)
        key = cv2.waitKey(0)
        if key is not None and chr(key) == "q":
            exit(1)

    return mask_hook, ObjectInfo(pos, quat, SIZE_HOOK)
