import cv2  # type: ignore
import numpy as np  # type: ignore

from utils import CameraInfo, PointCloud, filter_point_cloud

BBOX_TABLE = np.array(
    [
        (-0.345, 1.655),  # (back, front) from robot's perspective.
        (-0.5, 0.5),  # (right, left).
        (-0.1, 0.01),  # (bottom, top).
    ]
)
DEBUG = False


def segment_table(
    img_color: np.ndarray,
    img_depth: np.ndarray,
    point_cloud: PointCloud,
    camera: CameraInfo,
) -> np.ndarray:
    # Filter color range.
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
    mask_table = (20 < img_h) & (img_h < 110)
    mask_table &= (0 < img_s) & (img_s < 60)
    mask_table &= (70 < img_v) & (img_v < 250)

    if DEBUG:
        mask_table_color = mask_table

    mask_table &= filter_point_cloud(point_cloud, BBOX_TABLE)

    img_depth_plane = render_table_plane(mask_table, camera)
    img_d = img_depth * mask_table - img_depth_plane
    FILTER_SIZE = (5, 10)
    img_mean = cv2.filter2D(img_d, 1, np.full(FILTER_SIZE, 1 / np.prod(FILTER_SIZE)))
    img_d -= img_mean
    mask_table = (img_d > -5) & mask_table

    # img_d /= 500
    # img_d += 0.5
    # img_d[img_d < 0] = 0
    # img_d[img_d > 1] = 1

    if DEBUG:
        cv2.imshow("Color", img_color)
        cv2.imshow("h", img_h)
        cv2.imshow("s", img_s)
        cv2.imshow("v", img_v)
        cv2.imshow("color", mask_table_color.astype(np.uint8) * 255)
        cv2.imshow("Depth", img_depth / 2000)
        cv2.imshow("d", (img_d * 255).astype(np.uint8))
        cv2.imshow("table", mask_table.astype(np.uint8) * 255)
        key = cv2.waitKey(0)
        if key is not None and chr(key) == "q":
            exit(1)

    return mask_table


def augment_points(points: np.ndarray) -> np.ndarray:
    return np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)


def render_table_plane(mask: np.ndarray, camera: CameraInfo) -> np.ndarray:
    """Renders a table plane depth image.

    (xyz in camera frame)
    u     fx 0  cx   x
    v  =  0  fy cy * y
    1     0  0  1    z

    x = d * (u - cx) / fx
    y = d * (v - cy) / fy
    z = d

        (u - cx) / fx
    r = (v - cy) / fy
              1
    xyz = d r

    table origin: o
    table normal: n
    (d r - o) ' n = 0
    d r' n - o' n = 0
    d = o' n / r' n
    """
    C_TABLE = camera.T_world_to_camera[:3, 3]
    N_TABLE = camera.T_world_to_camera[:3, 2]

    H, W = mask.shape
    F_XY = camera.K.diagonal()[:2]
    C_XY = camera.K[:2, 2]

    vs, us = mask.nonzero()
    uvs = np.stack((us, vs), axis=1)
    rs = augment_points((uvs - C_XY[None, :]) / F_XY[None, :])
    ds = C_TABLE.dot(N_TABLE) / rs.dot(N_TABLE)

    img_depth = np.zeros_like(mask, dtype=np.float32)
    img_depth[mask] = ds * 1000

    return img_depth
