from typing import Optional, Tuple

import cv2  # type: ignore
from ctrlutils import eigen  # type: ignore
import numpy as np  # type: ignore


class CameraInfo:
    def __init__(self, K: np.ndarray, pos: np.ndarray, quat: eigen.Quaterniond):
        self.K = K
        self.pos = pos
        self.quat = quat
        self.R = quat.matrix()

        self.T_camera_to_world = np.eye(4)
        self.T_camera_to_world[:3, :3] = self.R
        self.T_camera_to_world[:3, 3] = self.pos

        self.T_world_to_camera = np.eye(4)
        self.T_world_to_camera[:3, :3] = self.R.T
        self.T_world_to_camera[:3, 3] = -self.R.T.dot(self.pos)


class ObjectInfo:
    def __init__(
        self,
        pos: Optional[np.ndarray] = None,
        quat: Optional[eigen.Quaterniond] = None,
        size: Optional[np.ndarray] = None,
    ):
        if pos is None:
            pos = np.zeros(3)
        if quat is None:
            quat = eigen.Quaterniond(x=0, y=0, z=0, w=1)
        if size is None:
            size = np.zeros(3)

        self.pos = pos.astype(np.float32)
        self.quat = quat
        self.R = quat.matrix().astype(np.float32)
        self.size = size.astype(np.float32)

    def __repr__(self) -> str:
        s = "ObjectInfo<"
        s += f"pos: {self.pos}, "
        s += f"quat: {self.quat.coeffs}, "
        s += f"size: {self.size}>"
        return s


class PointCloud:
    def __init__(self, img_depth: np.ndarray, camera: CameraInfo):
        d, idx_d = PointCloud._get_depths(img_depth)
        self.uv_coords = PointCloud._get_uv_coords(img_depth.shape[:2], idx_d)
        self.points = PointCloud._compute_points(self.uv_coords, d, camera)

        self._dim_img = img_depth.shape
        self._idx_img = idx_d.reshape(self._dim_img)

    @staticmethod
    def _get_depths(img_depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Flattens the depth image into a list of nonzero depths.

        Returns:
            (list of depths, indices of nonzero depths in the flattened image).
        """
        d = img_depth.flatten()
        idx_d = d > 0
        d = (d[idx_d] / 1000).astype(np.float32)

        return d, idx_d

    @staticmethod
    def _get_uv_coords(
        dim_image: Tuple[int, int], idx_points: np.ndarray
    ) -> np.ndarray:
        """Gets the uv coordinates for the nonzero depths in the depth image."""
        H, W = dim_image

        uu, vv = np.meshgrid(
            np.arange(H, dtype=np.int32), np.arange(W, dtype=np.int32), indexing="ij"
        )
        return np.stack((uu.flatten()[idx_points], vv.flatten()[idx_points]), axis=0)

    @staticmethod
    def _compute_points(
        uv_coords: np.ndarray, d: np.ndarray, camera: CameraInfo
    ) -> np.ndarray:
        """Computes xyz points in the world frame from the uv coordinates and depths."""
        x = d * (uv_coords[1].astype(np.float32) - camera.K[0, 2]) / camera.K[0, 0]
        y = d * ((uv_coords[0]).astype(np.float32) - camera.K[1, 2]) / camera.K[1, 1]
        points = np.stack((x, y, d), axis=1)

        return points.dot(camera.R.T) + camera.pos[None, :]

    def expand_indices(self, idx: np.ndarray) -> np.ndarray:
        idx_img = np.zeros(self._dim_img, dtype=bool)
        idx_img[self._idx_img] = idx
        return idx_img

    def flatten_indices(self, idx_img: np.ndarray) -> np.ndarray:
        idx = idx_img.flatten()[self._idx_img.flatten()]
        return idx

    @property
    def mask_image(self) -> np.ndarray:
        return self._idx_img


def filter_point_cloud(point_cloud: PointCloud, bbox: np.ndarray) -> np.ndarray:
    """Returns an image mask of points inside the given bounding box."""
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = bbox
    idx_behind = point_cloud.points[:, 0] < x_min
    idx_front = point_cloud.points[:, 0] > x_max
    idx_right = point_cloud.points[:, 1] < y_min
    idx_left = point_cloud.points[:, 1] > y_max
    idx_below = point_cloud.points[:, 2] < z_min
    idx_above = point_cloud.points[:, 2] > z_max
    idx_outofrange = (
        idx_behind | idx_front | idx_right | idx_left | idx_below | idx_above
    )

    return point_cloud.expand_indices(~idx_outofrange)


def extract_largest_contour(img_mask: np.ndarray) -> np.ndarray:
    """Extracts the largest contour from the segmentation image."""
    contours, _ = cv2.findContours(
        img_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return img_mask
    areas = [cv2.contourArea(contour) for contour in contours]
    idx_largest = np.argmax(areas)
    img_mask = cv2.drawContours(
        np.zeros_like(img_mask, dtype=np.uint8),
        contours,
        idx_largest,
        1,
        cv2.FILLED,
    ).astype(bool)
    return img_mask


