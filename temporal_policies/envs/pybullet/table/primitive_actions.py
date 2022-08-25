from typing import List, Optional, Tuple

import numpy as np


class PrimitiveAction:
    RANGES: List[Tuple[str, float, float]]

    def __init__(self, vector: Optional[np.ndarray] = None):
        if vector is None:
            vector = np.zeros(len(self.RANGES), dtype=np.float32)
        elif vector.shape[-1] != len(self.RANGES):
            vector = vector.reshape(
                (
                    *vector.shape[:-1],
                    vector.shape[-1] // len(self.RANGES),
                    len(self.RANGES),
                )
            )
        self.vector = vector

    @classmethod
    def range(cls) -> np.ndarray:
        return np.array([r[1:3] for r in cls.RANGES], dtype=np.float32).T

    @classmethod
    def random(cls):
        r = cls.range()
        return cls(np.random.uniform(r[0], r[1]).astype(np.float32))


class PickAction(PrimitiveAction):
    RANGES = [
        ("x", -0.2, 0.2),
        ("y", -0.1, 0.1),
        ("z", -0.05, 0.05),
        ("theta", -0.25 * np.pi, 0.75 * np.pi),
    ]

    def __init__(
        self,
        vector: Optional[np.ndarray] = None,
        pos: Optional[np.ndarray] = None,
        theta: Optional[float] = None,
    ):
        super().__init__(vector)
        if pos is not None:
            self.pos = pos
        if theta is not None:
            self.theta = theta  # type: ignore

    @property
    def pos(self) -> np.ndarray:
        return self.vector[..., :3]

    @pos.setter
    def pos(self, pos: np.ndarray) -> None:
        self.vector[..., :3] = pos

    @property
    def theta(self) -> np.ndarray:
        return self.vector[..., 3]

    @theta.setter
    def theta(self, theta: np.ndarray) -> None:
        self.vector[..., 3] = theta

    def __repr__(self) -> str:
        return "Pick {\n" f"    pos: {self.pos},\n" f"    theta: {self.theta},\n" "}"


class PlaceAction(PrimitiveAction):
    RANGES = [
        ("x", -0.2, 0.8),
        ("y", -0.4, 0.4),
        ("z", 0.0, 0.1),
        ("theta", -0.25 * np.pi, 0.75 * np.pi),
    ]

    def __init__(
        self,
        vector: Optional[np.ndarray] = None,
        pos: Optional[np.ndarray] = None,
        theta: Optional[float] = None,
    ):
        super().__init__(vector)
        if pos is not None:
            self.pos = pos
        if theta is not None:
            self.theta = theta  # type: ignore

    @property
    def pos(self) -> np.ndarray:
        return self.vector[..., :3]

    @pos.setter
    def pos(self, pos: np.ndarray) -> None:
        self.vector[..., :3] = pos

    @property
    def theta(self) -> np.ndarray:
        return self.vector[..., 3]

    @theta.setter
    def theta(self, theta: np.ndarray) -> None:
        self.vector[..., 3] = theta

    def __repr__(self) -> str:
        return "Place {\n" f"    pos: {self.pos},\n" f"    theta: {self.theta},\n" "}"


class PullAction(PrimitiveAction):
    RANGES = [
        ("r_reach", -0.2, 0.0),
        ("r_pull", -0.4, -0.1),
        ("y", -0.01, 0.01),
        ("theta", -0.5 * np.pi, 0.5 * np.pi),
    ]

    def __init__(
        self,
        vector: Optional[np.ndarray] = None,
        r_reach: Optional[float] = None,
        r_pull: Optional[float] = None,
        y: Optional[float] = None,
        theta: Optional[float] = None,
    ):
        super().__init__(vector)
        if r_reach is not None:
            self.r_reach = r_reach  # type: ignore
        if r_pull is not None:
            self.r_pull = r_pull  # type: ignore
        if y is not None:
            self.y = y  # type: ignore
        if theta is not None:
            self.theta = theta  # type: ignore

    @property
    def r_reach(self) -> np.ndarray:
        return self.vector[..., 0]

    @r_reach.setter
    def r_reach(self, r_reach: np.ndarray) -> None:
        self.vector[..., 0] = r_reach

    @property
    def r_pull(self) -> np.ndarray:
        return self.vector[..., 1]

    @r_pull.setter
    def r_pull(self, r_pull: np.ndarray) -> None:
        self.vector[..., 1] = r_pull

    @property
    def y(self) -> np.ndarray:
        return self.vector[..., 2]

    @y.setter
    def y(self, y: np.ndarray) -> None:
        self.vector[..., 2] = y

    @property
    def theta(self) -> np.ndarray:
        return self.vector[..., 3]

    @theta.setter
    def theta(self, theta: np.ndarray) -> None:
        self.vector[..., 3] = theta

    def __repr__(self) -> str:
        return (
            "Pull {\n"
            f"    r_reach: {self.r_reach},\n"
            f"    r_pull: {self.r_pull},\n"
            f"    y: {self.y},\n"
            f"    theta: {self.theta},\n"
            "}"
        )
