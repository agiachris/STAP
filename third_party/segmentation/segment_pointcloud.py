#!/usr/bin/env python

import argparse
import os
import signal
import time
from typing import Callable, Dict, List, Tuple

import ctrlutils
from ctrlutils import eigen
import cv2
import numpy as np

from hook import segment_hook  # type: ignore
from icecream import segment_icecream  # type: ignore
from milk import segment_milk  # type: ignore
from rack import segment_rack  # type: ignore
from salt import segment_salt  # type: ignore
from yogurt import segment_yogurt  # type: ignore
from table import BBOX_TABLE, segment_table, render_table_plane  # type: ignore
from utils import CameraInfo, ObjectInfo, PointCloud, filter_point_cloud  # type: ignore

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

KEY_INTRINSIC = "rgbd::camera_0::depth::intrinsic"
KEY_CAMERA_POS = "rgbd::camera_0::pos"
KEY_CAMERA_ORI = "rgbd::camera_0::ori"
KEY_COLOR_IMAGE = "rgbd::camera_0::color"
KEY_DEPTH_IMAGE = "rgbd::camera_0::depth"
KEY_DEPTH_TABLE = "segmentation::depth"
KEY_DEPTH_SEGMENTATION = "segmentation::image"
KEY_UNKNOWN_SEGMENTATION = "segmentation::unknown"
KEY_SEGMENTATION_LABELS = "segmentation::labels"
KEY_OBJECT_PREFIX = "segmentation::objects::"
KEY_TASK_OBJECTS = "segmentation::task_objects"

OBJECTS = [
    "",
    "table",
    "rack",
    "cracker",
    "sugar",
    "cream",
    "soup",
    "hook",
    "bowl",
    "container",
    "bar",
    "mustard",
    "macaroni",
    "honey",
    "tartar",
    "fudge",
    "spam",
    "salt",
    "butter",
    "vegetables",
    "tomato",
    "yogurt",
    "icecream",
    "milk",
    "juice",
    "tea",
    "paprika",
]

BBOX_WORKSPACE = np.array(
    [
        (0.1, 1.04),  # (back, front) from robot's perspective.
        (-0.45, 0.45),  # (right, left).
        (-0.05, 0.4),  # (bottom, top).
    ]
)

TASK_OBJECTS = {
    "": OBJECTS[2:],
    "hook_reach/task2": ["rack", "hook", "salt", "yogurt", "milk", "icecream"],  # idx0
    "constrained_packing/task1": ["rack", "salt", "yogurt", "icecream"],
    "constrained_packing/taskK": ["rack", "salt", "yogurt", "milk"],
    "constrained_packing/task2": ["rack", "salt", "yogurt", "milk", "icecream", "hook"],
    # "rearrangement_push/task0": ["rack", "hook", "icecream"],  # idx5
    "rearrangement_push/task0": [
        "rack",
        "hook",
        "icecream",
        "salt",
        "milk",
        "yogurt",
    ],  # idx5
}
TASK_OBJECTS.update({obj: [obj] for obj in OBJECTS})

REDIS_OBJECTS = {
    "rack": "rack",
    "hook": "hook",
    "salt": "blue_box",
    "yogurt": "yellow_box",
    "milk": "red_box",
    "icecream": "cyan_box",
}


def create_signal_handler() -> Callable[[], bool]:
    """Creates a ctrl-c handler.

    Returns:
        Lambda function that returns False after ctrl-c has been processed.
    """
    running = [True]

    def signal_handler(*_):
        running[0] = False

    def is_running():
        return running[0]

    signal.signal(signal.SIGINT, signal_handler)

    return is_running


def decode_quaternion(b_quat: bytes) -> eigen.Quaterniond:
    """Decodes a quaternion string from Redis."""
    q = ctrlutils.redis.decode_matlab(b_quat)
    quat = eigen.Quaterniond(w=q[3], x=q[0], y=q[1], z=q[2])

    return quat


def get_camera_info(redis: ctrlutils.RedisClient) -> CameraInfo:
    """Gets camera intrinsics and extrinsics from Redis."""
    redis_pipe = redis.pipeline()
    redis_pipe.get(KEY_INTRINSIC)
    redis_pipe.get(KEY_CAMERA_POS)
    redis_pipe.get(KEY_CAMERA_ORI)
    b_intrinsic, b_pos, b_quat = redis_pipe.execute()

    K = ctrlutils.redis.decode_matlab(b_intrinsic).astype(np.float32)
    pos = ctrlutils.redis.decode_matlab(b_pos).astype(np.float32)
    quat = decode_quaternion(b_quat)

    return CameraInfo(K, pos, quat)


def display_segmentation(img_depth: np.ndarray, img_segmentation: np.ndarray) -> None:
    COLORS = np.array(
        [
            (31, 119, 180),
            (255, 127, 14),
            (44, 160, 44),
            (214, 39, 40),
            (148, 103, 189),
            (140, 86, 75),
            (227, 119, 194),
            (127, 127, 127),
            (188, 189, 34),
            (23, 190, 207),
            (174, 199, 232),
            (255, 187, 120),
            (152, 223, 138),
            (255, 152, 150),
            (197, 176, 213),
            (196, 156, 148),
            (247, 182, 210),
            (199, 199, 199),
            (219, 219, 141),
            (158, 218, 229),
        ]
    )
    ALPHA = 0.4
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    img = np.zeros((*img_depth.shape[:2], 3), dtype=np.uint8)
    img[:, :, :] = (255 * img_depth / img_depth.max())[:, :, None]
    idx_labels = np.unique(img_segmentation)
    for idx_color, idx_label in enumerate(idx_labels[2:]):
        rgb = COLORS[idx_color % len(COLORS)]
        idx = img_segmentation == idx_label
        img[idx] = ALPHA * rgb[None, None, :] + (1 - ALPHA) * img[idx]
        vs, us = np.nonzero(idx)
        uv_text = (us.min(), vs.max() + 12)
        cv2.putText(img, OBJECTS[idx_label], uv_text, FONT, 0.5, rgb.tolist())

    cv2.imshow("Segmentation", img)


def manual_segmentation(
    img_color: np.ndarray,
    img_depth: np.ndarray,
    camera: CameraInfo,
    task_objects: List[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, ObjectInfo]]:
    point_cloud = PointCloud(img_depth, camera)

    # Select points within workspace.
    mask_table_real = segment_table(img_color, img_depth, point_cloud, camera)
    mask_workspace = filter_point_cloud(point_cloud, BBOX_WORKSPACE)
    mask_bg = mask_table_real | ~mask_workspace
    img_color[mask_bg] = 0

    img_table = render_table_plane(mask_bg, camera)
    mask_table_ideal = mask_table_real
    mask_table_ideal = filter_point_cloud(PointCloud(img_table, camera), BBOX_TABLE)
    img_depth_new = np.array(img_depth)
    img_depth_new[mask_table_ideal] = img_table[mask_table_ideal]

    img_segmentation = np.zeros_like(img_depth, dtype=np.uint8)
    img_segmentation[mask_table_ideal] = OBJECTS.index("table")

    poses = {}

    # Segment known objects.
    mask_others = mask_table_real.copy()
    if "rack" in task_objects:
        mask_rack, rack = segment_rack(img_color, img_depth, point_cloud)
        if rack is not None:
            img_segmentation[mask_rack] = OBJECTS.index("rack")
            poses["rack"] = rack
            mask_others |= mask_rack

    # if "bowl" in task_objects:
    #     mask_bowl, bowl = segment_bowl(img_color, img_depth, point_cloud)
    #     if bowl is not None:
    #         img_segmentation[mask_bowl] = OBJECTS.index("bowl")
    #         poses["bowl"] = bowl
    #         mask_others |= mask_bowl
    #
    # if "cracker" in task_objects:
    #     mask_cracker, cracker = segment_cracker(img_color, img_depth, point_cloud)
    #     if cracker is not None:
    #         img_segmentation[mask_cracker] = OBJECTS.index("cracker")
    #         poses["cracker"] = cracker
    #         mask_others |= mask_cracker
    #
    # if "bar" in task_objects:
    #     mask_bar, bar = segment_bar(img_color, img_depth, point_cloud, mask_others)
    #     if bar is not None:
    #         img_segmentation[mask_bar] = OBJECTS.index("bar")
    #         poses["bar"] = bar
    #         mask_others |= mask_bar
    #
    # if "mustard" in task_objects:
    #     mask_mustard, mustard = segment_mustard(
    #         img_color, img_depth, point_cloud, mask_others
    #     )
    #     if mustard is not None:
    #         img_segmentation[mask_mustard] = OBJECTS.index("mustard")
    #         poses["mustard"] = mustard
    #         mask_others |= mask_mustard
    #
    # if "macaroni" in task_objects:
    #     mask_macaroni, macaroni = segment_macaroni(
    #         img_color, img_depth, point_cloud, mask_others
    #     )
    #     if macaroni is not None:
    #         img_segmentation[mask_macaroni] = OBJECTS.index("macaroni")
    #         poses["macaroni"] = macaroni
    #         mask_others |= mask_macaroni
    #
    # if "honey" in task_objects:
    #     mask_honey, honey = segment_honey(
    #         img_color, img_depth, point_cloud, mask_others
    #     )
    #     if honey is not None:
    #         img_segmentation[mask_honey] = OBJECTS.index("honey")
    #         poses["honey"] = honey
    #         mask_others |= mask_honey
    #
    # if "container" in task_objects:
    #     mask_container, container = segment_container(
    #         img_color, img_depth, point_cloud, mask_others
    #     )
    #     if container is not None:
    #         img_segmentation[mask_container] = OBJECTS.index("container")
    #         poses["container"] = container
    #         mask_others |= mask_container
    #
    # if "tartar" in task_objects:
    #     mask_tartar, tartar = segment_tartar(
    #         img_color, img_depth, point_cloud, mask_others
    #     )
    #     if tartar is not None:
    #         img_segmentation[mask_tartar] = OBJECTS.index("tartar")
    #         poses["tartar"] = tartar
    #         mask_others |= mask_tartar
    #
    # if "fudge" in task_objects:
    #     mask_fudge, fudge = segment_fudge(
    #         img_color, img_depth, point_cloud, mask_others
    #     )
    #     if fudge is not None:
    #         img_segmentation[mask_fudge] = OBJECTS.index("fudge")
    #         poses["fudge"] = fudge
    #         mask_others |= mask_fudge
    #
    # if "spam" in task_objects:
    #     mask_spam, spam = segment_spam(img_color, img_depth, point_cloud, mask_others)
    #     if spam is not None:
    #         img_segmentation[mask_spam] = OBJECTS.index("spam")
    #         poses["spam"] = spam
    #         mask_others |= mask_spam

    if "salt" in task_objects:
        mask_salt, salt = segment_salt(img_color, img_depth, point_cloud, mask_others)
        if salt is not None:
            img_segmentation[mask_salt] = OBJECTS.index("salt")
            poses["salt"] = salt
            mask_others |= mask_salt

    # if "butter" in task_objects:
    #     mask_butter, butter = segment_butter(
    #         img_color, img_depth, point_cloud, mask_others
    #     )
    #     if butter is not None:
    #         img_segmentation[mask_butter] = OBJECTS.index("butter")
    #         poses["butter"] = butter
    #         mask_others |= mask_butter
    #
    # if "vegetables" in task_objects:
    #     mask_vegetables, vegetables = segment_vegetables(
    #         img_color, img_depth, point_cloud, mask_others
    #     )
    #     if vegetables is not None:
    #         img_segmentation[mask_vegetables] = OBJECTS.index("vegetables")
    #         poses["vegetables"] = vegetables
    #         mask_others |= mask_vegetables
    #
    # if "tomato" in task_objects:
    #     mask_tomato, tomato = segment_tomato(
    #         img_color, img_depth, point_cloud, mask_others
    #     )
    #     if tomato is not None:
    #         img_segmentation[mask_tomato] = OBJECTS.index("tomato")
    #         poses["tomato"] = tomato
    #         mask_others |= mask_tomato

    if "yogurt" in task_objects:
        mask_yogurt, yogurt = segment_yogurt(
            img_color, img_depth, point_cloud, mask_others
        )
        if yogurt is not None:
            img_segmentation[mask_yogurt] = OBJECTS.index("yogurt")
            poses["yogurt"] = yogurt
            mask_others |= mask_yogurt

    if "icecream" in task_objects:
        mask_icecream, icecream = segment_icecream(
            img_color, img_depth, point_cloud, mask_others
        )
        if icecream is not None:
            img_segmentation[mask_icecream] = OBJECTS.index("icecream")
            poses["icecream"] = icecream
            mask_others |= mask_icecream

    # if "paprika" in task_objects:
    #     mask_paprika, paprika = segment_paprika(
    #         img_color, img_depth, point_cloud, mask_others
    #     )
    #     if paprika is not None:
    #         img_segmentation[mask_paprika] = OBJECTS.index("paprika")
    #         poses["paprika"] = paprika
    #         mask_others |= mask_paprika

    # if "sugar" in task_objects:
    #     mask_sugar, sugar = segment_sugar(
    #         img_color, img_depth, point_cloud, mask_others
    #     )
    #     if sugar is not None:
    #         img_segmentation[mask_sugar] = OBJECTS.index("sugar")
    #         poses["sugar"] = sugar
    #         mask_others |= mask_sugar
    #
    # if "cream" in task_objects:
    #     mask_cream, cream = segment_cream(
    #         img_color, img_depth, point_cloud, mask_others
    #     )
    #     if cream is not None:
    #         img_segmentation[mask_cream] = OBJECTS.index("cream")
    #         poses["cream"] = cream
    #         mask_others |= mask_cream
    #
    # if "soup" in task_objects:
    #     mask_soup, soup = segment_soup(
    #         img_color,
    #         img_depth,
    #         point_cloud,
    #         mask_others,
    #     )
    #     if soup is not None:
    #         img_segmentation[mask_soup] = OBJECTS.index("soup")
    #         poses["soup"] = soup
    #         mask_others |= mask_soup

    if "milk" in task_objects:
        mask_milk, milk = segment_milk(img_color, img_depth, point_cloud, mask_others)
        if milk is not None:
            img_segmentation[mask_milk] = OBJECTS.index("milk")
            poses["milk"] = milk
            mask_others |= mask_milk

    # if "tea" in task_objects:
    #     mask_tea, tea = segment_tea(img_color, img_depth, point_cloud, mask_others)
    #     if tea is not None:
    #         img_segmentation[mask_tea] = OBJECTS.index("tea")
    #         poses["tea"] = tea
    #         mask_others |= mask_tea
    #
    # if "juice" in task_objects:
    #     mask_juice, juice = segment_juice(
    #         img_color, img_depth, point_cloud, mask_others
    #     )
    #     if juice is not None:
    #         img_segmentation[mask_juice] = OBJECTS.index("juice")
    #         poses["juice"] = juice
    #         mask_others |= mask_juice

    if "hook" in task_objects:
        mask_hook, hook = segment_hook(
            img_color,
            img_depth,
            point_cloud,
            mask_others,
        )
        if hook is not None:
            img_segmentation[mask_hook] = OBJECTS.index("hook")
            poses["hook"] = hook
            mask_others |= mask_hook

    return img_depth_new, img_segmentation, poses


def main(redis_host: str, redis_port: str, redis_pass: str, task: str):
    redis = ctrlutils.RedisClient(host=redis_host, port=redis_port, password=redis_pass)
    redis_pipe = redis.pipeline()

    camera = get_camera_info(redis)

    timer = ctrlutils.Timer(frequency=30)
    is_running = create_signal_handler()

    task_objects = TASK_OBJECTS[task]

    while is_running():
        timer.sleep()

        tic = time.time()

        redis_pipe.get(KEY_DEPTH_IMAGE)
        redis_pipe.get(KEY_COLOR_IMAGE)
        b_img_depth, b_img_color = redis_pipe.execute()
        img_depth = ctrlutils.redis.decode_opencv(b_img_depth)
        img_color = ctrlutils.redis.decode_opencv(b_img_color)

        # cv2.imshow("Color", img_color)
        # print(img_color.shape)
        # cv2.imshow("Depth", img_depth / img_depth.max())
        # print(img_color.shape, img_depth.shape)
        # key = cv2.waitKey(0)
        # if key >= 0 and chr(key) == "q":
        #     break

        img_depth_table, img_segmentation, poses = manual_segmentation(
            img_color,
            img_depth,
            camera,
            task_objects,
        )

        # cv2.imshow("Depth", img_depth_table / img_depth.max())
        display_segmentation(img_depth, img_segmentation)
        key = cv2.waitKey(1)
        if key >= 0 and chr(key) == "q":
            break

        # redis_pipe.set_image(KEY_DEPTH_TABLE, img_depth_table)
        # redis_pipe.set_image(KEY_DEPTH_SEGMENTATION, img_segmentation)
        # redis_pipe.set(KEY_SEGMENTATION_LABELS, json.dumps(OBJECTS[1:]))
        # redis_pipe.set(KEY_TASK_OBJECTS, json.dumps(task_objects))
        for obj, pose in poses.items():
            obj_name = REDIS_OBJECTS[obj]
            redis_pipe.set_matrix(KEY_OBJECT_PREFIX + f"{obj_name}::pos", pose.pos)
            redis_pipe.set_matrix(
                KEY_OBJECT_PREFIX + f"{obj_name}::ori", pose.quat.coeffs
            )
            # redis_pipe.set_matrix(KEY_OBJECT_PREFIX + f"{obj_name}::size", pose.size)
            print(obj, pose)
        redis_pipe.execute()

        # with open(f"{task}.pkl", "wb") as f:
        #     pickle.dump(
        #         {
        #             "color": img_color,
        #             "depth": img_depth,
        #             "segmentation": img_segmentation,
        #             "labels": OBJECTS[1:],
        #         },
        #         f,
        #     )
        #     exit()
        print(time.time() - tic)

    print(f"Average {timer.num_iters / timer.time_elapsed()} Hz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--redis_host", "-rh", default="127.0.0.1", help="Redis hostname."
    )
    parser.add_argument(
        "--redis_port", "-p", default=6379, type=int, help="Redis port."
    )
    parser.add_argument("--redis_pass", "-a", default="", help="Redis password.")
    parser.add_argument(
        "--task",
        "-t",
        default="",
        choices=TASK_OBJECTS.keys(),
        help="Table env task ID.",
    )
    args = parser.parse_args()

    main(**vars(args))
