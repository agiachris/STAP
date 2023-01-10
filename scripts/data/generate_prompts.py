import os
import tyro

from configs.base_config import *


def main(config: str):

    # Structure Language Task 1.
    inst = "How would you use the hook to pull the red box and pick it up?"

    # Structure Language Task 2.
    inst = "How would you pick the hook up from the rack and place it on the table?"

    # Long-horizon Task 1.
    inst = "How would you move all the boxes to the rack?"

    # Long-horizon Task 2.
    inst = "How would you move the blue box under the rack?"

    # Long-horizon Task 3.
    inst = ""

    # Lifted Task 1.
    inst = "How would you move boxes to create the most room on the table?"

    # Lifted Task 2.
    inst = "How would you put a prime colored box on the rack?"

    pass


if __name__ == "__main__":
    tyro.cli(main)
