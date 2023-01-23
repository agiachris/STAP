import os
import tyro

from configs.base_config import *


def main(config: str):

    # Structure Language Task 1.
    inst = "How would you use the hook to pull the red box closer, then pick it up?"

    # Structure Language Task 2.
    inst = "How would you pick the hook up from the rack and place it on the table?"

    # Structure Language Task 3.
    inst = "How would you pick and place the cyan box, then use the hook to push the yellow box under the rack?"

    # Long-horizon Task 2.
    inst = "How would you move the blue box under the rack?"

    # Long-horizon Task 1.
    inst = "How would you move all of the boxes to the rack?"

    # Lifted Task 1.
    inst = "How would you move the boxes to create more space on the table?"

    # Lifted Task 2.
    inst = "How would you put a prime colored box on the rack?"


if __name__ == "__main__":
    tyro.cli(main)
