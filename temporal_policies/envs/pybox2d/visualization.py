import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from .constants import COLORS


class PyBox2DPlotter:

    def __init__(self, 
                 env, 
                 global_to_image):
        self._env = env
        self._global_to_image = global_to_image

    def overlay_values_xdim(self, x, y):
        """
        """

    def overlay_values_theta(self, x, y):
        """
        """
        pass


def draw_caption(image, caption, color="black"):
    """Draw text on image.
    args:
        image: RGB image as np.array HxWx3
        caption: str text
    returns:
        image: PIL.Image
    """
    image = Image.fromarray(image)
    d = ImageDraw.Draw(image)
    d.text((10, 0), caption, COLORS[color])
    return image
