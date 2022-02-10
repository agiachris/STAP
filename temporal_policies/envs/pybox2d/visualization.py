import os
from copy import deepcopy
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from .constants import COLORS, TAB_COLORS
from .utils import to_homogenous


def draw_caption(image, caption, color="black", loc="top_left"):
    """Draw text on image.
    args:
        image: RGB image as np.array HxWx3
        caption: str text
    returns:
        image: PIL.Image
    """
    h, w, _ = image.shape
    if loc == "top_left": loc = (10, 0)
    elif loc == "bottom_left": loc = (h - 10, 0)
    elif loc == "center": loc = (h // 2, w // 2)
    else: raise ValueError(f"Location {loc} is not supported")
    image = Image.fromarray(image)
    d = ImageDraw.Draw(image)
    d.text(loc, caption, COLORS[color])
    return image


class PyBox2DVisualizer:

    def __init__(self, env, image=None):
        self._env = env
        self._image = image

    def save(self, path, image=None, clear=False, format="png"):
        if image is None:
            assert self._image is not None, "Must render an image before saving"
            image = self._image
        image = Image.fromarray(image)
        assert os.path.splitext(path)[-1][1:] == format, "Save path must match format"
        image.save(path, format)
        if clear: self.clear()
    
    def clear(self):
        self._image = None

    @staticmethod
    def _get_color(color=None):
        assert isinstance(color, str) or isinstance(color, int)
        if isinstance(color, int):
            return list(TAB_COLORS.values())[color]
        return TAB_COLORS[color]

    @staticmethod
    def _format_list(x):
        for i, _x in enumerate(deepcopy(x)):
            assert isinstance(_x, np.ndarray)
            x[i] = _x.squeeze()

    def render_values_xdim(self, **kwargs):
        """Plot x-component values over rendered image.
        """
        assert isinstance(kwargs["x"], list) and isinstance(kwargs["y"], list)
        self._format_list(kwargs["x"])
        self._format_list(kwargs["y"])
        
        # Normalize value estimates
        for i, y in enumerate(kwargs["y"]):
            kwargs["y"][i] = (y - y.min()) / (y.max() - y.min())
        
        # Project to global x to image coordinates
        for i, x in enumerate(kwargs["x"]):
            x = to_homogenous(np.vstack((x * self._env._r, np.zeros_like(x))))
            kwargs["x"][i] = (self._env._global_to_plot @ x)[0, :]
        if len(kwargs["x"]) == 1 and len(kwargs["x"]) != len(kwargs["y"]):
            kwargs["x"] *= len(kwargs["y"])

        # Other 
        if "colors" not in kwargs: kwargs["colors"] = [self._get_color(i) for i in range(len(kwargs["y"]))]
        else: kwargs["colors"] = [self._get_color(k) for k in kwargs["colors"]]
        if "labels" not in kwargs: kwargs["labels"] = [f"Q(s, a) k={i}" for i in range(len(kwargs["y"]))]
        if "yticks" not in kwargs: kwargs["yticks"] = np.around(np.linspace(0, 1, 11), 1)
        if "xticks" not in kwargs:
            x_min = self._env.observation_space.low[0]
            x_max = self._env.observation_space.high[0]
            kwargs["xticks"] = np.around(np.arange(x_min, x_max, 1), 1)

        for k in ["x", "labels", "colors"]: assert len(kwargs[k]) == len(kwargs["y"])
        image = self._env.render(mode="rgb_array")
        image = self._plot_values_xdim(image, **kwargs)
        self._image = self._env._render_util(image)
        return self._image.copy()

    def render_values_theta(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _plot_values_xdim(image,
                          x, y,
                          labels,
                          colors,
                          xticks,
                          yticks,
                          mode="prod",
                          scale = 0.6,
                          ):
        """Overlay function values over x-coordinate.

        args:
            image: uint8 RGB image -- np.array (h, w)
            x: list of numpy arrays of image space x-axis values
            y: list of numpy arrays of normalized y-axis values
            labels: list of labels
            colors: list of colors
            xticks: xtick labels
            yticks: ytick labels
            mode: mode for superimposing y-axis values
        returns: 
            image: rendered plot over image -- np.array (h, w)
        """
        h, w, _ = image.shape
        fig, ax = plt.subplots()
        ax.imshow(image, extent=[0, w, 0, h])

        # Superimpose value estimates
        if mode:
            y.append(getattr(np, mode)(np.array(y), axis=0))
            x.append(x[-1])
            labels.append(f"{mode.capitalize()} Q(s, a)")
            colors.append("tab:purple")

        # Plot normalized values
        for i in range(len(x)):
            y_scaled = y[i] * scale * h
            ax.plot(
                x[i], y_scaled, 
                label=labels[i],
                color=colors[i],
                linewidth=2,
                linestyle="--"
            )
            ax.fill_between(
                x[i], y_scaled,
                color=colors[i],
                alpha=0.1
            )    
        ax.set_title("Q-function estimates across x-component")
        ax.set_xlabel("x-dim [m]")
        ax.set_ylabel(f"Normalized Q(s, a) [units] (scale {scale:0.1f})")
        ax.set_xticks(np.linspace(0, w, len(xticks)).round().astype(np.int), xticks)
        ax.set_yticks(np.linspace(0, h, len(yticks)).round().astype(np.int), yticks)
        ax.legend(loc="best")
        plt.tight_layout()
        
        # Convert image to array 
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image

    def _plot_values_theta(self, 
                           x, y,
                           labels,
                           colors, 
                           ):
        """
        """
        raise NotImplementedError
