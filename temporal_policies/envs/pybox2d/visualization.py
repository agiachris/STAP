# import os
# from copy import deepcopy
# import numpy as np
# import matplotlib
# import pathlib
# from typing import Optional, Union

# matplotlib.use("agg")
# import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from .constants import COLORS  # , TAB_COLORS

# from .utils import to_homogenous


def draw_caption(image, caption, color="black", loc="top_left"):
    """Draw text on image.
    args:
        image: RGB image as np.array HxWx3
        caption: str text
    returns:
        image: PIL.Image
    """
    h, w, _ = image.shape
    if loc == "top_left":
        loc = (10, 0)
    elif loc == "bottom_left":
        loc = (h - 10, 0)
    elif loc == "center":
        loc = (h // 2, w // 2)
    else:
        raise ValueError(f"Location {loc} is not supported")
    image = Image.fromarray(image)
    d = ImageDraw.Draw(image)
    d.text(loc, caption, COLORS[color])
    return image


# class Box2DVisualizer:
#     def __init__(self, env, image=None):
#         self._env = env
#         self._image = image
#
#     @property
#     def env(self):
#         return self._env
#
#     @env.setter
#     def env(self, env):
#         self._env = env
#
#     @property
#     def image(self):
#         return self._image
#
#     @image.setter
#     def image(self, image):
#         self._image = image
#
#     def clear(self, hard=False):
#         self._image = None
#         if hard:
#             self._env = None
#
#     def save(self, path, image=None, clear=True, format="png"):
#         if image is None:
#             assert self._image is not None, "Must render an image before saving"
#             image = self._image
#         image = Image.fromarray(image)
#         assert os.path.splitext(path)[-1][1:] == format, "Save path must match format"
#         image.save(path, format)
#         if clear:
#             self.clear()
#
#     @staticmethod
#     def _get_color(color=None):
#         assert isinstance(color, str) or isinstance(color, int)
#         if isinstance(color, int):
#             return list(TAB_COLORS.values())[color]
#         return TAB_COLORS[color]
#
#     @staticmethod
#     def _format_list(x):
#         for i, _x in enumerate(deepcopy(x)):
#             assert isinstance(_x, np.ndarray)
#             x[i] = _x.squeeze()
#
#     def _format_kwargs_2d(self, kwargs):
#         assert isinstance(kwargs["x"], list) and isinstance(kwargs["y"], list)
#         # Squeeze numpy arrays of values
#         self._format_list(kwargs["x"])
#         self._format_list(kwargs["y"])
#
#         # Normalize value estimates
#         for i, y in enumerate(kwargs["y"]):
#             kwargs["y"][i] = (y - y.min()) / (y.max() - y.min())
#
#         # Replicate x
#         if len(kwargs["x"]) == 1 and len(kwargs["x"]) != len(kwargs["y"]):
#             kwargs["x"] *= len(kwargs["y"])
#
#         # Other
#         if "colors" not in kwargs:
#             kwargs["colors"] = [self._get_color(i) for i in range(len(kwargs["y"]))]
#         else:
#             kwargs["colors"] = [self._get_color(k) for k in kwargs["colors"]]
#         if "labels" not in kwargs:
#             kwargs["labels"] = [f"Q(s, a) k={i}" for i in range(len(kwargs["y"]))]
#         if "yticks" not in kwargs:
#             kwargs["yticks"] = np.around(np.linspace(0, 1, 11), 1)
#         for k in ["x", "labels", "colors"]:
#             assert len(kwargs[k]) == len(kwargs["y"])
#
#     def render_values_xdim(self, **kwargs):
#         """Plot x-component values over rendered image."""
#         self._format_kwargs_2d(kwargs)
#
#         # Project to global x to image coordinates
#         for i, x in enumerate(kwargs["x"]):
#             x = to_homogenous(np.vstack((x * self._env._r, np.zeros_like(x))))
#             kwargs["x"][i] = (self._env._global_to_plot @ x)[0, :]
#
#         if "xticks" not in kwargs:
#             x_min = self._env.observation_space.low[0]
#             x_max = self._env.observation_space.high[0]
#             kwargs["xticks"] = np.around(np.arange(x_min, x_max, 1), 1)
#
#         # image = self._env.render(mode="rgb_array")
#         image = self._env.render(mode="default")
#         image = self._plot_values_xdim(image, **kwargs)
#         caption = self._env._render_caption()
#         self._image = self._env._render_util(image, caption=caption)
#         return self._image.copy()
#
#     @staticmethod
#     def _plot_values_xdim(
#         image, x, y, labels, colors, xticks, yticks, mode="prod", scale=0.6
#     ):
#         """Plot function values over x-coordinate.
#
#         args:
#             image: uint8 RGB image -- np.array (h, w)
#             x: list of numpy arrays of image space x-axis values
#             y: list of numpy arrays of normalized y-axis values
#             labels: list of labels
#             colors: list of colors
#             xticks: xtick labels
#             yticks: ytick labels
#             mode: mode for superimposing y-axis values
#         returns:
#             image: rendered plot over image -- np.array (h, w)
#         """
#         h, w, _ = image.shape
#         fig, ax = plt.subplots()
#         ax.imshow(image, extent=[0, w, 0, h])
#
#         # Superimpose value estimates
#         if mode:
#             y.append(getattr(np, mode)(np.array(y), axis=0))
#             x.append(x[-1])
#             labels.append(f"{mode.capitalize()} Q(s, a)")
#             colors.append("tab:purple")
#
#         # Plot normalized values
#         for i in range(len(x)):
#             y_scaled = y[i] * scale * h
#             ax.plot(
#                 x[i],
#                 y_scaled,
#                 label=labels[i],
#                 color=colors[i],
#                 linewidth=2,
#                 linestyle="--",
#             )
#             ax.fill_between(x[i], y_scaled, color=colors[i], alpha=0.1)
#         ax.set_title("Q-function estimates across x-component")
#         ax.set_xlabel("x-dim [m]")
#         ax.set_ylabel(f"Normalized Q(s, a) [units] (scale {scale:0.1f})")
#         ax.set_xticks(np.linspace(0, w, len(xticks)).round().astype(np.int), xticks)
#         ax.set_yticks(np.linspace(0, h, len(yticks)).round().astype(np.int), yticks)
#         ax.legend(loc="best")
#         plt.tight_layout()
#
#         # Convert image to array
#         fig.canvas.draw()
#         data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#         image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#         plt.close()
#         return image
#
#     def render_values_theta(self, **kwargs):
#         """Plot theta values alongside rendered image."""
#         self._format_kwargs_2d(kwargs)
#         image = self._env.render()
#         image = self._plot_values_theta(image, **kwargs)
#         self._image = image
#         return self._image.copy()
#
#     @staticmethod
#     def _plot_values_theta(
#         image, x, y, labels, colors, yticks, xticks=None, mode="prod", scale=0.75
#     ):
#         """Plot values over theta alongside image.
#
#         args:
#             image: uint8 RGB image -- np.array (h, w)
#             x: list of numpy arrays of theta values
#             y: list of numpy arrays of normalized y-axis values
#             labels: list of labels
#             colors: list of colors
#             yticks: ytick labels
#             xticks: xtick labels
#             mode: mode for superimposing y-axis values
#         returns:
#             image: rendered plot alongside image -- np.array (h_new, w_new)
#         """
#         fig = plt.figure(figsize=(8, 8))
#         ax = fig.add_subplot(211)
#         ax.imshow(image)
#         ax.set_title("Environment under Q-function estimates")
#         ax.set_axis_off()
#
#         # Superimpose value estimates
#         if mode:
#             y.append(getattr(np, mode)(np.array(y), axis=0))
#             x.append(x[-1])
#             labels.append(f"{mode.capitalize()} Q(s, a)")
#             colors.append("tab:purple")
#
#         # Plot normalized values
#         ax = fig.add_subplot(212)
#         for i in range(len(x)):
#             y_scaled = y[i] * scale
#             ax.plot(
#                 x[i],
#                 y_scaled,
#                 label=labels[i],
#                 color=colors[i],
#                 linewidth=1,
#                 linestyle="-",
#             )
#             ax.fill_between(x[i], y_scaled, color=colors[i], alpha=0.1)
#         ax.set_title("Q-function estimates across theta-component")
#         ax.set_xlabel("theta [rad]")
#         if xticks is not None:
#             ax.set_xticks(x[0], np.around(xticks, 1))
#         ax.set_ylabel(f"Normalized Q(s, a) [units] (scale {scale:0.1f})")
#         ax.set_yticks(yticks)
#         ax.legend(loc="best")
#
#         # Convert image to array
#         fig.canvas.draw()
#         data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#         image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#         plt.close()
#         return image
#
#     @staticmethod
#     def plot_xdim_theta_3d(
#         x,
#         y,
#         z,
#         labels,
#         xticks,
#         yticks,
#         path,
#         mode="prod",
#     ):
#         """Plot values"""
#         # Normalize value estimates
#         assert isinstance(z, list)
#         z_min = np.array(z).min()
#         z_max = np.array(z).max()
#         for i, _z in enumerate(deepcopy(z)):
#             z[i] = (_z - z_min) / (z_max - z_min)
#         z.append(getattr(np, mode)(z, axis=0))
#         labels.append(mode.capitalize())
#
#         # Plot normalized values
#         fig, axes = plt.subplots(1, 3, subplot_kw={"projection": "3d"}, figsize=(16, 5))
#
#         xtick_labels = np.around(np.unique(np.sort(xticks)), 1)
#         ytick_labels = np.around(np.unique(np.sort(yticks)), 1)
#         xticks = np.unique(np.sort(x))
#         yticks = np.unique(np.sort(y))
#         for i in range(len(z)):
#             axes[i].plot_trisurf(x, y, z[i], cmap="plasma", linewidth=0)
#             axes[i].set_title(f"{labels[i]} Q(s, a)")
#             axes[i].set_xlabel("x-dim [m]")
#             axes[i].set_xticks(xticks, xtick_labels, rotation=20)
#             axes[i].set_yticks(yticks, ytick_labels, rotation=-10)
#             axes[i].set_ylabel("theta [rad]")
#             axes[i].set_zlabel("Normalized Q(s, a) [units]")
#         str_title = "3D Visualization of Learned Optimization Landscape"
#         plt.suptitle(str_title, fontweight="bold")
#         plt.savefig(path)
#         plt.close()
#
#
# def plot_toy_demo(
#     episode,
#     visualizer,
#     env,
#     planner,
#     output_path,
#     samples=100,
#     ensure_unbiased=True,
#     plot_2d=True,
#     plot_3d=True,
# ):
#     """Plot 2D-3D value function visualization for toy 2d environment.
#     args:
#         visualizer: Box2DVisualizer instance
#         env: gym environment subclass of Box2DBase
#         planner: planner instance subclass of Box2DPlannerBase
#         output_path: path to save 3D plot
#         samples: number of individual samples
#         ensure_unbiased: if planner._default_critic == "v_fn", this variables is redundant, else
#             if planner._default_critic == "q_fn":
#                 if ensure_unbiased == True, Q(s, a) will be evaluated under a ~ learned policy
#                 if ensure_unbiased == False, Q(s, a) will be evaluated under a ~ planner._default_actor
#                 Notice that V(s) = E[Q(s, a)] = Q(s, a), a ~ policy is only an unbiased estimate
#                 iff policy = learned policy.
#     """
#     use_learned_dynamics = planner._use_learned_dynamics(0)
#     config = planner._get_config(0)
#     curr_env = planner._get_env_cls(0).clone(env, **config)
#
#     # Current V(s) or Q(s, a) over all actions (x, theta)
#     curr_state = curr_env._get_observation()
#     if use_learned_dynamics:
#         curr_state = planner._encode_state(0, curr_state)
#     curr_actions, curr_action_dims = curr_env._interp_actions(samples, [0, 1])
#     curr_returns = planner._critic_interface(0, states=curr_state, actions=curr_actions)
#
#     # Store x, y, xticks, yticks, curr_z
#     x = curr_actions[:, 0].copy()
#     y = curr_actions[:, 1].copy()
#     xticks = curr_action_dims[:, 0].copy()
#     yticks = curr_action_dims[:, 1].copy()
#     curr_z = curr_returns.copy()
#
#     # Simulate forward environments
#     num = None if use_learned_dynamics else curr_actions.shape[0]
#     curr_envs = planner._clone_env(0, curr_env, num=num)
#     next_states, _ = planner._simulate_interface(
#         0, envs=curr_envs, states=curr_state, actions=curr_actions
#     )
#
#     next_envs = planner._load_env(1, curr_envs)
#     if use_learned_dynamics:
#         next_envs = [next_envs] * len(curr_actions)
#
#     # Next V(s) or Q(s, a) over states, actions (x, theta)
#     next_z = np.zeros_like(curr_z)
#     for i, (next_env, next_state) in enumerate(zip(next_envs, next_states)):
#         if not use_learned_dynamics:
#             next_state = next_env._get_observation()
#         if ensure_unbiased:
#             next_action = planner._policy(1, next_state)
#         else:
#             next_action = planner._actor_inferface(1, envs=next_env, states=next_state)
#         next_z[i] = planner._critic_interface(1, states=next_state, actions=next_action)
#
#     if plot_3d:
#         # Plot over xdim and theta
#         visualizer.plot_xdim_theta_3d(
#             x=x.copy(),
#             y=y.copy(),
#             z=deepcopy([curr_z, next_z]),
#             labels=[type(curr_env).__name__, type(next_env).__name__],
#             xticks=xticks,
#             yticks=yticks,
#             path=os.path.join(output_path, f"example_{episode}_3d.png"),
#             mode=planner._mode,
#         )
#
#     if plot_2d:
#         # Parse optimal xdim values
#         unique_x = np.unique(np.sort(x))
#         curr_y_xdim = np.zeros_like(unique_x)
#         next_y_xdim = np.zeros_like(unique_x)
#         for k, v in enumerate(unique_x):
#             x_idx = np.where(x == v)
#             curr_y_xdim[k] = curr_z[x_idx].max()
#             next_y_xdim[k] = next_z[x_idx].max()
#
#         # Render over xdim
#         visualizer.render_values_xdim(
#             x=[np.unique(np.sort(xticks))],
#             y=[curr_y_xdim, next_y_xdim],
#             labels=[type(curr_env).__name__, type(next_env).__name__],
#         )
#         visualizer.save(os.path.join(output_path, f"example_{episode}_2d_xdim.png"))
#
#         # Plot variation across theta at top scoring xdim position
#         prod_z = getattr(np, planner._mode)(np.array([curr_z, next_z]), axis=0)
#         theta_idx = np.where(x == x[prod_z.argmax()])
#         unique_theta = y[theta_idx]
#         theta = np.sort(unique_theta)
#         theta_ticks = yticks[theta_idx][unique_theta.argsort()]
#         curr_y_theta = curr_z[theta_idx][unique_theta.argsort()]
#         next_y_theta = next_z[theta_idx][unique_theta.argsort()]
#         assert np.all(np.unique(unique_theta) == unique_theta)
#
#         # Render over theta
#         planner._simulate_interface(
#             0, envs=env, states=curr_state, actions=curr_actions[prod_z.argmax()]
#         )
#         visualizer.env = env
#         visualizer.render_values_theta(
#             x=[theta],
#             xticks=theta_ticks,
#             y=[curr_y_theta, next_y_theta],
#             labels=[type(curr_env).__name__, type(next_env).__name__],
#         )
#         visualizer.save(os.path.join(output_path, f"example_{episode}_2d_theta.png"))
