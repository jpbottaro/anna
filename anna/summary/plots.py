"""Utilities to work with plots."""

import matplotlib.pyplot as plt


# Colors for Plots
COLORS = [(31, 119, 180),  (174, 199, 232), (255, 127, 14),  (255, 187, 120),
          (44, 160, 44),   (152, 223, 138), (214, 39, 40),   (255, 152, 150),
          (148, 103, 189), (197, 176, 213), (140, 86, 75),   (196, 156, 148),
          (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
          (188, 189, 34),  (219, 219, 141), (23, 190, 207),  (158, 218, 229)]

for i in range(len(COLORS)):
    r, g, b = COLORS[i]
    COLORS[i] = (r / 255., g / 255., b / 255.)


def subplots(nrows=1, ncols=1, figsize=None, xlabel=None, ylabel=None):
    """
    Creates subplots from matplotlib.

    Args:
        nrows (int):
        ncols (int):
        figsize (tuple[int, int]):
        xlabel (str):
        ylabel (str):

    Returns:
        fig (matplotlib.figure.Figure): the matplotlib figure
        ax (matplotlib.Axes or list[matplotlib.Axes]): axes for the subplot
    """
    plt.rc("font", size=20)
    pre, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    all_axs = [axs] if nrows * ncols == 1 else axs
    for ax in all_axs:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linewidth=.2)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.color_nr = 0

    return pre, axs


def plot(ax, x, y, label=None, color=None):
    """
    Plots the given `x` and `y` into `ax`. Optionally adds a label and a color.

    Args:
        ax (matplotlib.Axis): matplotlib's axis
        x (list[float]): list of values for the `x` axis
        y (list[float]): list of values for the `y` axis
        label (str): name for the line plot
        color (int or None): number for a color, or None to chose automatically
    """
    if color is None:
        color = COLORS[ax.color_nr]
        ax.color_nr += 1
    elif isinstance(color, int):
        color = COLORS[color]

    ax.plot(x, y, color=color, label=label)


def moving_average(ys, window=6):
    """
    Calculates moving average window of the given list of values.

    Args:
        ys (list[float]): list of values
        window (int): size of the window

    Returns:
        values (list[float]): list of values as moving average from `ys`. Same
          size as `ys`.
    """
    values = []
    result = []
    for y in ys:
        values.append(y)
        if len(values) > window:
            values = values[1:]
        result.append(sum(values) / len(values))
    return result
