# Copyright 2022 Diabwellness.ai, Inc.
# All rights reserved

"""Utility function for plotting."""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from pandas.plotting import parallel_coordinates
import seaborn as sns

from bokeh import core, io, palettes, models
from bokeh.plotting import output_file, figure, show
from bokeh.models import BoxAnnotation, LinearAxis, Range1d
from bokeh.io import show, output_notebook


palette = sns.color_palette("bright", 10)


def display_factorial_planes(
    X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None
):
    """Display a scatter plot on a factorial plane, one for each factorial plane.

    Library of Functions for the OpenClassrooms Multivariate Exploratory Data Analysis Course
    https://github.com/OpenClassrooms-Student-Center/Multivariate-Exploratory-Analysis/blob/master/functions.py

    """
    # For each factorial plane
    for d1, d2 in axis_ranks:
        if d2 < n_comp:

            # Initialise the matplotlib figure
            fig = plt.figure(figsize=(7, 6))

            # Display the points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(
                        X_projected[selected, d1],
                        X_projected[selected, d2],
                        alpha=alpha,
                        label=value,
                    )
                plt.legend()

            # Display the labels on the points
            if labels is not None:
                for i, (x, y) in enumerate(X_projected[:, [d1, d2]]):
                    plt.text(x, y, labels[i], fontsize="14", ha="center", va="center")

            # Define the limits of the chart
            boundary = np.max(np.abs(X_projected[:, [d1, d2]])) * 1.1
            plt.xlim([-boundary, boundary])
            plt.ylim([-boundary, boundary])

            # Display grid lines
            plt.plot([-100, 100], [0, 0], color="grey", ls="--")
            plt.plot([0, 0], [-100, 100], color="grey", ls="--")

            # Label the axes, with the percentage of variance explained
            plt.xlabel(
                "PC{} ({}%)".format(
                    d1 + 1, round(100 * pca.explained_variance_ratio_[d1], 1)
                )
            )
            plt.ylabel(
                "PC{} ({}%)".format(
                    d2 + 1, round(100 * pca.explained_variance_ratio_[d2], 1)
                )
            )

            plt.title("Projection of points (on PC{} and PC{})".format(d1 + 1, d2 + 1))
            # plt.show(block=False)


def display_parallel_coordinates_centroids(df, num_clusters):
    """Display a parallel coordinates plot for the centroids in df"""

    # Create the plot
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle("Parallel Coordinates plot for the Centroids")
    fig.subplots_adjust(top=0.9, wspace=0)

    # Draw the chart
    parallel_coordinates(df, "cluster", color=palette)

    # Stagger the axes
    ax = plt.gca()


def create_metrics_plot(
    input_df: pd.DataFrame,
    varea_plots: Dict[str, List],
    line_plots: Dict[str, str],
    extra_y_ranges: Dict[str, Range1d],
    title: str,
    default_fig_kwargs: Optional[Dict] = None,
    default_line_kwargs: Optional[Dict] = None,
    default_varea_kwargs: Optional[Dict] = None,
    stacked_area_plots: bool = False,
) -> figure:
    # pylint: disable-msg=too-many-locals
    """Create stacked area and line plots with the desired settings of color and y_range.

    Stacked area plot: https://docs.bokeh.org/en/latest/docs/gallery/stacked_area.html

    Args:
        input_df: Input dataframe containing the desired measurements to plot.
        varea_plots: The dictionary of area plots mapping from
            corresponding column name to color, y_range and start value.
        line_plots: The dictionary of line plots mapping from
            corresponding column name to color
        extra_y_ranges: additional y ranges to add to the plot layout.
        title: Name of the plot.
        default_fig_kwargs: Figure information to be passed in
            such as height, width and axis labels.
        default_line_kwargs: Line plot information to be passed in
            such as line_width and line_alpha.
        default_varea_kwargs: Area plot information to be passed in such as fill_alpha.
        stacked_area_plots: If True, area plots are stacked over each other.

    Returns:
        The bokeh figure containing the desired line and area plots.
    """
    if not default_fig_kwargs:
        default_fig_kwargs = {
            "height": 200,
            "width": 900,
            "x_axis_label": "Time",
            "y_axis_label": "Value",
            "x_axis_type": "datetime",
        }
    if not default_line_kwargs:
        default_line_kwargs = {"line_width": 1.5, "line_alpha": 0.8}
    if not default_varea_kwargs:
        default_varea_kwargs = {"fill_alpha": 0.8}

    plot = figure(**default_fig_kwargs)
    plot.title = title
    plot.extra_y_ranges = extra_y_ranges

    for y_range in extra_y_ranges:
        plot.add_layout(LinearAxis(y_range_name=y_range), "right")

    if stacked_area_plots:
        names = list(varea_plots.keys())
        colors = list(attribute[0] for attribute in varea_plots.values())
        y_range_names = list(attribute[1] for attribute in varea_plots.values())

        plot.varea_stack(
            stackers=names,
            x="index",
            color=colors,
            source=input_df,
            legend_label=names,
            y_range_name=y_range_names,
            **default_varea_kwargs,
        )
    else:
        for name, attributes in varea_plots.items():
            color: str
            y_range_name: str
            start_val: float
            color, y_range_name, start_val = attributes
            plot.varea(
                input_df.index,
                start_val,
                input_df[name],
                legend_label=name,
                color=color,
                y_range_name=y_range_name,
                **default_varea_kwargs,
            )
            plot.circle(
                input_df.index,
                input_df[name],
                legend_label=name,
                color=color,
                y_range_name=y_range_name,
                **default_varea_kwargs,
            )

    for name, color in line_plots.items():
        plot.line(
            input_df.index,
            input_df[name],
            legend_label=name,
            color=color,
            **default_line_kwargs,
        )
        plot.circle(
            input_df.index,
            input_df[name],
            legend_label=name,
            color=color,
            **default_line_kwargs,
        )

    plot.toolbar.logo = None
    plot.toolbar.active_scroll = plot.select_one(models.WheelZoomTool)

    return plot
