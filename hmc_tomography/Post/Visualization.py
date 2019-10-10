"""A module with assorted visualization functions.
"""
import matplotlib.pyplot as _plt
from typing import List as _List

from hmc_tomography.Post import Processing as _Processing
from hmc_tomography.Post import Samples as _Samples
import matplotlib.gridspec as _gridspec
import numpy as _numpy


def marginal_grid(
    samples: _Samples,
    dimensions_list: _List[int],
    bins: int = 25,
    show: bool = True,
    colormap_2d=_plt.get_cmap("Greys"),
    color_1d="black",
):
    number_of_plots = len(dimensions_list)

    _plt.figure(figsize=(8, 8))
    gs1 = _gridspec.GridSpec(number_of_plots, number_of_plots)
    gs1.update(wspace=0.05, hspace=0.05)  # set the spacing between axes.

    # Get extent of every set
    dim_range = []
    for i_dim in range(number_of_plots):
        min = samples[dimensions_list[i_dim], :].min()
        max = samples[dimensions_list[i_dim], :].max()
        dim_range.append((min, max))

    for i_plot in range(number_of_plots):
        # print(i_plot, i_plot) # grid indices for diagonal
        axis = _plt.subplot(gs1[i_plot + (number_of_plots) * i_plot])

        # Modify axes
        if i_plot != number_of_plots - 1:
            axis.set_xticklabels([])
            axis.tick_params(axis="x", which="both", bottom=False, top=False)
        else:
            axis.set_xlabel(f"dimension {dimensions_list[number_of_plots-1]}")

        axis.set_yticklabels([])
        axis.tick_params(axis="y", which="both", left=False, right=False)
        if i_plot == 0:
            axis.set_ylabel("relative density")

        # Plot histogram on diagonal
        axis.hist(
            samples[dimensions_list[i_plot], :],
            bins=bins,
            density=False,
            range=dim_range[i_plot],
            color=color_1d,
        )

        for j_plot in range(i_plot):
            # print(i_plot, j_plot) # grid indices for lower left
            axis = _plt.subplot(gs1[j_plot + (number_of_plots) * i_plot])

            # Modify axes
            if i_plot != number_of_plots - 1:
                axis.set_xticklabels([])
                axis.tick_params(axis="x", which="both", bottom=False, top=False)
            else:
                axis.set_xlabel(f"dimension {dimensions_list[j_plot]}")

            if j_plot != 0:
                axis.set_yticklabels([])
                axis.tick_params(axis="y", which="both", left=False, right=False)
            else:
                axis.set_ylabel(f"dimension {dimensions_list[i_plot]}")

            # Plot 2d marginals
            axis.hist2d(
                samples[dimensions_list[j_plot], :],
                samples[dimensions_list[i_plot], :],
                bins=bins,
                range=[dim_range[j_plot], dim_range[i_plot]],
                cmap=colormap_2d,
            )

            # print(i_plot, j_plot) # grid indices for lower left
            axis = _plt.subplot(gs1[i_plot + (number_of_plots) * j_plot])

            axis.set_xticklabels([])
            axis.tick_params(axis="x", which="both", bottom=False, top=False)
            axis.set_yticklabels([])
            axis.tick_params(axis="y", which="both", left=False, right=False)

            axis.axis("off")

            correlation = _numpy.corrcoef(
                samples[dimensions_list[j_plot], :], samples[dimensions_list[i_plot], :]
            )[1][0]
            axis.text(
                0.5,
                0.5,
                f"{correlation:.2f}",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=40 * _numpy.abs(correlation),
                transform=axis.transAxes,
            )

    if show:
        _plt.show()


def visualize_2_dimensions(
    samples: _Samples,
    dim1: int = 0,
    dim2: int = 1,
    bins: int = 25,
    show: bool = True,
    colormap_2d=_plt.get_cmap("Greys"),
    color_1d="black",
):
    """Method to jointly investigate 2 dimensions of a sampled posterior.

    Parameters
    ==========
    samples : hmc_tomography.Post.Samples
        Samples object.
    dim1 : int
        First dimension to investigate.
    dim2 : int
        Second dimension to investigate.
    bins : int
        Bins used for 1d and 2d histograms.
    show : bool
        Whether or not to render the output. If false, the plot is only show after
        using ``matplotlib.pyplot.show()``. If true, plot is immediately shown.

    """
    figure_analysis = _plt.figure(figsize=(14, 8))
    axis_2d_histogram = figure_analysis.add_axes([0.07, 0.1, 0.45 / 2, 0.4])

    axis_1d_histogram_x = figure_analysis.add_axes(
        [0.07, 0.5, 0.45 / 2, 0.4], sharex=axis_2d_histogram
    )
    axis_1d_histogram_y = figure_analysis.add_axes(
        [0.07 + 0.5 * 0.45, 0.1, 0.45 / 2, 0.4], sharey=axis_2d_histogram
    )
    axis_1d_traceplot = figure_analysis.add_axes(
        [0.52, 0.1, 0.45, 0.4], sharey=axis_2d_histogram
    )
    axis_autocorrelation = figure_analysis.add_axes(
        [0.52, 0.5, 0.45, 0.4], sharex=axis_1d_traceplot
    )

    axis_2d_histogram.hist2d(samples[dim1, :], samples[dim2, :], bins, cmap=colormap_2d)
    axis_1d_histogram_x.hist(samples[dim1, :], bins, color=color_1d)
    axis_1d_histogram_y.hist(
        samples[dim2, :], bins, orientation="horizontal", color=color_1d
    )
    axis_1d_traceplot.plot(samples[dim2, :], "--", color=color_1d)
    axis_1d_traceplot.set_xlim([0, samples[dim2, :].size])
    axis_autocorrelation.plot(
        _Processing.autocorrelation(samples[dim1, :]), "r", label=f"Dimension {dim1}"
    )
    axis_autocorrelation.plot(
        _Processing.autocorrelation(samples[dim2, :]), "b", label=f"Dimension {dim2}"
    )
    axis_autocorrelation.plot(
        _Processing.crosscorrelation(samples[dim1, :], samples[dim2, :]),
        "k",
        alpha=0.25,
        label=f"Cross",
        color=color_1d,
    )
    axis_autocorrelation.legend()

    axis_1d_histogram_x.set_ylabel("count")
    axis_1d_histogram_y.set_xlabel("count")

    axis_2d_histogram.set_xlabel(f"Dimension {dim1}")
    axis_2d_histogram.set_ylabel(f"Dimension {dim2}")

    axis_1d_traceplot.set_xlabel("sample number")
    axis_autocorrelation.set_xlabel("sample delay")
    axis_autocorrelation.set_ylabel("correlation")
    axis_autocorrelation.xaxis.tick_top()
    axis_autocorrelation.xaxis.set_label_position("top")

    axis_autocorrelation.set_yticks([0, 0.5, 1])

    # Disabling ticks
    axis_1d_histogram_x.tick_params(
        axis="x", which="both", bottom=False, labelbottom=False
    )
    axis_1d_histogram_y.tick_params(axis="y", which="both", left=False, labelleft=False)
    # axis_autocorrelation.tick_params(
    #     axis="x", which="both", bottom=False, labelbottom=False
    # )
    axis_1d_traceplot.tick_params(axis="y", which="both", left=False, labelleft=False)

    if show:
        _plt.show()

    return (
        figure_analysis,
        (
            axis_2d_histogram,
            axis_1d_histogram_x,
            axis_1d_histogram_y,
            None,  # For unused space
            axis_1d_traceplot,
            axis_autocorrelation,
        ),
    )
