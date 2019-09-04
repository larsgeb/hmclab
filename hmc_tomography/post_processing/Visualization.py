import matplotlib.pyplot as _plt

from hmc_tomography.post_processing import Processing as _Processing


def visualize_2_dimensions(
    samples, dim1: int = 0, dim2: int = 1, bins: int = 25
):
    figure_analysis = _plt.figure(figsize=(16, 8))
    axis_2d_histogram = figure_analysis.add_axes([0.025, 0.52, 0.2, 0.4])

    axis_1d_histogram_x = figure_analysis.add_axes(
        [0.025, 0.08, 0.2, 0.4], sharex=axis_2d_histogram
    )
    axis_1d_histogram_y = figure_analysis.add_axes(
        [0.025 + 0.2 + 0.03, 0.52, 0.2, 0.4], sharey=axis_2d_histogram
    )
    axis_1d_traceplot = figure_analysis.add_axes(
        [
            0.025 + 0.2 + 0.03 + 0.2 + 0.03,
            0.52,
            1 - (0.025 + 0.2 + 0.03 + 0.2 + 0.03) - 0.025,
            0.4,
        ],
        sharey=axis_2d_histogram,
    )
    axis_autocorrelation = figure_analysis.add_axes(
        [
            0.025 + 0.2 + 0.03 + 0.2 + 0.03,
            0.08,
            1 - (0.025 + 0.2 + 0.03 + 0.2 + 0.03) - 0.025,
            0.4,
        ]
    )

    axis_2d_histogram.hist2d(samples[dim1, :], samples[dim2, :], bins)
    axis_1d_histogram_x.hist(samples[dim1, :], bins)
    axis_1d_histogram_y.hist(samples[dim2, :], bins, orientation="horizontal")
    axis_1d_traceplot.plot(samples[dim2, :], "--")
    axis_1d_traceplot.set_xlim([0, samples[dim2, :].size])
    axis_autocorrelation.plot(
        _Processing.autocorrelation(samples[dim1, :]),
        "r",
        label=f"Dimension {dim1}",
    )
    axis_autocorrelation.plot(
        _Processing.autocorrelation(samples[dim2, :]),
        "k",
        label=f"Dimension {dim2}",
    )
    axis_autocorrelation.legend()

    return (
        figure_analysis,
        (
            axis_2d_histogram,
            axis_1d_histogram_x,
            axis_1d_histogram_y,
            None,  # For last space
            axis_1d_traceplot,
            axis_autocorrelation,
        ),
    )
