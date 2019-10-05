import matplotlib.pyplot as _plt


from hmc_tomography.Post import Processing as _Processing
from hmc_tomography.Post import Samples as _Samples


def visualize_2_dimensions(
    samples: _Samples, dim1: int = 0, dim2: int = 1, bins: int = 25, show: bool = True
):
    figure_analysis = _plt.figure(figsize=(14, 8))
    axis_2d_histogram = figure_analysis.add_axes([0.05, 0.1, 0.45 / 2, 0.4])

    axis_1d_histogram_x = figure_analysis.add_axes(
        [0.05, 0.5, 0.45 / 2, 0.4], sharex=axis_2d_histogram
    )
    axis_1d_histogram_y = figure_analysis.add_axes(
        [0.05 + 0.5 * 0.45, 0.1, 0.45 / 2, 0.4], sharey=axis_2d_histogram
    )
    axis_1d_traceplot = figure_analysis.add_axes(
        [0.5, 0.1, 0.45, 0.4], sharey=axis_2d_histogram
    )
    axis_autocorrelation = figure_analysis.add_axes(
        [0.5, 0.5, 0.45, 0.4], sharex=axis_1d_traceplot
    )

    axis_2d_histogram.hist2d(
        samples.raw_samples[dim1, :], samples.raw_samples[dim2, :], bins
    )
    axis_1d_histogram_x.hist(samples.raw_samples[dim1, :], bins)
    axis_1d_histogram_y.hist(
        samples.raw_samples[dim2, :], bins, orientation="horizontal"
    )
    axis_1d_traceplot.plot(samples.raw_samples[dim2, :], "--")
    axis_1d_traceplot.set_xlim([0, samples.raw_samples[dim2, :].size])
    axis_autocorrelation.plot(
        _Processing.autocorrelation(samples.raw_samples[dim1, :]),
        "r",
        label=f"Dimension {dim1}",
    )
    axis_autocorrelation.plot(
        _Processing.autocorrelation(samples.raw_samples[dim2, :]),
        "b",
        label=f"Dimension {dim2}",
    )
    axis_autocorrelation.plot(
        _Processing.crosscorrelation(
            samples.raw_samples[dim1, :], samples.raw_samples[dim2, :]
        ),
        "k",
        alpha=0.1,
        label=f"Cross",
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
