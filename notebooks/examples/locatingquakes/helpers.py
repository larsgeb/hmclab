import numpy
import tilemapbase
import obspy.signal
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms



def tilemapbase_project_array(x, y, tilemapbase_project):
    # For some stupid reason the package does not work on arrays.
    assert numpy.array(x).shape == numpy.array(y).shape
    x_p, y_p = numpy.empty_like(x), numpy.empty_like(y)

    for _i, (_x, _y) in enumerate(zip(x, y)):
        _x_p, _y_p = tilemapbase_project(_x, _y)

        x_p[_i] = _x_p
        y_p[_i] = _y_p

    return x_p, y_p


def tilemapbase_create_extent(midpoint, degree_range):
    extent = tilemapbase.Extent.from_lonlat(
        midpoint[0] - degree_range * 2,
        midpoint[0] + degree_range * 2,
        midpoint[1] - degree_range,
        midpoint[1] + degree_range,
    )
    extent = extent.to_aspect(1.0)
    return extent


def rows_contain_nans(df):
    return (
        numpy.sum(numpy.vstack([(numpy.isnan(df[r].values)) for r in df]).T, axis=1) > 0
    )


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor="none", **kwargs):
    """
    See also: https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html


    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Inumpyut data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = numpy.cov(x, y)
    pearson = cov[0, 1] / numpy.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = numpy.sqrt(1 + pearson)
    ell_radius_y = numpy.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = numpy.sqrt(cov[0, 0]) * n_std
    mean_x = numpy.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = numpy.sqrt(cov[1, 1]) * n_std
    mean_y = numpy.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def to_xyz(lon, lat, origin):
    _x, _y = numpy.hsplit(
        numpy.vstack(
            [obspy.signal.util.util_geo_km(*origin, *point) for point in zip(lon, lat)]
        ),
        2,
    )

    _x.shape = lon.shape
    _y.shape = lon.shape

    return _x, _y


def to_lonlat(x, y, origin):
    _lon, _lat = numpy.hsplit(
        numpy.vstack(
            [obspy.signal.util.util_lon_lat(*origin, *point) for point in zip(x, y)]
        ),
        2,
    )

    _lon.shape = x.shape
    _lat.shape = y.shape

    return _lon, _lat


def match_arrays(arr1, arr2):
    return numpy.argmax(arr1[:, None] == arr2, axis=0)
