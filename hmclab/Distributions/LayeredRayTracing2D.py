# ------------------------------------------------------------------------
#
#    PestoSeis, a numerical laboratory to learn about seismology, written
#    in the Python language.
#    Copyright (C) 2022  Andrea Zunino
#
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# ------------------------------------------------------------------------

"""
Calculate rays in a horizontally layered model.

"""

############################################################################
import matplotlib as _matplotlib
import matplotlib.pyplot as _plt
import numpy as _numpy
from mpl_toolkits.axes_grid1.axes_divider import (
    make_axes_locatable as _make_axes_locatable,
)
from hmclab.Distributions import _AbstractDistribution

cmap = _matplotlib.cm.get_cmap("nipy_spectral")

############################################################################


class LayeredRayTracing2D(_AbstractDistribution):

    description = {
        "Free parameter": "Layer velocities",
        "Control on dimensionality": "Number of layers",
    }

    def __init__(
        self,
        layer_interfaces,
        shot_distances,
        receiver_depths,
        tolerance=None,
    ) -> None:

        self.layer_interfaces = _check1darray(layer_interfaces)
        self.shot_distances = _check1darray(shot_distances)
        self.receiver_depths = _check1darray(receiver_depths)
        self.dimensions = len(layer_interfaces)

        self.data_shape = (self.shot_distances.size, self.receiver_depths.size)

        if tolerance is None:
            self.tolerance = 0.1 * _numpy.mean(_numpy.diff(self.receiver_depths))
        else:
            self.tolerance = tolerance

    def plot_rays(
        self,
        angles,
        velocities,
        shot_to_plot=0,
        keep_upgoing=True,
        domain_x_axis_margin=200,
        force_aspect=True,
        vlims=None,
    ):
        return _plot_rays_and_model(
            self.layer_interfaces,
            self.shot_distances[shot_to_plot],
            self.receiver_depths,
            angles,
            velocities,
            keep_upgoing,
            domain_x_axis_margin,
            force_aspect,
            vlims,
        )

    def misfit(self, coordinates: _numpy.ndarray) -> float:
        return super().misfit(coordinates)

    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:
        return super().gradient(coordinates)

    def generate(self, repeat=1, rng=...) -> _numpy.ndarray:
        raise NotImplementedError

    def search_angles(
        self,
        velocities,
        shot_to_solve=0,
        angles=None,
        max_attempts=100,
        randomize_angle_fraction=0.1,
        verbose=False,
        rayparameter_spacing=False,
    ):
        angles, _, _, _, traveltimes = _search_angles(
            self.layer_interfaces,
            self.shot_distances[shot_to_solve],
            self.receiver_depths,
            self.tolerance,
            velocities,
            angles=angles,
            max_attempts=max_attempts,
            randomize_angle_fraction=randomize_angle_fraction,
            verbose=verbose,
            rayparameter_spacing=rayparameter_spacing,
        )

        return angles, traveltimes


def _tracerayhorlay(
    laydep,
    vel,
    xystart,
    takeoffangle,
    receivers_x,
    receivers_z,
    maxnumiterations=20000,
    keep_upgoing=False,
    trace_layers=False,
):
    """
    Trace rays in a horizontally layered model.

    Args:
      laydep (ndarray): input depth of layers
      vel (ndarray): velocity for each layer
      xystart(ndarray): origin coordinates of the ray
      takeoffangle (float): take off angle
      maxnumiterations (int): limit the number of ray segments to calculate, in case
                               the ray never reaches the surface

    Returns:
         (ndarray,float,float): coordinates of the the ray path, traveltime
                                and distance covered

    """

    #
    # v1,theta1
    #
    # --------xpt1------------
    #           \
    # v2,theta2  \
    #             \
    # ------------xpt2--------
    #
    #
    #   |\
    #   | \   theta
    #   |  \
    #   |<->\
    #   |    *
    #

    assert laydep.size == vel.size  # +1
    assert xystart[1] >= 0.0
    assert all(laydep > 0.0)  # first layer starts at 0 by definition

    laydep = _numpy.append(0.0, _numpy.asarray(laydep))
    nlay = laydep.size - 1

    ids = _numpy.where(laydep >= xystart[1])  ## >= !!
    ideplay = ids[0][0]  # _numpy.argmin(abs(pt[1]-laydep[ids]))
    raypar = _numpy.sin(_numpy.deg2rad(takeoffangle)) / vel[ideplay]
    thetarad = _numpy.deg2rad(takeoffangle)

    raycoo = _numpy.zeros((1, 2))
    raycoo[0, :] = _numpy.array([xystart[0], xystart[1]])

    ##=================

    i = -1
    tt = 0.0
    dist = 0.0
    firstsegment = True

    if trace_layers:
        distance_per_layer = _numpy.zeros((nlay))

    while True:
        i += 1  # start from 0

        pt = _numpy.array([raycoo[i, 0], raycoo[i, 1]])

        ## find closest layer below starting point (z)...
        ids = _numpy.where(laydep >= pt[1])  ## >= !!
        ideplay = ids[0][0]

        if ideplay > nlay - 1:
            return raycoo, None, None
        else:
            pass

        ##===============================

        if _numpy.cos(thetarad) >= 0.0:
            direction = "down"
        else:
            direction = "up"
            if not keep_upgoing:
                break

        ##===============================

        if direction == "down" and (not firstsegment):
            ## arcsin domain goes [-1 1], so if
            ##   asarg>=0.0 we have a turning ray
            asarg = vel[ideplay] * raypar

            if abs(asarg) >= 1.0:
                # turning ray
                # get the new angle (Snell's law), vel of layer above
                thetarad = _numpy.pi / 2.0 - thetarad + _numpy.pi / 2.0
                zlay = laydep[ideplay - 1]
                laythick = laydep[ideplay] - laydep[ideplay - 1]
                vellay = vel[ideplay - 1]
            else:
                # get the new angle (Snell's law), vel of layer below
                thetarad = _numpy.arcsin(vel[ideplay] * raypar)
                zlay = laydep[ideplay + 1]
                laythick = laydep[ideplay + 1] - laydep[ideplay]
                vellay = vel[ideplay]

        elif direction == "up" and (not firstsegment):
            # get the new angle (Snell's law), vel of layer above
            asarg = vel[ideplay - 1] * raypar

            if abs(asarg) >= 1.0:
                # turning ray
                thetarad = thetarad - _numpy.pi / 2.0
                zlay = laydep[ideplay]  # laydep[ideplay+1]
                laythick = (
                    laydep[ideplay] - laydep[ideplay - 1]
                )  # laydep[ideplay+1]-laydep[ideplay]
                vellay = vel[ideplay]  # vel[ideplay+1]
            else:
                ## going up..
                thetarad = (
                    _numpy.pi / 2.0
                    - _numpy.arcsin(vel[ideplay - 1] * raypar)
                    + _numpy.pi / 2.0
                )
                zlay = laydep[ideplay - 1]
                laythick = laydep[ideplay] - laydep[ideplay - 1]
                vellay = vel[ideplay - 1]

        #############
        if i == 0:
            firstsegment = False  # take off (so angle is fixed)
            if direction == "down":
                zlay = laydep[ideplay]
                vellay = vel[ideplay]
            elif direction == "up":
                zlay = laydep[ideplay]
                vellay = vel[ideplay]

        #####################################

        if abs(_numpy.cos(thetarad)) <= 1e-12:
            print(
                " timeray(): take off angle: {}. Horizontal ray.".format(
                    _numpy.rad2deg(thetarad)
                )
            )
            if _tracerayhorlay:
                return raycoo, None, None, None
            else:
                return raycoo, None, None
        elif abs(_numpy.sin(thetarad)) <= 1e-12:
            xray = raycoo[i, 0]
        else:
            # get the angular coefficient of ray segment
            m = _numpy.cos(thetarad) / _numpy.sin(thetarad)
            xray = (zlay + m * pt[0] - pt[1]) / m

        if xray > receivers_x:
            m = _numpy.cos(thetarad) / _numpy.sin(thetarad)

            xray = receivers_x
            zlay = m * xray - m * pt[0] + pt[1]

        ##-----------------------------------
        ## Do ray path calculations
        curdist = _numpy.sqrt((xray - pt[0]) ** 2 + (zlay - pt[1]) ** 2)
        if trace_layers:
            distance_per_layer[ideplay] += curdist
        dist += curdist
        tt += curdist / vellay

        ##-----------------------------------
        raycoo = _numpy.r_[raycoo, _numpy.array([[xray, zlay]])]

        if xray > receivers_x:
            break

        if (raycoo[-2, 1] > 0.0) and (
            abs(zlay - 0.0) <= 1e-6
        ):  # direction=="up" and (abs(zlay-0.0)<=1e-6) :
            break
        elif (raycoo[-2, 1] < 0.0) and (abs(zlay - 0.0) <= 1e-6):
            break

        elif i > maxnumiterations:
            break

    #############################
    #### return coordinates of the ray path, total traveltime and total length
    if trace_layers:
        return raycoo, tt, dist, distance_per_layer
    else:
        return raycoo, tt, dist


def _search_angles(
    interfaces,
    receivers_x,
    receivers_z,
    tolerance,
    velocities,
    angles=None,
    max_attempts=100,
    randomize_angle_fraction=0.1,
    verbose=False,
    rayparameter_spacing=False,
):
    # Create a lookup table for all computed values
    lookup_table = _numpy.empty((0, 3))

    # If no angles are given (i.e. count or specific ones) start with double the amount
    # of receivers
    if angles is None:
        angles = receivers_z.size * 2

    # If an integer is given, that's how many angles the algorithm starts off with.
    if type(angles) == int:
        angles = _numpy.linspace(0, 89.9, angles)
        # if rayparameter_spacing:
        #     angle_1, angle_2 = 0, 89.9
        #     p_1, p_2 = (
        #         _numpy.sin(2 * _numpy.pi * angle_1 / 360) / velocities[0],
        #         _numpy.sin(2 * _numpy.pi * angle_2 / 360) / velocities[0],
        #     )
        #     angles = (
        #         _numpy.arcsin(_numpy.linspace(p_1, p_2, 100) * velocities[0])
        #         * 360
        #         / (2 * _numpy.pi)
        #     )
    else:
        # Otherwise, make sure any initial angles are sorted
        angles = _numpy.sort(angles)[::-1]

    for angle in angles:
        ding = _tracerayhorlay(
            interfaces,
            velocities,
            _numpy.array([0, 0]),
            angle,
            receivers_x,
            receivers_z,
            maxnumiterations=interfaces.size * 3,
            keep_upgoing=False,
        )
        if ding is not None:
            RAYCO, TT, DIST = ding
            if (RAYCO[-1][0]) == receivers_x:

                lookup_table = _numpy.vstack(
                    (lookup_table, _numpy.array([angle, RAYCO[-1][-1], TT]))
                )

    distance_to_closest_ray = _numpy.min(
        _numpy.abs(lookup_table[:, 1] - receivers_z[:, None]), axis=1
    )

    converged = distance_to_closest_ray < tolerance

    n_converged = converged.sum()
    failed_refines = 0

    while not _numpy.all(converged) and failed_refines < max_attempts:

        if n_converged < converged.sum():
            n_converged = converged.sum()
            failed_refines = 0
        else:
            failed_refines += 1

        if verbose:
            print(converged.sum(), failed_refines)

        closest_ray_above = _numpy.argmin(
            (lookup_table[:, 1] - receivers_z[:, None]) > 0, axis=1
        )
        # Filter out converged stations
        closest_ray_above = closest_ray_above[converged == False]

        refine, counts = _numpy.unique(closest_ray_above, return_counts=True)

        for refine_here, count in zip(refine, counts):

            a1 = lookup_table[refine_here - 1, 0]
            a2 = lookup_table[refine_here, 0]
            new_angles = _numpy.linspace(
                a1, a2, count * (1 + int(1.5**failed_refines)) + 2
            )[1:-1]

            for newa in new_angles:
                newa = newa * (1 + randomize_angle_fraction * _numpy.random.randn())
                ding = _tracerayhorlay(
                    interfaces,
                    velocities,
                    _numpy.array([0, 0]),
                    newa,
                    receivers_x,
                    receivers_z,
                    maxnumiterations=interfaces.size * 3,
                    keep_upgoing=False,
                )
                if ding is not None:
                    RAYCO, TT, DIST = ding
                    if (RAYCO[-1][0]) == receivers_x:

                        lookup_table = _numpy.vstack(
                            (lookup_table, _numpy.array([newa, RAYCO[-1][-1], TT]))
                        )

        lookup_table = lookup_table[_numpy.argsort(lookup_table[:, 0])]

        distance_to_closest_ray = _numpy.min(
            _numpy.abs(lookup_table[:, 1] - receivers_z[:, None]), axis=1
        )

        converged = distance_to_closest_ray < tolerance

    print(f"Found: {converged.sum()} / {receivers_z.size} receivers.")

    closest_ray = _numpy.argmin(
        _numpy.abs(lookup_table[:, 1] - receivers_z[:, None]), axis=1
    )

    angles = lookup_table[:, 0][closest_ray]
    angles_og = angles.copy()
    angles[converged == False] = _numpy.nan
    traveltimes = lookup_table[:, 2][closest_ray]

    return (
        _numpy.asfarray(angles),
        converged,
        _numpy.asfarray(distance_to_closest_ray),
        _numpy.asfarray(angles_og),
        _numpy.asfarray(traveltimes),
    )


def _derivative_to_layer_speeds(
    velocities,
    interfaces,
    receivers_x,
    receivers_z,
    angles,
):

    TTS = _numpy.empty((angles.size))
    DTS = _numpy.empty((angles.size, velocities.size))

    for iangle, angle in enumerate(angles):
        RAYCO, TT, _, distance_per_layer = _tracerayhorlay(
            interfaces,
            velocities,
            _numpy.array([0, 0]),
            angle,
            receivers_x,
            receivers_z,
            maxnumiterations=interfaces.size * 3,
            keep_upgoing=False,
            trace_layers=True,
        )

        TTS[iangle] = TT
        DTS[iangle, :] = distance_per_layer

    return TTS, DTS


def _plot_rays_and_model(
    interfaces,
    receivers_x,
    receivers_z,
    angles,
    velocities,
    keep_upgoing=False,
    domain_x_axis_margin=200,
    force_aspect=True,
    vlims=None,
):
    fig = _plt.figure(figsize=(12, 12))
    ax1 = _plt.gca()
    _plt.title("Ray tracing vertical profile")

    for angle in angles:
        ding = _tracerayhorlay(
            interfaces,
            velocities,
            _numpy.array([0, 0]),
            angle,
            receivers_x,
            receivers_z,
            maxnumiterations=interfaces.size * 5,
            keep_upgoing=keep_upgoing,
        )
        if ding is not None:
            RAYCO, TT, DIST = ding

            rgba = cmap(angle / 90)

            _plt.plot(RAYCO[:, 0] - receivers_x, -RAYCO[:, 1], alpha=0.5, color=rgba)

    _plt.plot(
        [-domain_x_axis_margin - 100 - receivers_x, domain_x_axis_margin + 100],
        [0, 0],
        "k",
    )
    for interface in interfaces:
        _plt.plot(
            [-domain_x_axis_margin - 100 - receivers_x, domain_x_axis_margin + 100],
            [-interface, -interface],
            "k",
            alpha=0.25,
        )
    _plt.scatter(_numpy.ones_like(receivers_z) * 0, -receivers_z)
    _plt.xlim([-domain_x_axis_margin - receivers_x, domain_x_axis_margin])

    _plt.ylabel("Depth [m]")
    _plt.xlabel("Horitontal distance [m]]")
    if force_aspect:
        _plt.gca().set_aspect("equal")

    cb = _plt.colorbar(
        _matplotlib.cm.ScalarMappable(
            norm=_matplotlib.colors.Normalize(vmin=0, vmax=90), cmap=cmap
        ),
        orientation="horizontal",
        fraction=0.046,
        pad=0.1,
    )

    cb.set_label("Take-off angle")

    ax2_divider = _make_axes_locatable(ax1)
    ax2 = ax2_divider.append_axes("right", size="30%", pad="10%")
    ax2.set_title("Velocity model")

    max_vel = velocities.max()

    _plt.plot([0, max_vel + 200], [0, 0], "k")
    for interface in interfaces:
        _plt.plot([0, max_vel + 200], [-interface, -interface], "k", alpha=0.1)

    _plt.plot(
        _numpy.repeat(velocities, 2),
        _numpy.hstack([0, _numpy.repeat(-interfaces, 2)[:-1]]),
        "r",
    )
    _plt.gca().yaxis.set_ticklabels([])
    _plt.xlabel("Medium velocity [m/s]")
    if vlims is None:
        _plt.xlim([0, max_vel + 200])
    else:
        _plt.xlim(vlims)
    return fig


def _check1darray(array):
    if not (type(array) == list or type(array) == _numpy.ndarray):
        raise AttributeError(
            "Don't know what to do with the given object. Try to pass a one-dimensional"
            "list or NumPy array."
        )
    if type(array) == list:
        array = _numpy.array(array)
    return array.flatten()
