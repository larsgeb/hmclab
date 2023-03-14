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
import multiprocess as _multiprocess
from functools import partial as _partial
from hmclab.Distributions import _AbstractDistribution
from hmclab.Helpers.CustomExceptions import InvalidCaseError as _InvalidCaseError


_cmap = _matplotlib.colormaps.get_cmap("nipy_spectral")

############################################################################


class LayeredRayTracing2D(_AbstractDistribution):
    description = {
        "Free parameter": "Layer velocities",
        "Control on dimensionality": "Number of layers",
    }
    solved_angles = None
    last_model = None
    traveltimes_observed = None
    verbose = False

    approximate_speed = None
    approximate_origin_time = None

    parallel = True
    processes = 8

    data_sigma = 0.0005

    def __init__(
        self,
        layer_interfaces,
        shot_offset,
        receiver_depths,
        traveltimes_observed=None,
        tolerance=None,
    ) -> None:
        self.layer_interfaces = _check1darray(layer_interfaces)
        self.shot_offset = _check1darray(shot_offset)
        self.receiver_depths = _check1darray(receiver_depths)
        if traveltimes_observed is not None:
            self.traveltimes_observed = _check1darray(traveltimes_observed)
        self.dimensions = len(layer_interfaces)

        self.data_shape = (self.shot_offset.size, self.receiver_depths.size)

        self.distances = (self.receiver_depths**2 + self.shot_offset[0] ** 2) ** 0.5

        if tolerance is None:
            self.tolerance = 0.1 * _numpy.mean(_numpy.diff(self.receiver_depths))
        else:
            self.tolerance = tolerance

        assert (
            self.layer_interfaces.max() > receiver_depths.max()
        ), "Last layers has to be below last receiver."

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
            self.shot_offset[shot_to_plot],
            self.receiver_depths,
            angles,
            velocities,
            keep_upgoing,
            domain_x_axis_margin,
            force_aspect,
            vlims,
        )

    def misfit(
        self, velocities: _numpy.ndarray, verbose=None, force_new_angles=False
    ) -> float:
        velocities = velocities.flatten()
        if verbose is None:
            verbose = self.verbose
        traveltimes_synthetic = self.forward(
            velocities, verbose=verbose, force_new_angles=force_new_angles
        )

        self.last_model = velocities

        return self._misfit(self.traveltimes_observed, traveltimes_synthetic)

    def _misfit(self, tts_obs, tts_syn):
        return _numpy.nansum(
            (tts_syn - tts_obs - _numpy.nanmean(tts_syn - tts_obs)) ** 2
        ) / (self.data_sigma**2)

    def _dmisfitdsyn(self, tts_obs, tts_syn):
        return (tts_syn - tts_obs - _numpy.nanmean(tts_syn - tts_obs)) / (
            self.data_sigma**2
        )

    def distance_per_layer(self, velocities):
        TTS, DTS = _derivative_to_layer_speeds(
            velocities=velocities,
            interfaces=self.layer_interfaces,
            receivers_x=self.shot_offset[0],
            receivers_z=self.receiver_depths,
            angles=self.solved_angles,
            parallel=self.parallel,
            processes=self.processes,
        )
        return TTS, DTS

    def gradient(self, velocities):
        velocities = velocities.flatten()
        if (self.last_model is None) or not (
            _numpy.allclose(self.last_model, velocities)
        ):
            self.misfit(velocities)

        TTS, dTTSdSL = self.distance_per_layer(velocities)

        dXdTT = self._dmisfitdsyn(tts_obs=self.traveltimes_observed, tts_syn=TTS)

        nnan = _numpy.logical_not(_numpy.isnan(dXdTT))

        gradient = dXdTT[nnan] @ dTTSdSL[nnan]

        return (-gradient.flatten() / velocities**2)[:, None]

    def generate(self, repeat=1, rng=...) -> _numpy.ndarray:
        raise NotImplementedError

    def forward(self, velocities, force_new_angles=False, verbose=None):
        if self.solved_angles is None or force_new_angles:
            angles = 100
        else:
            angles = self.solved_angles

        if verbose is None:
            verbose = self.verbose

        angles, traveltimes_synthetic, lt = self.search_angles(
            velocities, angles=angles, verbose=verbose
        )

        self.solved_angles = angles

        return traveltimes_synthetic

    def search_angles(
        self,
        velocities,
        shot_to_solve=0,
        angles=None,
        max_attempts=100,
        randomize_angle_fraction=0.1,
        verbose=False,
    ):
        angles, _, _, _, traveltimes, lt = _search_angles(
            self.layer_interfaces,
            self.shot_offset[shot_to_solve],
            self.receiver_depths,
            self.tolerance,
            velocities,
            angles=angles,
            max_attempts=max_attempts,
            randomize_angle_fraction=randomize_angle_fraction,
            verbose=verbose,
            parallel=self.parallel,
            processes=self.processes,
        )

        return angles, traveltimes, lt

    def create_default(dimensions):
        raise _InvalidCaseError()

    def fit_homogeneous(self):
        if self.traveltimes_observed is None:
            raise AttributeError("No observations provided")

        nnan = _numpy.logical_not(_numpy.isnan(self.traveltimes_observed))

        A = _numpy.array([self.distances, _numpy.ones_like(self.distances)]).T[nnan, :]
        b = self.traveltimes_observed[nnan]

        best_fit = _numpy.linalg.lstsq(A, b, rcond=-1)

        approximate_speed = (1.0 / best_fit[0][0]).item()
        approximate_origin_time = (best_fit[0][1]).item()

        print(
            f"Approximate medium velocity:\t{approximate_speed:.2f}"
            f"m/s\r\nApproximate origin time:\t{approximate_origin_time:.2f} s"
        )

        self.approximate_speed, self.approximate_origin_time = (
            approximate_speed,
            approximate_origin_time,
        )

        return approximate_speed, approximate_origin_time

    def homogeneous_model(self):
        return _numpy.ones((self.dimensions)) * self.approximate_speed

    def plot_data(self):
        if self.approximate_speed is None:
            self.fit_homogeneous()

        # _plt.figure(figsize=(12, 6))
        _plt.subplot(121)
        _plt.scatter(
            self.traveltimes_observed - self.approximate_origin_time,
            self.receiver_depths,
            s=1,
            label="Observed data",
        )
        # _plt.xlim([0, 0.5])
        # _plt.ylim([0, 1600])
        _plt.xlabel("Travel time [s]")
        _plt.ylabel("Channel depth [m]")
        _plt.gca().invert_yaxis()
        _plt.subplot(122)
        _plt.scatter(
            self.traveltimes_observed
            - self.approximate_origin_time
            - self.distances / self.approximate_speed,
            self.receiver_depths,
            s=1,
            label="Observed data",
        )
        # _plt.ylim([0, 1600])
        _plt.xlabel("Travel time deviation from homogeneous [s]")
        _plt.ylabel("Channel depth [m]")
        _plt.gca().invert_yaxis()


def _tracerays(
    interfaces,
    velocities,
    xystart,
    receivers_x,
    receivers_z,
    takeoffangle,
    maxnumiterations=20000,
    keep_upgoing=False,
    trace_layers=False,
):
    assert interfaces.size == velocities.size
    assert xystart[1] >= 0.0
    # first layer starts at 0 by definition
    assert all(interfaces > 0.0)
    assert _numpy.max(interfaces) > _numpy.max(receivers_z)

    interfaces = _numpy.append(0.0, _numpy.asarray(interfaces))
    nlay = interfaces.size - 1

    ids = _numpy.where(interfaces >= xystart[1])  ## >= !!
    ideplay = ids[0][0]  # _numpy.argmin(abs(pt[1]-interfaces[ids]))
    raypar = _numpy.sin(_numpy.deg2rad(takeoffangle)) / velocities[ideplay]
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
        ids = _numpy.where(interfaces >= pt[1])  ## >= !!
        ideplay = ids[0][0]

        if ideplay > nlay - 1:
            if trace_layers:
                return raycoo, None, None, None
            else:
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
            asarg = velocities[ideplay] * raypar

            if abs(asarg) >= 1.0:
                # turning ray
                # get the new angle (Snell's law), velocities of layer above
                if not keep_upgoing:
                    break
                thetarad = _numpy.pi / 2.0 - thetarad + _numpy.pi / 2.0
                layer_depth = interfaces[ideplay - 1]
                layer_thickness = interfaces[ideplay] - interfaces[ideplay - 1]
                layer_velocity = velocities[ideplay - 1]
            else:
                # get the new angle (Snell's law), velocities of layer below
                thetarad = _numpy.arcsin(velocities[ideplay] * raypar)
                layer_depth = interfaces[ideplay + 1]
                layer_thickness = interfaces[ideplay + 1] - interfaces[ideplay]
                layer_velocity = velocities[ideplay]

        elif direction == "up" and (not firstsegment):
            # get the new angle (Snell's law), velocities of layer above
            asarg = velocities[ideplay - 1] * raypar

            if abs(asarg) >= 1.0:
                # turning ray
                thetarad = thetarad - _numpy.pi / 2.0
                layer_depth = interfaces[ideplay]  # interfaces[ideplay+1]
                layer_thickness = (
                    interfaces[ideplay] - interfaces[ideplay - 1]
                )  # interfaces[ideplay+1]-interfaces[ideplay]
                layer_velocity = velocities[ideplay]  # velocities[ideplay+1]
            else:
                ## going up..
                thetarad = (
                    _numpy.pi / 2.0
                    - _numpy.arcsin(velocities[ideplay - 1] * raypar)
                    + _numpy.pi / 2.0
                )
                layer_depth = interfaces[ideplay - 1]
                layer_thickness = interfaces[ideplay] - interfaces[ideplay - 1]
                layer_velocity = velocities[ideplay - 1]

        #############
        if i == 0:
            firstsegment = False  # take off (so angle is fixed)
            if direction == "down":
                layer_depth = interfaces[ideplay]
                layer_velocity = velocities[ideplay]
            elif direction == "up":
                layer_depth = interfaces[ideplay]
                layer_velocity = velocities[ideplay]

        #####################################

        if abs(_numpy.cos(thetarad)) <= 1e-12:
            print(
                " timeray(): take off angle: {}. Horizontal ray.".format(
                    _numpy.rad2deg(thetarad)
                )
            )
            if trace_layers:
                return raycoo, None, None, None
            else:
                return raycoo, None, None
        elif abs(_numpy.sin(thetarad)) <= 1e-12:
            xray = raycoo[i, 0]
        else:
            # get the angular coefficient of ray segment
            m = _numpy.cos(thetarad) / _numpy.sin(thetarad)
            xray = (layer_depth + m * pt[0] - pt[1]) / m

        if xray > receivers_x:
            m = _numpy.cos(thetarad) / _numpy.sin(thetarad)

            xray = receivers_x
            layer_depth = m * xray - m * pt[0] + pt[1]

        ##-----------------------------------
        ## Do ray path calculations
        curdist = _numpy.sqrt((xray - pt[0]) ** 2 + (layer_depth - pt[1]) ** 2)
        if trace_layers:
            distance_per_layer[ideplay] += curdist
        dist += curdist
        tt += curdist / layer_velocity

        ##-----------------------------------
        raycoo = _numpy.r_[raycoo, _numpy.array([[xray, layer_depth]])]

        if xray > receivers_x:
            break

        if (raycoo[-2, 1] > 0.0) and (
            abs(layer_depth - 0.0) <= 1e-6
        ):  # direction=="up" and (abs(layer_depth-0.0)<=1e-6) :
            break
        elif (raycoo[-2, 1] < 0.0) and (abs(layer_depth - 0.0) <= 1e-6):
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
    parallel=False,
    processes=8,
):
    # Create a lookup table for all computed values of take-off angles
    lookup_table = _numpy.empty((0, 3))

    # If no angles are given (i.e. count or specific ones) start with double the amount
    # of receivers
    if angles is None:
        angles = receivers_z.size * 2

    # If an integer is given, that's how many angles the algorithm starts off with.
    if type(angles) == int:
        angles = _numpy.linspace(0, 89.9, angles)
    else:
        # Otherwise, make sure any initial angles are sorted
        angles = _numpy.sort(angles)[::-1]

    improper_ray = []

    if parallel:
        with _multiprocess.Pool(processes) as pool:
            results = pool.map(
                _partial(
                    _tracerays,
                    interfaces,
                    velocities,
                    _numpy.array([0, 0]),
                    receivers_x,
                    receivers_z,
                    maxnumiterations=interfaces.size * 3,
                    keep_upgoing=False,
                ),
                angles,
            )
    else:
        results = []
        for angle in angles:
            results.append(
                _partial(
                    _tracerays,
                    interfaces,
                    velocities,
                    _numpy.array([0, 0]),
                    receivers_x,
                    receivers_z,
                    maxnumiterations=interfaces.size * 3,
                    keep_upgoing=False,
                )(angle)
            )

    # Compute refraction for all take-off angles
    for angle, result in zip(angles, results):
        # If results are not failing...
        if result is not None:
            RAYCO, TT, DIST = result

            # ... then check if the ray made it to the receiver line ...
            if (RAYCO[-1][0]) == receivers_x:
                # Then add result to look-up table
                lookup_table = _numpy.vstack(
                    (lookup_table, _numpy.array([angle, RAYCO[-1][-1], TT]))
                )
            else:
                improper_ray.append(angles)
        else:
            improper_ray.append(angles)

    # Check for every receiver which ray is closest
    distance_to_closest_ray = _numpy.min(
        _numpy.abs(lookup_table[:, 1] - receivers_z[:, None]), axis=1
    )

    # Check which receivers have converged (i.e. found a close enough ray)
    converged = distance_to_closest_ray < tolerance

    # Count converged receivers
    n_converged = converged.sum()

    # Start refining the take-off angles, while keeping track of the number of failures
    failed_refines = 0
    while not _numpy.all(converged) and failed_refines < max_attempts:
        # Check if we found extra rays since last pass, if so, reset failure count
        if n_converged < converged.sum():
            n_converged = converged.sum()
            failed_refines -= 1  # check dit nog even TODO
        else:
            failed_refines += 1

        failed_refines = max(0, failed_refines)

        if verbose:
            print(converged.sum(), failed_refines)

        # Find closest ray above receiver
        closest_ray_above = _numpy.argmin(
            (lookup_table[:, 1] - receivers_z[:, None]) > 0, axis=1
        )

        # Filter out converged receiver
        closest_ray_above = closest_ray_above[converged == False]

        refine, counts = _numpy.unique(closest_ray_above, return_counts=True)

        for refine_here, count in zip(refine, counts):
            a1 = lookup_table[refine_here - 1, 0]
            a2 = lookup_table[refine_here, 0]

            # subinterval_refines = 3
            subinterval_refines = count * (1 + int(1.5**failed_refines)) + 2
            new_angles_linspace = _numpy.linspace(a1, a2, subinterval_refines)[1:-1]

            z1 = lookup_table[refine_here - 1, 1]
            z2 = lookup_table[refine_here, 1]

            z_focus = receivers_z[
                _numpy.logical_and(receivers_z < z1, receivers_z > z2)
            ]

            new_angles_interpolation = a1 + (a2 - a1) * ((z_focus - z2) / (z1 - z2))
            new_angles = _numpy.concatenate(
                [new_angles_linspace.flatten(), new_angles_interpolation.flatten()]
            )

            new_angles = new_angles * (
                1 + randomize_angle_fraction * _numpy.random.randn(*new_angles.shape)
            )

            if parallel:
                with _multiprocess.Pool(processes) as pool:
                    results = pool.map(
                        _partial(
                            _tracerays,
                            interfaces,
                            velocities,
                            _numpy.array([0, 0]),
                            receivers_x,
                            receivers_z,
                            maxnumiterations=interfaces.size * 3,
                            keep_upgoing=False,
                        ),
                        new_angles,
                    )
            else:
                results = []
                for new_angle in new_angles:
                    results.append(
                        _partial(
                            _tracerays,
                            interfaces,
                            velocities,
                            _numpy.array([0, 0]),
                            receivers_x,
                            receivers_z,
                            maxnumiterations=interfaces.size * 3,
                            keep_upgoing=False,
                        )(new_angle)
                    )

            for new_angle, result in zip(new_angles, results):
                if result is not None:
                    RAYCO, TT, DIST = result
                    if (RAYCO[-1][0]) == receivers_x:
                        lookup_table = _numpy.vstack(
                            (lookup_table, _numpy.array([new_angle, RAYCO[-1][-1], TT]))
                        )

        lookup_table = lookup_table[_numpy.argsort(lookup_table[:, 0])]

        distance_to_closest_ray = _numpy.min(
            _numpy.abs(lookup_table[:, 1] - receivers_z[:, None]), axis=1
        )

        converged = distance_to_closest_ray < tolerance

    if verbose:
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
        lookup_table,
    )


def _derivative_to_layer_speeds(
    velocities,
    interfaces,
    receivers_x,
    receivers_z,
    angles,
    parallel=True,
    processes=8,
):
    TTS = _numpy.empty((angles.size))
    DTS = _numpy.empty((angles.size, velocities.size))

    if parallel:
        with _multiprocess.Pool(processes) as pool:
            results = pool.map(
                _partial(
                    _tracerays,
                    interfaces,
                    velocities,
                    _numpy.array([0, 0]),
                    receivers_x,
                    receivers_z,
                    maxnumiterations=interfaces.size * 3,
                    keep_upgoing=False,
                    trace_layers=True,
                ),
                angles,
            )
    else:
        results = []
        for angle in angles:
            results.append(
                _partial(
                    _tracerays,
                    interfaces,
                    velocities,
                    _numpy.array([0, 0]),
                    receivers_x,
                    receivers_z,
                    maxnumiterations=interfaces.size * 3,
                    keep_upgoing=False,
                    trace_layers=True,
                )(angle)
            )

    for iangle, (angle, result) in enumerate(zip(angles, results)):
        RAYCO, TT, _, distance_per_layer = result
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
    parallel=True,
    processes=8,
):
    fig = _plt.figure(figsize=(12, 12))
    ax1 = _plt.gca()
    _plt.title("Ray tracing vertical profile")

    assert interfaces.size == velocities.size
    assert all(interfaces > 0.0)
    assert _numpy.max(interfaces) > _numpy.max(
        receivers_z
    ), "Last layer needs to be below the last receiver."

    if parallel:
        with _multiprocess.Pool(processes) as pool:
            results = pool.map(
                _partial(
                    _tracerays,
                    interfaces,
                    velocities,
                    _numpy.array([0, 0]),
                    receivers_x,
                    receivers_z,
                    maxnumiterations=interfaces.size * 5,
                    keep_upgoing=keep_upgoing,
                ),
                angles,
            )
    else:
        results = map(
            _partial(
                _tracerays,
                interfaces,
                velocities,
                _numpy.array([0, 0]),
                receivers_x,
                receivers_z,
                maxnumiterations=interfaces.size * 5,
                keep_upgoing=keep_upgoing,
            ),
            angles,
        )

    for result, angle in zip(results, angles):
        if result is not None:
            RAYCO, TT, DIST = result

            rgba = _cmap(angle / 90)

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
            norm=_matplotlib.colors.Normalize(vmin=0, vmax=90), cmap=_cmap
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
            "list or _numpy array."
        )
    if type(array) == list:
        array = _numpy.array(array)
    return array.flatten()
