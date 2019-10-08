import numpy


def generate_rays(extent, n_rays: int = 100, rng_seed: float = 0):

    numpy.random.seed(seed=rng_seed)

    # TODO ... extract extent
    xmin = 0.0
    xmax = 30.0
    ymin = 0.0
    ymax = 18.0

    xlength = xmax - xmin
    ylength = ymax - ymin

    total_length = xlength * 2 + ylength * 2

    paths = []

    for i_ray in range(n_rays):

        path = [None, None, None, None]

        first_point_on = None

        # generate first point
        distance_along_boundary = numpy.random.uniform(0, total_length)
        if distance_along_boundary < xlength:
            # on the first part of the boundary
            first_point_on = 1
            x = distance_along_boundary + xmin
            y = ymin
        elif distance_along_boundary < xlength + ylength:
            # on the second part of the boundary
            first_point_on = 2
            x = xmax
            y = (distance_along_boundary - xlength) + ymin
        elif distance_along_boundary < xlength + ylength + xlength:
            # on the third part of the boundary
            first_point_on = 3
            x = (distance_along_boundary - xlength - ylength) + xmin
            y = ymax
        else:
            # on the fourth part of the boundary
            first_point_on = 4
            x = xmin
            y = (distance_along_boundary - xlength - ylength - xlength) + ymin

        path[0] = x
        path[1] = y

        # generate first point
        no_succes_yet = True
        while no_succes_yet:
            distance_along_boundary = numpy.random.uniform(0, total_length)
            if distance_along_boundary < xlength:
                # on the first part of the boundary
                if first_point_on != 1 and first_point_on != 2 and first_point_on != 4:
                    no_succes_yet = False
                x = distance_along_boundary + xmin
                y = ymin
            elif distance_along_boundary < xlength + ylength:
                # on the second part of the boundary
                if first_point_on != 2 and first_point_on != 1 and first_point_on != 3:
                    no_succes_yet = False
                x = xmax
                y = (distance_along_boundary - xlength) + ymin
            elif distance_along_boundary < xlength + ylength + xlength:
                # on the third part of the boundary
                if first_point_on != 3 and first_point_on != 2 and first_point_on != 4:
                    no_succes_yet = False
                x = (distance_along_boundary - xlength - ylength) + xmin
                y = ymax
            else:
                # on the fourth part of the boundary
                if first_point_on != 4 and first_point_on != 3 and first_point_on != 1:
                    no_succes_yet = False
                x = xmin
                y = (distance_along_boundary - xlength - ylength - xlength) + ymin

        path[2] = x
        path[3] = y
        paths.append(path)

    return paths
