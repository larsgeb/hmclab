import numpy as np
import matplotlib.pyplot as plt

# from PIL import Image

import tqdm

# Andrew Valentine (andrew.valentine@anu.edu.au)
# Malcolm Sambridge (malcolm.sambridge@anu.edu.au)
#
# Research School of Earth Sciences
# The Australian National University
#
# May 2018
#
# X-ray tomography, after Tarantola Ch. 5.


def generateExampleDataset(filename):
    noiseLevels = None  # [0.005,0.01,0.015,0.02,0.025]
    m = pngToModel("csiro_logo.png", 1024, 1024, 1, 1)
    srcs = np.array(
        [[0, 0], [0, 0.2], [0.0, 0.4], [0, 0.5], [0, 0.6], [0.0, 0.65], [0.0, 0.7]]
        + [[0, x] for x in np.linspace(0.71, 1.0, 30)]
        + [[x, 0] for x in np.linspace(0.3, 0.6, 30)]
    )
    recs = generateSurfacePoints(40, surface=[False, True, False, True])
    # recs = generateSurfacePoints(50,surface=[False,True,False,False])
    # srcs = generateSurfacePoints(50,surface=[False,False,True,False])
    paths = buildPaths(srcs, recs)
    Isrc = np.random.uniform(10, 1, size=paths.shape[0])
    attns, A = tracer(m, paths)
    Irec = Isrc * np.exp(-attns)
    if noiseLevels is not None:
        noise = np.zeros([paths.shape[0]])
        for i in range(0, paths.shape[0]):
            noise[i] = np.random.choice(noiseLevels)
            Irec[i] += np.random.normal(0, noise[i])
            if Irec[i] <= 0:
                Irec[i] = 1.0e-3

    fp = open(filename, "w")
    fp.write("# Src-x Src-y Src-Int Rec-x Rec-y Rec-Int")
    if noiseLevels is None:
        fp.write("\n")
    else:
        fp.write(" Rec-sig\n")
    for i in range(0, paths.shape[0]):
        fp.write(
            "%2.4f %2.4f %2.4f %2.4f %2.4f %2.4f"
            % (paths[i, 0], paths[i, 1], Isrc[i], paths[i, 2], paths[i, 3], Irec[i])
        )
        if noiseLevels is None:
            fp.write("\n")
        else:
            fp.write(" %2.4f\n" % noise[i])
    fp.close()


def buildPaths(srcs, recs):
    if type(srcs) is type([]):
        srcs = np.array(srcs)
    try:
        nsrcs, nc = srcs.shape
    except:
        raise ValueError("Argument 'srcs' must be a 2-D nummpy array")
    if nc != 2:
        raise ValueError("Argument 'srcs' should have shape (N x 2)")
    if type(recs) is type([]):
        recs = np.array(recs)
    try:
        nrecs, nc = recs.shape
    except:
        raise ValueError("Argument 'recs' must be a 2-D nummpy array")
    if nc != 2:
        raise ValueError("Argument 'recs' should have shape (N x 2)")
    npaths = nsrcs * nrecs
    paths = np.zeros([npaths, 4])
    ip = 0
    for isrc in range(nsrcs):
        for irec in range(nrecs):
            paths[ip, 0:2] = srcs[isrc, :]
            paths[ip, 2:4] = recs[irec, :]
            ip += 1
    return paths


def generateSurfacePoints(
    nPerSide, extent=(0, 1, 0, 1), surface=[True, True, True, True], addCorners=True
):
    out = []
    if surface[0]:
        out += [
            [extent[0], x]
            for x in np.linspace(extent[2], extent[3], nPerSide + 2)[1 : nPerSide + 1]
        ]
    if surface[1]:
        out += [
            [extent[1], x]
            for x in np.linspace(extent[2], extent[3], nPerSide + 2)[1 : nPerSide + 1]
        ]
    if surface[2]:
        out += [
            [x, extent[2]]
            for x in np.linspace(extent[0], extent[1], nPerSide + 2)[1 : nPerSide + 1]
        ]
    if surface[3]:
        out += [
            [x, extent[3]]
            for x in np.linspace(extent[0], extent[1], nPerSide + 2)[1 : nPerSide + 1]
        ]
    if addCorners:
        if surface[0] or surface[2]:
            out += [[extent[0], extent[2]]]
        if surface[0] or surface[3]:
            out += [[extent[0], extent[3]]]
        if surface[1] or surface[2]:
            out += [[extent[1], extent[2]]]
        if surface[1] or surface[3]:
            out += [[extent[1], extent[3]]]
    return np.array(out)


# def pngToModel(pngfile, nx, ny, bg=1.0, sc=1.0):
#     png = Image.open(pngfile)
#     png.load()

#     model = sc * (
#         bg
#         + np.asarray(png.convert("L").resize((nx, ny)).transpose(Image.ROTATE_270))
#         / 255.0
#     )
#     return model


def displayModel(
    model,
    paths=None,
    extent=(0, 1, 0, 1),
    clim=None,
    cmap=None,
    figsize=(6, 6),
    axes=None,
):
    #     plt.figure(figsize=figsize)
    if cmap is None:
        cmap = plt.cm.bone_r

    if axes is None:
        plt.imshow(model.T, origin="lower", extent=extent, cmap=cmap)
        if paths is not None:
            for p in paths:
                plt.plot([p[0], p[2]], [p[1], p[3]], "b")
        if clim is not None:
            plt.clim(clim)
        plt.colorbar()
    else:
        axes.imshow(model.T, origin="lower", extent=extent, cmap=cmap)
        if paths is not None:
            for p in paths:
                axes.plot([p[0], p[2]], [p[1], p[3]], "b")
        if clim is not None:
            plt.clim(clim)
        axes.colorbar()


#     plt.show()


def tracer(model, paths, extent=(0, 1, 0, 1)):
    try:
        nx, ny = model.shape
    except:
        raise ValueError("Argument 'model' must be a 2-D numpy array")
    try:
        xmin, xmax, ymin, ymax = extent
    except:
        raise ValueError(
            "Argument 'extent' must be a tuple,list or 1-D array with 4 elements (xmin,xmax,ymin,ymax)"
        )
    if type(paths) == type([]):
        paths = np.array(paths)
    try:
        npaths, ncomp = paths.shape
    except:
        raise ValueError("Argument 'paths' must be a list or 2-D array")
    if ncomp != 4:
        raise ValueError(
            "Each path must be described by four elements (xstart,ystart,xend,yend)"
        )
    if (
        any(paths[:, 0] < xmin)
        or any(paths[:, 0] > xmax)
        or any(paths[:, 1] < ymin)
        or any(paths[:, 1] > ymax)
        or any(paths[:, 2] < xmin)
        or any(paths[:, 2] > xmax)
        or any(paths[:, 3] < ymin)
        or any(paths[:, 3] > ymax)
    ):
        raise ValueError(
            "All sources and receivers must be within or on boundary of model region"
        )

    xGridBounds = np.linspace(xmin, xmax, nx + 1)
    yGridBounds = np.linspace(ymin, ymax, ny + 1)
    A = np.zeros([npaths, nx * ny])
    attns = np.zeros([npaths])
    # print ""
    t = tqdm.tqdm(desc="Evaluating paths", total=npaths)
    for ip, p in enumerate(paths):
        xs, ys, xr, yr = p
        pathLength = np.sqrt((xr - xs) ** 2 + (yr - ys) ** 2)
        # Compute lambda for intersection with each grid-line
        lamX = (
            np.array([])
            if xr == xs
            else np.array([(d - xs) / (xr - xs) for d in xGridBounds])
        )
        lamY = (
            np.array([])
            if yr == ys
            else np.array([(d - ys) / (yr - ys) for d in yGridBounds])
        )
        # Discard any intersections that would occur outside the region
        lamX = np.extract(np.logical_and(lamX >= xmin, lamX <= xmax), lamX)
        nlamX = len(lamX)
        lamY = np.extract(np.logical_and(lamY >= ymin, lamY <= ymax), lamY)
        lam = np.concatenate((lamX, lamY))
        lamSort = np.argsort(lam, kind="mergesort")
        dx = 1 if xr > xs else -1
        dy = 1 if yr > ys else -1
        # print lam
        try:
            if lam[lamSort[0]] != 0:
                lam = np.concatenate((np.array([0]), lam))
                lamSort = np.concatenate((np.array([0]), lamSort + 1))
                nlamX += 1
        except IndexError:
            lam = np.array([0])
            lamSort = np.array([0])
        if lam[lamSort[-1]] != 1:
            lam = np.concatenate((lam, np.array([1])))
            lamSort = np.concatenate((lamSort, np.array([lam.shape[0] - 1])))
        # print lam,lamSort
        if xs == xmin:
            ix = 0
        elif xs == xmax:
            ix = nx - 1
        else:
            ix = (
                np.searchsorted(xGridBounds, xs, side="right" if xr > xs else "left")
                - 1
            )
        if ys == ymin:
            iy = 0
        elif ys == ymax:
            iy = ny - 1
        else:
            iy = (
                np.searchsorted(yGridBounds, ys, side="right" if yr > ys else "left")
                - 1
            )
        # print ix,iy
        pathSensitivity = np.zeros_like(model)
        ilam0 = 2 if lam[lamSort[1]] == 0 else 1
        for ilam in range(ilam0, len(lam)):
            dl = (lam[lamSort[ilam]] - lam[lamSort[ilam - 1]]) * pathLength
            pathSensitivity[ix, iy] = dl
            attns[ip] += dl * model[ix, iy]
            if lamSort[ilam] >= nlamX:
                iy += dy
            else:
                ix += dx

            # print ix,iy,lam[lamSort[ilam]],(xs+lam[lamSort[ilam]]*(xr-xs),ys+lam[lamSort[ilam]]*(yr-ys))
            if lam[lamSort[ilam]] == 1.0:
                break

        A[ip, :] = pathSensitivity.flatten()
        t.update(1)
    t.close()

    return attns, A

