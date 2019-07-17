import sys
import numpy
import matplotlib.pyplot as pyplot

sys.path.append("..")
from hmc_tomography import Targets

target = Targets.Himmelblau()

xv = numpy.arange(-6, 6, 0.1)
yv = numpy.arange(-6, 6, 0.1)

grid_x, grid_y = numpy.meshgrid(xv, yv)

misfit = numpy.empty_like(grid_x)
gradx = numpy.empty_like(grid_x)
grady = numpy.empty_like(grid_x)

for (ix, iy), value in numpy.ndenumerate(grid_x):
    misfit[ix, iy] = target.misfit(
        numpy.array([[grid_x[ix, iy]], [grid_y[ix, iy]]])
    )
    gr = target.gradient(numpy.array([[grid_x[ix, iy]], [grid_y[ix, iy]]]))
    gradx[ix, iy] = gr[0, 0] / numpy.sqrt(gr[0, 0] ** 2 + gr[1, 0] ** 2)
    grady[ix, iy] = gr[1, 0] / numpy.sqrt(gr[0, 0] ** 2 + gr[1, 0] ** 2)

# pyplot.imshow(numpy.log(misfit + 1e-10), vmin=-10)
# pyplot.show()
# pyplot.quiver(xv, yv, gradx, grady)
# pyplot.show()
# pyplot.streamplot(xv, yv, gradx, grady, density=10)
# pyplot.show()
