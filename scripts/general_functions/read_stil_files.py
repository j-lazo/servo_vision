import numpy as np
from stl import mesh
import os
from matplotlib import pyplot
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

dir_mesh = os.getcwd() + '/data/calibration/STL_files/' + 'circular_second_part_ground_truth.STL'
your_mesh = mesh.Mesh.from_file(dir_mesh)
print(np.shape(your_mesh))


# Create a new plot
figure = pyplot.figure()
axes = mplot3d.Axes3D(figure)
print(np.shape(your_mesh.vectors))

for vector in your_mesh.vectors:
    print(vector)

axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

# Auto scale to the mesh size
scale = your_mesh.points.flatten()
axes.auto_scale_xyz(scale, scale, scale)

pyplot.show()