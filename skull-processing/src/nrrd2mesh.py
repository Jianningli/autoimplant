import open3d
import matplotlib.pylab as plt
import sys
import numpy as np
import nrrd
import os
from skimage import measure, morphology




if __name__ == '__main__':
    volume=nrrd.read('filename.nrrd')
    skin_verts, skin_faces, skin_norm, skin_val = measure.marching_cubes_lewiner(volume,step_size=1)
    skin_verts = skin_verts
    skin_points = skin_verts[:, [1, 0, 2]]
    # create mesh
    skin_mesh = open3d.TriangleMesh()
    skin_mesh.vertices = open3d.Vector3dVector(skin_points)
    skin_mesh.triangles = open3d.Vector3iVector(skin_faces)
    skin_mesh.compute_vertex_normals()

    # write mesh
    open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Error)
    open3d.write_triangle_mesh('filemane' + '.ply', skin_mesh)
    open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Info)
    # plot mesh
    fig = plt.figure()
    ax = Axes3D(fig)
    verts, faces = mcubes.marching_cubes(volume, 10)
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='Spectral', lw=1)
    plt.show()


        