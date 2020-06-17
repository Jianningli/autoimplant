import numpy as np
import nrrd
from glob import glob
import open3d as o3d
import math
import stl
from stl import mesh
import numpy

import os
import sys




def find_mins_maxs(obj):
    minx = maxx = miny = maxy = minz = maxz = None
    for p in obj.points:
        if minx is None:
            minx = p[stl.Dimension.X]
            maxx = p[stl.Dimension.X]
            miny = p[stl.Dimension.Y]
            maxy = p[stl.Dimension.Y]
            minz = p[stl.Dimension.Z]
            maxz = p[stl.Dimension.Z]
        else:
            maxx = max(p[stl.Dimension.X], maxx)
            minx = min(p[stl.Dimension.X], minx)
            maxy = max(p[stl.Dimension.Y], maxy)
            miny = min(p[stl.Dimension.Y], miny)
            maxz = max(p[stl.Dimension.Z], maxz)
            minz = min(p[stl.Dimension.Z], minz)
    return minx, maxx, miny, maxy, minz, maxz


if __name__ == '__main__':
    # where the STL files are stored
    base_dir='D:/skull-volume/TU_200/Segmentiert/TU_fertig/stl'
    data_list=glob('{}/*.stl'.format(base_dir))
    x=[]
    y=[]
    z=[]

    for i in range(len(data_list)):
    	main_body = mesh.Mesh.from_file(data_list[i])
    	minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(main_body)
    	x.append(maxx-minx)
    	y.append(maxy-miny)
    	z.append(maxz-minz)


    x=np.array(x)
    y=np.array(y)
    z=np.array(z)

    print('x min',x.min())
    print('x max',x.max())


    print('y min',y.min())
    print('y max',y.max())

    print('z min',z.min())
    print('z max',z.max())



