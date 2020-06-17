# Skull-Data-Processing
>Python/Matlab scripts for skull segmentation from head CT, skull noise removal (e.g., the CT table), volume to mesh conversion and defective skull creation.
* **_Python 3.6_**
* **_MATLAB R2018b_**
* **_WIN10_**

### skull segmentation from head CT
```
dependency: pynrrd (https://pypi.org/project/pynrrd/), pydicom(https://pydicom.github.io/pydicom/). 
installation: pip install pynrrd, pip install -U pydicom
usage:
1.change the directory \
2.run the code: python segmentation.py
Need to specify the threshold for the skull segmentation, usually 100--max HU value is recommended. 
```
### noise removal
```
dependency: 3D connected component analysis (https://pypi.org/project/connected-components-3d/).
installation: pip install connected-components-3d
usage:
1. change  **_data_dir_**  and  **_save_dir_** 
2. run the code: python denoising.py
```

### artificial defect injection
```
dependency: PyMRT(https://pypi.org/project/pymrt/).
installation: pip install pymrt
usage:
1.specify  pair_list, defected_dir, implant_dir
2.specify the size of defect to be injected into the skull 
3.run in the code: python defectinject.py
```

### create mesh model from skull volume (.nrrd files)
```
dependency: 
Open3D(http://www.open3d.org/), scikit-image(https://scikit-image.org/), PyMCubes(https://github.com/pmneila/PyMCubes).
installation: pip install open3d, pip install scikit-image, pip install --upgrade PyMCubes
usage: python nrrd2mesh.py
```

### voxelization : create voxel grid from mesh (matlab/python)
```
dependency: 
Polygon2Voxel(https://www.mathworks.com/matlabcentral/fileexchange/24086-polygon2voxel)
stlread (https://www.mathworks.com/matlabcentral/fileexchange/6678-stlread). 
usage:
--------------
matlab:
[F,V] = stlread('skullmesh.stl');
FV.faces=F;
FV.vertices=V;
Volume=polygon2voxel(FV,512,'none',true);
python:
import scipy.io as sio, import nrrd 
volume=sio.loadmat('Volumen.mat')['Volume']
nrrd.write('volume.nrrd',volume.astype(float64)) 
```

### STL files dimension calculation
```
dependency: Open3D(http://www.open3d.org/), numpy-stl(https://pypi.org/project/numpy-stl/)
installation: pip install open3d  pip install numpy-stl
usage: python stl_dimension.py
note: calculate the actual size of the STL files in millimeter (mm), which is the size of the 3D printed model.
```
