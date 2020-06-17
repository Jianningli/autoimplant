
from glob import glob
import numpy as np
import nrrd
from scipy.ndimage import zoom
import random
import pymrt.geometry

'''
_note:_** the current code provide functionalities to generate cubic defect  
`generate_cude(defect_size)` and spherical dfects `generate_sphere(defect_size)`
'''

#**************************Square Hole Generation**************************************

def generate_hole_implants(data,cube_dim):
	x_=data.shape[0]
	y_=data.shape[1]
	z_=data.shape[2]
	full_masking=np.ones(shape=(x_,y_,z_))
	x=random.randint(int(cube_dim/2),x_-int(cube_dim/2))
	y=random.randint(int(cube_dim/2),y_-int(cube_dim/2))
	z=int(z_*(3/4))
	cube_masking=np.zeros(shape=(cube_dim,cube_dim,z_-z))
	print(cube_masking.shape)
	full_masking[x-int(cube_dim/2):x+int(cube_dim/2),y-int(cube_dim/2):y+int(cube_dim/2),z:z_]=cube_masking
	return full_masking


def generate_cude(size):
	for i in range(len(pair_list)):
		print('generating data:',pair_list[i])
		temp,header=nrrd.read(pair_list[i])

		full_masking=generate_hole_implants(temp,size)
		
		c_masking_1=(full_masking==1)
		c_masking_1=c_masking_1+1-1

		defected_image=c_masking_1*temp

		c_masking=(full_masking==0)
		c_masking=c_masking+1-1
		implants=c_masking*temp

		f1=defected_dir+pair_list[i][-10:-5]+'.nrrd'
		f2=implant_dir+pair_list[i][-10:-5]+'.nrrd'
		nrrd.write(f1,defected_image,header)
		nrrd.write(f2,implants,header)





#****************************Sphere Hole Generation********************************

def sphere(shape, radius, position):
    semisizes = (radius,) * 3
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        arr += (np.abs(x_i / semisize) ** 2)
    return arr <= 1.0



def generate_sphere_hole_implants(data,size):
	x_=data.shape[0]
	y_=data.shape[1]
	z_=data.shape[2]
	z=int(z_*(3/4))
	x=random.randint(z_+size-z,x_-(z_+size-z))
	y=random.randint(z_+size-z,y_-(z_+size-z))
	arr = sphere((x_, y_, z_+size),z_+size-z, (x, y, z))
	return arr

def generate_sphere(size1):
	for i in range(len(pair_list)):
		size=size1
		print('generating data:',pair_list[i])
		temp=nrrd.read(pair_list[i])[0]
		print(temp.shape)
		temp_=np.zeros(shape=(temp.shape[0],temp.shape[1],temp.shape[2]+size))
		temp_[:,:,0:temp.shape[2]]=temp
		arr=generate_sphere_hole_implants(temp,size)
		arr=(arr==1)
		arr=arr+1-1	
		implants=arr*temp_
		arr=(arr==0)
		arr=arr+1-1
		defected_image=arr*temp_
		f1=defected_dir+pair_list[i][-10:-5]+'.nrrd'
		f2=implant_dir+pair_list[i][-10:-5]+'.nrrd'
		nrrd.write(f1,defected_image[:,:,0:temp.shape[2]].astype('float64'))
		nrrd.write(f2,implants[:,:,0:temp.shape[2]].astype('float64'))
		print(defected_image[:,:,0:temp.shape[2]].shape)






if __name__ == "__main__":
	# Directory of the healthy skull
	pair_list=glob('{}/*.nrrd'.format('C:/Users/Jianning/Desktop'))

	defected_dir='C:/Users/Jianning/Desktop/1/'
	implant_dir='C:/Users/Jianning/Desktop/2/'

	generate_cude(128)
	#generate_sphere(20)























