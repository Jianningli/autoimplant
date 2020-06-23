
from glob import glob
import numpy as np
import nrrd
from scipy.ndimage import zoom
import random
import pymrt.geometry
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_real_hole(data,radius,loc,loc1,idxxx):
	x_=data.shape[0]
	y_=data.shape[1]
	z_=data.shape[2]
	full_masking=np.zeros(shape=(x_,y_,z_))
	
	#*** Modify the proportion to change the size of the hole(modify this line)
	masked_x=int(x_*1/3)
	masked_y=int(y_*1/3)
	# leave this line unchanged
	masked_z=int(z_/3)
    #********************

	cylinder1=pymrt.geometry.cylinder((masked_x,masked_y,masked_z),int(z_),radius,2,position=(1/6,1/6,1))
	cylinder2=pymrt.geometry.cylinder((masked_x,masked_y,masked_z),int(z_),radius,2,position=(1/6,5/6,1))
	cylinder3=pymrt.geometry.cylinder((masked_x,masked_y,masked_z),int(z_),radius,2,position=(5/6,1/6,1))
	cylinder4=pymrt.geometry.cylinder((masked_x,masked_y,masked_z),int(z_),radius,2,position=(5/6,5/6,1))

	cylinder1=cylinder1+1-1
	cylinder2=cylinder2+1-1
	cylinder3=cylinder3+1-1
	cylinder4=cylinder4+1-1


	cube=np.zeros(shape=(masked_x,masked_y,masked_z))

	cube[int((1/6)*masked_x):int((5/6)*masked_x),int((1/6)*masked_y):int((5/6)*masked_y),0:masked_z]=1

	combined=cube+cylinder1+cylinder2+cylinder3+cylinder4
	combined=(combined!=0)
	combined=combined+1-1

	if idxxx==1:
		full_masking[int((loc/4)*x_):int((loc/4)*x_)+masked_x,int((1/2)*y_):int((1/2)*y_)+masked_y,z_-masked_z:z_]=combined
	if idxxx==2:
		full_masking[int((loc/8)*x_):int((loc/8)*x_)+masked_x,int((1/2)*y_):int((1/2)*y_)+masked_y,z_-masked_z:z_]=combined
	if idxxx==3:
		full_masking[int((1/2)*x_):int((1/2)*x_)+masked_x,int((1/2)*y_):int((1/2)*y_)+masked_y,z_-masked_z:z_]=combined
	if idxxx==4:
		full_masking[int((loc/4)*x_):int((loc/4)*x_)+masked_x,int((loc1/8)*y_):int((loc1/8)*y_)+masked_y,z_-masked_z:z_]=combined
	if idxxx==5:
		full_masking[int((loc/8)*x_):int((loc/8)*x_)+masked_x,int((loc1/8)*y_):int((loc1/8)*y_)+masked_y,z_-masked_z:z_]=combined
	if idxxx==6:
		full_masking[int((1/2)*x_):int((1/2)*x_)+masked_x,int((loc/8)*y_):int((loc/8)*y_)+masked_y,z_-masked_z:z_]=combined
	if idxxx==7:
		full_masking[int((loc/8)*x_):int((loc/8)*x_)+masked_x,int((loc/4)*y_):int((loc/4)*y_)+masked_y,z_-masked_z:z_]=combined
	if idxxx==8:
		full_masking[int((loc/4)*x_):int((loc/4)*x_)+masked_x,int((loc/4)*y_):int((loc/4)*y_)+masked_y,z_-masked_z:z_]=combined
	if idxxx==9:
		full_masking[int((1/3)*x_):int((1/3)*x_)+masked_x,int((loc/4)*y_):int((loc/4)*y_)+masked_y,z_-masked_z:z_]=combined
	return full_masking





def generate_real(temp,size,loc,loc1,idxxx):
	full_masking=generate_real_hole(temp,size,loc,loc1,idxxx)
	full_masking=(full_masking==1)
	full_masking=full_masking+1-1		
	implants=full_masking*temp

	c_masking=(full_masking==0)
	c_masking=c_masking+1-1
	defected_image=c_masking*temp
	return implants, defected_image




if __name__ == "__main__":
	data,header=nrrd.read('C:/Users/Jianning/Desktop/new18data/Case24 cropped bone.nrrd')
	defected_real_dir='C:/Users/Jianning/Desktop/new18data/case24_defected_skulls/'
	implant_real_dir='C:/Users/Jianning/Desktop/new18data/case24_implants/'
	f=['01','02','03','04','05','06','07','08','09','10']
	for i in range(10):
		size=np.random.randint(6,14,1)[0]
		loc=np.random.randint(1,3,1)[0]
		loc1=np.random.randint(1,3,1)[0]
		idxxx=np.random.randint(1,10,1)[0]
		print(idxxx)
		f1=defected_real_dir+'Case24_Defects'+f[i]+'.nrrd'
		f2=implant_real_dir+'Case24_Implants'+f[i]+'.nrrd'
		implants,defected_image=generate_real(data,size,loc,loc1,idxxx)
		nrrd.write(f1,defected_image,header)
		nrrd.write(f2,implants,header)


























