import numpy as np
import nrrd
from glob import glob
import scipy
import scipy.ndimage
import random
from scipy.ndimage import zoom


def resizingbbox(data,z_dim):
    a,b,c=data.shape
    resized_data = zoom(data,(512/a,512/b,z_dim/c),order=2, mode='constant')  
    return resized_data


def bbox_cal(data,dim):
    a=resizingbbox(data,dim)
    a=np.round(a)    
    x0=np.sum(a,axis=2)
    xx=np.sum(x0,axis=1)
    yy=np.sum(x0,axis=0)
    resx = next(x for x, val in enumerate(list(xx)) 
                                      if val > 0)

    resxx = next(x for x, val in enumerate(list(xx)[::-1]) 
                                      if val > 0)


    resy = next(x for x, val in enumerate(list(yy)) 
                                      if val > 0)

    resyy = next(x for x, val in enumerate(list(yy)[::-1]) 
                                      if val > 0)
    z0=np.sum(a,axis=1)
    zz=np.sum(z0,axis=0)
    resz = next(x for x, val in enumerate(list(zz)) 
                                      if val > 0)

    reszz = next(x for x, val in enumerate(list(zz)[::-1]) 
                                      if val > 0)

    return resx,resxx,resy,resyy,resz,reszz



bbox_list=glob('{}/*.nrrd'.format('../predictions_n1'))
pred_list=glob('{}/*.nrrd'.format('../predictions_n2'))
original_list=glob('{}/*.nrrd'.format('../testing_defective_skulls'))
save_dir='../final_implants/'


for i in range(len(pred_list)):

    data,hd=nrrd.read(original_list[i])
    print('original data',original_list[i])


    bbox,hb=nrrd.read(bbox_list[i])
    print('bbox',bbox_list[i])

    pred,hd=nrrd.read(pred_list[i])
    print('initial pred',pred_list[i])

    resx,resxx,resy,resyy,resz,reszz=bbox_cal(bbox,data.shape[2])


    x_len=512+40-(resxx+resx)

    y_len=512+40-(resyy+resy)


    xl=int(x_len/2)
    xl_r=x_len-xl

    yl=int(y_len/2)
    y1_r=y_len-yl



    boundingboximp=pred[128-xl:128+xl_r,128-yl:128+y1_r,]
    orig=np.zeros(shape=(512,512,data.shape[2]))
    margin=20

    orig[resx-margin:512-resxx+margin,resy-margin:512-resyy+margin,data.shape[2]-128:data.shape[2]]=boundingboximp

    outfile=save_dir+bbox_list[i][-16:-5]+'.nrrd'
    nrrd.write(outfile,orig,hd)









