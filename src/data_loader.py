from __future__ import division
import numpy as np
import scipy
import scipy.ndimage
import random
import nrrd
import numpy as np
from scipy.ndimage import zoom



def resizing(label):
    a,b,c=label.shape
    resized_data = zoom(label,(128/a,128/b,64/c),order=2, mode='constant')  
    return resized_data

def load_batch_pair(list1,list2):
    idx=random.randrange(0,100,1)
    data,hd=nrrd.read(list1[idx])
    print('data',list1[idx])
    label,hl=nrrd.read(list2[idx])
    print('label',list2[idx])
    print('shape:',label.shape)
    data_defected=resizing(data)
    label_=resizing(label)
    data_defected=np.expand_dims(data_defected,axis=0)
    data_defected=np.expand_dims(data_defected,axis=4)
    label_=np.expand_dims(label_,axis=0)
    label_=np.expand_dims(label_,axis=4)
    return data_defected,label_,hd,hl


def load_batch_pair_test(list1,idx):
    data,hd=nrrd.read(list1[idx])
    print('data',list1[idx])
    data_defected=resizing(data)
    data_defected=np.expand_dims(data_defected,axis=0)
    data_defected=np.expand_dims(data_defected,axis=4)
    return data_defected,hd



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


margin=20


def padding(data):
    temp=np.zeros(shape=(256,256,128))
    xl=int(data.shape[0]/2)
    xl_r=data.shape[0]-xl

    yl=int(data.shape[1]/2)
    y1_r=data.shape[1]-yl    
    temp[128-xl:128+xl_r,128-yl:128+y1_r,]=data

    return temp



def load_bbox_pair(list_bbox,list_data,list_label):
    idx=random.randrange(0,100,1)

    bbox,hb=nrrd.read(list_bbox[idx])
    print('bbox',list_bbox[idx])

    data,hd=nrrd.read(list_data[idx])
    print('data',list_data[idx])

    label,hl=nrrd.read(list_label[idx])
    print('label',list_label[idx])

    resx,resxx,resy,resyy,resz,reszz=bbox_cal(bbox,data.shape[2])
    print(resx)
    print(resxx)
    print(resy)
    print(resyy)

    data_inp=data[resx-margin:512-resxx+margin,resy-margin:521-resyy+margin,data.shape[2]-128:data.shape[2]]
    data_lb=label[resx-margin:512-resxx+margin,resy-margin:512-resyy+margin,data.shape[2]-128:data.shape[2]]

    data_inp=padding(data_inp)
    data_lb=padding(data_lb)

    data_inp=np.expand_dims(data_inp,axis=0)
    data_inp=np.expand_dims(data_inp,axis=4)

    data_lb=np.expand_dims(data_lb,axis=0)
    data_lb=np.expand_dims(data_lb,axis=4)

    return data_inp,data_lb,hd,hl


def load_bbox_pair_test(list_bbox,list_data,idx):


    bbox,hb=nrrd.read(list_bbox[idx])
    print('bbox',list_bbox[idx])

    data,hd=nrrd.read(list_data[idx])
    print('data',list_data[idx])



    resx,resxx,resy,resyy,resz,reszz=bbox_cal(bbox,data.shape[2])
    print(resx)
    print(resxx)
    print(resy)
    print(resyy)

    data_inp=data[resx-margin:512-resxx+margin,resy-margin:521-resyy+margin,data.shape[2]-128:data.shape[2]]
    print(data_inp.shape)


    data_inp=padding(data_inp)


    data_inp=np.expand_dims(data_inp,axis=0)
    data_inp=np.expand_dims(data_inp,axis=4)



    return data_inp,hd




