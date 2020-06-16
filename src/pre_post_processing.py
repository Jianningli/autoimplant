# Denoising using 3D connected-component
import cc3d
import numpy as np
import nrrd


def skull_id(labels_out):
	labels_out=labels_out.reshape((1,-1))
	labels_out=labels_out[0,:]
	label=np.unique(labels_out)
	hist, bin_edges=np.histogram(labels_out,bins=label)
	hist=np.ndarray.tolist(hist)
	hist_=hist
	hist_=np.array(hist_)
	hist.sort(reverse = True)
	#print('hist',hist)
	idx=(hist_==hist[1])
	idx=idx+1-1
	idx_=np.sum(idx*label[0:len(idx)])
	print('idx',idx_)
	return idx_





def pre_processing(data):
	# original data (512,512,z)
	labels_out = cc3d.connected_components(data.astype('int32'))
	skull_label=skull_id(labels_out)
	skull=(labels_out==skull_label)
	skull=skull+1-1
	return skull	



def skull_id1(labels_out):
	labels_out=labels_out.reshape((1,-1))
	labels_out=labels_out[0,:]
	label=np.unique(labels_out)
	hist, bin_edges=np.histogram(labels_out,bins=label)
	hist=np.ndarray.tolist(hist)
	hist_=hist
	hist_=np.array(hist_)
	hist.sort(reverse = True)
	#print('hist',hist)
	idx=(hist_==hist[2])
	idx=idx+1-1
	idx_=np.sum(idx*label[0:len(idx)])
	print('idx',idx_)
	return idx_





def pre_processing1(data):
	# original data (512,512,z)
	labels_out = cc3d.connected_components(data.astype('int32'))
	skull_label=skull_id1(labels_out)
	skull=(labels_out==skull_label)
	skull=skull+1-1
	return skull	


def post_processing(data):
	# original data (512,512,z)
	# or implants
	labels_out = cc3d.connected_components(data.astype('int32'))
	skull_label=skull_id(labels_out)
	skull=(labels_out==skull_label)
	skull=skull+1-1
	return skull



