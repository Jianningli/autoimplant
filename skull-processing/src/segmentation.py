# python scripts for skull segmentation from CT scan
# read nrrd files


import numpy as np
import nrrd
from glob import glob

if __name__ == '__main__':
    # directory of original nrrd files
    data_dir = "D:/skull-nrrd"
    data_list=glob('{}/*.nrrd'.format(data_dir))
    # directory to save the segmented nrrd file
    save_dir = "D:/skull-nrrd/segmented/"
    for i in range(len(data_list)):
	print('current data to segment:',data_list[i])
	# read nrrd file. data: skull volume. header: nrrd header
	data,header=nrrd.read(data_list[i])
	# set threshold, 100--max
	segmented_data=(data>=100)
	segmented_data=segmented_data+1-1
	# file name of the cleaned skull
	filename=save_dir+data_list[i][-10:-5]+'.nrrd'
	print('writing the cleaned skull to nrrd...')
	nrrd.write(filename,segmented_data,h)
	print('writing done...')
