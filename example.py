import glob

import numpy as np
import matplotlib.pyplot as plt
import steerablepyrtexture as spt
from PIL import Image

from skimage.color import rgb2gray

from texture_tools import import_process_raking_folder
from texture_tools import sp_param_extract_folder

# %% Import a folder of raking light images
directory_list = []
directory = r'C:...\texture_binder2_jd'  
directory_list.append(directory)

output_dir = r'C:...\output_images'

import_process_raking_folder(directory_list, output_dir,
                                        same_ff = True,
                                        output_images=True, output_params=False)

# %% Calculate SP features from the processed images
IDs_NEW, SP_NEW = sp_param_extract_folder(output_dir)

np.savetxt(r'C:...\IDvector.csv', IDs_NEW, delimiter=", ", fmt='% s')
np.save(r'C:...\SPparams.npy', SP_NEW)