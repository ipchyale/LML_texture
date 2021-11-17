import numpy as np
from PIL import Image
import steerablepyrtexture as spt

from texture_tools import LinearDimReduce
from texture_tools import image_tiles

# %%
# Load the dimensionality reduction transform.
ldr = LinearDimReduce()
ldr.load(r'C:\...\ldr01.pickle')

# Load a processed image.
file_name = r'C:\...\image01.tif'
img = np.asarray(Image.open(file_name))

# Cut the image into 256x256 tiles and compute features from each.
tiles = image_tiles(img,256,2,3)
     
sp = np.zeros((0,2195))           
for tilei in tiles:
    tmpparams = spt.texture_analyze(tilei, 5, 4, 7, vector_output=True)
    tmpparams = np.reshape(tmpparams, (1, tmpparams.size))
    sp = np.concatenate((sp, tmpparams), axis=0)

# Transform the features into the 100-dimensional space and save
# to be imported in Colabs to run inference.    
x2 = ldr.transform(sp)

# optionally, average over the tiles
# x2 = np.average(x2, axis=0).reshape((1, x2.shape[1]))

np.save(r'C:\...\image01_features.npy', x2)