import warnings

import numpy as np
from skimage.transform import rescale
from skimage import filters 



def imgcrop_pct(img,pct):
    """ Crop a fixed percent from both the horizontal and vertical dimension 
    of an image. Cropped equally on both top/bottom and sides to maintain
    the centering of the original image.
    

    Parameters
    ----------
    img : ndarray
        The image to crop.
    pct : float
        The percentage of horizontal and vertical dimension to crop.

    Returns
    -------
    imgout : ndarray
        The cropped image.

    """
    rmidx0 = int(np.round(pct/100 * np.size(img,0)/2))
    rmidx1 = int(np.round(pct/100 * np.size(img,1)/2))
    
    imgout = img[rmidx0:-1-rmidx0,rmidx1:-1-rmidx1]
    
    return imgout


def standard_img_proc(img,ds=1):
    """ Standard raking light image preprocessing for texture analysis.
    

    Parameters
    ----------
    img : ndarray
        The raw image to be preprocessed.
    ds : float
        Optional downsampling factor.

    Returns
    -------
    imgout : ndarray
        A raking light image corrected for vignetting and exposure.

    """

    scale = 10/np.size(img,0)
    img = rescale(img,scale/.00488281)
    
    # Approximate flat-field correction
    FF = filters.gaussian(img, sigma=100)
    FFmean = np.average(FF, axis=None)  
    imgC = np.divide(img,FF)*FFmean   
    
    # Clean up image.  Crop, adjust exposure, and remove out-of-bounds values.
    imgout = imgcrop_pct(imgC,10)
    imgout = imgout / np.percentile(imgout,99.9)*0.975
    #imgout = imgout / np.average(imgout) * 0.75
    imgout[imgout < 0] = 0
    imgout[imgout > 1] = 1
    
    
    if ds > 1:
        imgout = rescale(imgout, 1/ds)
    
    return imgout


def image_tiles(img,tilesize,n0,n1):
    """ Cut an image into square tiles of an arbitrary size and number of
    vertically and horizontally moved. The tiles will be equally spaced 
    and may overlap each other. If only one tile is chosen in a dimension, it
    will be located randomly within that dimension.
    

    Parameters
    ----------
    img : ndarray
        Image to be divided.
    tilesize : int
        The outputsize, in pixels, of the tiles.
    n0 : int
        The number of tiles to fit into the vertical dimension.
    n1 : int
        The number of tile sto fit into the horizontal dimension.

    Returns
    -------
    tiles : list
        A list of image arrays -- each corresponding to a tile.

    """

    len0 = np.size(img,0)
    len1 = np.size(img,1)
    offset0 = 0
    offset1 = 0

    if n0 > 1:
        shift0 = (len0 - tilesize)//(n0-1)
        vovl = (tilesize-shift0)/tilesize * 100
        if vovl > 0:
            warnings.warn(f"Vertical overlap = {vovl}%")
    else:
        offset0 = int(np.floor((len0 - tilesize)*np.random.rand()))
        shift0 = 0
        
    if n1 > 1:
        shift1 = (len1 - tilesize)//(n1-1)
        hovl = (tilesize-shift1)/tilesize * 100
        if hovl > 0:
            warnings.warn(f"Horizontal overlap = {hovl}%")
    else:
        offset1 = int(np.floor((len1 - tilesize)*np.random.rand()))
        shift1 = 0
        
        
    tiles = []    
    for i in range(n0):
        for j in range(n1):
            tiles.append(img[ i*shift0 + offset0 : 
                              tilesize + i*shift0 + offset0, 
                              j*shift1 + offset1 : 
                              tilesize + j*shift1 + offset1 ])

    return tiles            
    

