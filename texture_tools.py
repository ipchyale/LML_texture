import warnings
#import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import steerablepyrtexture as spt
from PIL import Image

from skimage.transform import rescale
from skimage import filters 
from skimage.color import rgb2gray
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted
from sklearn.covariance import empirical_covariance
from scipy.spatial.distance import cdist

from sklearn.utils.estimator_checks import check_estimator



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
    


def import_process_raking_folder(directory, output_tiles=False):
    """ Import all the tifs in a directory, preprocess, tile, and extract
        features from the images.
    

    Parameters
    ----------
    directory : str
        The location of the directory to import.
    output_tiles : bool, optional
        If True, will save the tile images in the directory. 
        The default is False.

    Returns
    -------
    id_labelvec : list of strings
        List of the names of each image (labeled per tile).
    sp_params : ndarray
        The feature vector describing each tile. Shape (n_tiles, n_parameters).

    """  
    #os.chdir(directory)
    num_import = len(glob.glob(directory + "\\*.tif"))
    if num_import == 0:
        print('No TIFF images found in directory')
    
    
    id_labelvec = []
    sp_params = np.zeros((2195,0))
    ct = 0
    for file in glob.glob(directory + "\\*.tif"):
        print(file)
        
        I = plt.imread(file)
        img = rgb2gray(I)
        imgout = standard_img_proc(img,3)
        tiles = image_tiles(imgout,256,2,3)
        
        for tilei in tiles:
            tmpparams = spt.texture_analyze(tilei, 5, 4, 7, vector_output=True)
            tmpparams = np.reshape(tmpparams,(2195,1))
            file_name = file[len(directory)+1:-4]
            id_name = file_name.split('_')[0]
            id_labelvec.append(id_name)
            sp_params = np.concatenate((sp_params, tmpparams), axis=1)
            
            if output_tiles:
                tmpImage = Image.fromarray(tilei)
                tmpImage.save(directory + '\\output_tiles1' + f'\\{ct:05}.tif')
            
            ct += 1
        
        if (ct/6)%10 == 0:
            print(f'{ct/num_import/6*100:.2f}% Complete')
        
            
        #if ct/6 >= 15:
            #break
    
    return id_labelvec, sp_params   


def _class_means(X, y):
    """Compute class means.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.
    Returns
    -------
    means : array-like of shape (n_classes, n_features)
        Class means.
    """
    classes, y = np.unique(y, return_inverse=True)
    cnt = np.bincount(y)
    means = np.zeros(shape=(len(classes), X.shape[1]))
    np.add.at(means, y, X)
    means /= cnt[:, None]
    return means
    

    
def _class_cov(X, y, priors, shrinkage=0):
    """Compute weighted within-class covariance matrix.
    The per-class covariance are weighted by the class priors.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.
    priors : array-like of shape (n_classes,)
        Class priors.
    shrinkage : float, default=0.0
        Shrinkage parameter, possible values:
          - float between 0 and 1: fixed shrinkage parameter.
    Returns
    -------
    cov : array-like of shape (n_features, n_features)
        Weighted within-class covariance matrix
    """
    classes = np.unique(y)
    cov = np.zeros(shape=(X.shape[1], X.shape[1]))
    for idx, group in enumerate(classes):
        Xg = X[y == group, :]
        tmp = empirical_covariance(Xg)
        tmp = (1-shrinkage)*tmp + shrinkage*np.diagflat(np.diag(tmp))
        cov += priors[idx] * np.atleast_2d(tmp)
    return cov


class LDAReducedRank(ClassifierMixin, TransformerMixin, BaseEstimator):
    """ Linear Discriminant Anaysis with regularization towards Diagonal
    
    Modification of the sklearn LinearDiscriminantAnalysis class.
    
    Parameters
    ----------
    gamma : float, default=0.0
        Regularization parameter that controls the off-diagonal elements of
        the covariance matrix. A value of 1 leaves only the variances in which 
        case the transform is a standardization and the classifier is 
        Gaussian Naive Bayes.
    priors : array-like of shape (n_classes,), default='uniform'
        The class prior probabilities. By default, the class proportions are
        assumed to be uniform for all classes. Setting to None results in
        the values being inferred from the training data.
    n_components : int, default=None
        Number of components (<= min(n_classes - 1, n_features)) for
        dimensionality reduction. If None, will be set to
        min(n_classes - 1, n_features). This parameter only affects the
        `transform` method.
    tol : float, default=1.0e-6
        Threshold for singular values of covariance relative to the largest
        singular value to be considered significant. Insignificant singular
        values are discarded.
    
    Attributes
    ----------
    priors_ : array-like of shape (n_classes,)
        Class priors (sum to 1).
    covariance_ : array-like of shape (n_features, n_features)
        Weighted within-class covariance matrix. It corresponds to
        `sum_k prior_k * C_k` where `C_k` is the covariance matrix of the
        samples in class `k`. The `C_k` are estimated using the (potentially
        shrunk) biased estimato
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each of the selected components.
        If ``n_components`` is not set then all components are stored and the
        sum of explained variances is equal to 1.0. Only available when eigen
        or svd solver is used.
        means_ : array-like of shape (n_classes, n_features)
        Class-wise means.
    scalings_ : array-like of shape (rank, n_classes - 1)
        Scaling of the features in the space spanned by the class centroids.
    xbar_ : array-like of shape (n_features,)
        Overall mean.
    classes_ : array-like of shape (n_classes,)
        Unique class labels.
    
    """
    
    def __init__(
        self,
        gamma=0,
        priors='uniform',
        n_components=None,
        tol=1e-6,
    ):
        self.gamma = gamma
        self.priors = priors
        self.n_components = n_components
        self.tol = tol  # used only in svd solver

    


    def fit(self, X, y):
        """Fit the Linear Discriminant Analysis model.
           .. versionchanged:: 0.19
              *store_covariance* has been moved to main constructor.
           .. versionchanged:: 0.19
              *tol* has been moved to main constructor.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = self._validate_data(
            X, y, ensure_min_samples=2, estimator=self, dtype=[np.float64, np.float32]
        )
        self.classes_ = unique_labels(y)
        n_samples, _ = X.shape
        n_classes = len(self.classes_)

        if n_samples == n_classes:
            raise ValueError(
                "The number of samples must be more than the number of classes."
            )

        if self.priors is None:  # estimate priors from sample
            _, y_t = np.unique(y, return_inverse=True)  # non-negative ints
            self.priors_ = np.bincount(y_t) / float(len(y))
        elif self.priors == 'uniform':
            self.priors_ = np.ones((n_classes,))/n_classes
        else:
            self.priors_ = np.asarray(self.priors)

        if (self.priors_ < 0).any():
            raise ValueError("priors must be non-negative")
        if not np.isclose(self.priors_.sum(), 1.0):
            warnings.warn("The priors do not sum to 1. Renormalizing", UserWarning)
            self.priors_ = self.priors_ / self.priors_.sum()

        # Maximum number of components no matter what n_components is
        # specified:
        max_components = min(len(self.classes_) - 1, X.shape[1])

        if self.n_components is None:
            self._max_components = max_components
        else:
            if self.n_components > max_components:
                raise ValueError(
                    "n_components cannot be larger than min(n_features, n_classes - 1)."
                )
            self._max_components = self.n_components

        # SVD fit
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        self.means_ = _class_means(X, y)
        self.covariance_ = _class_cov(X, y, self.priors_, shrinkage=self.gamma)
        self.xbar_ = np.dot(self.priors_, self.means_)
        
        U, S, _ = linalg.svd(self.covariance_)
        w_sphere =  np.matmul(U, np.power(np.sqrt(S), -1)*np.eye(len(S)))
        m_star = np.matmul(self.means_, w_sphere)

        B_star = empirical_covariance(m_star)
        U, S, _ = linalg.svd(B_star)
        
        rank = np.sum(S > self.tol * S[0])
        self.scalings_ = np.matmul(w_sphere, U[:, 0:rank])
        self.explained_variance_ratio_ = (S ** 2 / np.sum(S ** 2))
       
        return self
    
    def transform(self, X):
        """Project data to maximize class separation.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data.
        """

        check_is_fitted(self)
        X = self._validate_data(X, reset=False)
        X_new = np.dot(X - self.xbar_, self.scalings_)

        return X_new[:, : self._max_components]
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        C : array, shape [n_samples]
            Predicted class label per sample.
        """
        
        X = self._validate_data(X, reset=False)
        X_new = self.transform(X)
        means = self.transform(self.means_)
        
        dists = cdist(X_new, means)
        return self.classes_[dists.argmin(axis=1)]
    
    def predict_proba(self, X):
        """
        Predict class labels for samples in X.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        C : array, shape [n_samples]
            Predicted class label per sample.
        """
        
        X = self._validate_data(X, reset=False)
        X_new = self.transform(X)
        means = self.transform(self.means_)
        
        dists = cdist(X_new, means)
        log_proba = (-0.5 * dists**2) + np.log(self.priors_)
        
        proba = np.exp(log_proba)
        proba[proba == 0.0] += np.finfo(proba.dtype).tiny
        proba = proba/proba.sum(axis=1, keepdims=1)
        
        return proba


def mahal_dist(x1, x2, invcov):
    """ Mahalanobis distance between x1 and x2.
    

    Parameters
    ----------
    x1 : ndarray
        Array of n p-dimensional vectors [n x p].
    x2 : ndarray
        Array of m p-dimensional vectors [m x p].
    invcov : ndarray
        The inverse of the covariance matrix.

    Returns
    -------
    ndarray
        Pairwise Mahalanobis distances for each x1 and x2 pair [n x m].

    """

    D = np.zeros((x1.shape[0], x2.shape[0]))
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            dx = x1[i,:] - x2[j,:]
            D[i,j] = np.matmul(np.matmul(dx.T, invcov), dx)
    
    return np.sqrt(D)

def bhatta_dist_normal(x1, x2):
    """ Estimated Bhattacharyya distance between x1 and x2 assuming both 
        Are normally distributed with estimates based empricially on x1 and x2.
    

    Parameters
    ----------
    x1 : ndarray
        Array of n p-dimensional vectors [n x p].
    x2 : ndarray
        Array of m p-dimensional vectors [m x p].

    Returns
    -------
    ndarray
        Pairwise Bhattacharyya distances for each x1 and x2 pair [n x m].

    """

    mean1 = np.average(x1, axis=0).reshape(-1, 1)
    mean2 = np.average(x2, axis=0).reshape(-1, 1)
    cov1 = np.cov(x1, rowvar=False)
    cov2 = np.cov(x2, rowvar=False) 
    cov3 = 0.5 * (cov1 + cov2)
    logdet1 = np.linalg.slogdet(cov1)[1]
    logdet2 = np.linalg.slogdet(cov2)[1]
    logdet3 = np.linalg.slogdet(cov3)[1]
    

                
    dist_bhatta = 0/8 * np.matmul ( np.matmul( (mean1-mean2).T, np.linalg.pinv(cov3)), 
                                    (mean1-mean2) ) \
                 + 1/2 * (logdet3 - 0.5*(logdet1 + logdet2))

    return dist_bhatta

