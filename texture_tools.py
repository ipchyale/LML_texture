import warnings
import time
import glob

import numpy as np
import matplotlib.pyplot as plt
import steerablepyrtexture as spt
from PIL import Image
import pickle

from skimage.transform import rescale, resize
from skimage import filters 
from skimage.color import rgb2gray
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted
from sklearn.covariance import empirical_covariance
from scipy.spatial.distance import cdist, squareform
from sklearn.cluster import MeanShift

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
    
    img = imgcrop_pct(img,5)
        
    tmp = img.flatten()
    y, _ = np.histogram(tmp, bins=np.linspace(0,1,101))
    peak = np.argmax(y)
    exposure_correction = 0.5/(peak/100)
    
    imgout = img*exposure_correction
    
    if ds != 1:
        imgout = rescale(imgout, 1/ds)
    
    imgout[imgout < 0] = 0
    imgout[imgout > 1] = 1
    
    
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
    


def import_process_raking_folder(directory_list, output_directory, 
                                 same_ff=False,
                                 output_images=True, overwrite=False,
                                 output_params=False, output_ff=False):
    """ Import all the tifs in a directory, preprocess, tile, and extract
        features from the images.
    

    Parameters
    ----------
    directory : str
        The directory to import.
    output_directory: str
        The directory to save output images.
    same_ff : bool
        If True, the same flat-field correction will be applied to every image
        that is imported, but the average will still use only images part of 
        the largest identified cluster.
        The default is False.
    output_images : bool, optional
        If True, will save the images in the directory.
        The default is True.
    overwrite : bool, optional
        If False, a new image will be created with a new filename for any 
        images with the same name. If True, repeated names will cause the 
        previous image to be overwritten.
        The default is False.
    output_params : bool, optional
        If True, will compute the steerable pyramid parameters. 
        The default is False.
    output_ff : bool, optional
        If True, will save the cluster assignments and cluster flat-fields in
        the output directory as numpy files.

    Returns
    -------
    FF_cluster : list of ndarrays
        List containing the estimated flat field image for each cluster.
    id_labelvec : list of strings
        List of the names of each image (labeled per tile).
    sp_params : ndarray
        The feature vector describing each tile. Shape (n_tiles, n_parameters).

    """  
    
    
    ### MAIN PARAMETERS
    # Assigned clusters smaller than size will be unassigned, but may be 
    # merged back into larger clusters on the next step
    MIN_CLUSTER_SIZE = 15 
    
    # Ratio of cluster-to-cluster distance used for merging
    MERGE_OUTLIER_FACTOR = 0.75
    
    # If there is only 1 cluster used this distance for merging
    OUTLIER_MAX_DIST = 5
    
    # The alpha value is the weighting factor between the averaged FF and the 
    # Gaussian blur FF. For very large clusters alpha approaches 1.0. This 
    # value should be greater than MIN_CLUSTER_SIZE to avoid negative weights.
    ALPHA_NUMERATOR = 8
    if ALPHA_NUMERATOR >= MIN_CLUSTER_SIZE:
        ALPHA_NUMERATOR  = MIN_CLUSTER_SIZE + 1
        
    # The value of alpha above which the Gaussian FF will be ignored. 
    # Effectively makes it so large enough clusters will only be corrected
    # with the FF estimated by averaging the cluster.
    ALPHA_CUTOFF = 0.9
    
    
    file_list = []
    for i in range(len(directory_list)):
        file_list.extend(glob.glob(directory_list[i] + "\\*.tif"))
        
    num_import = len(file_list)
    print('  ')
    if num_import == 0:
        print('No TIFF images found in directory')
    else:
        print(f'{num_import} images to import \n')
    
    # Import and downsample the whole list to get FF estimates for clustering.
    image_sizes = []
    image_stacks = []
    image_sizes_idx = []
    ct = 0
    for file in file_list:
        
        I = np.asarray(Image.open(file))
        img = rgb2gray(I)
        
        tmp = np.array([img.shape])
        
        img2 = rescale(img, 0.10)
        
        if not np.all(np.any(tmp == image_sizes, axis=0)):
            print(tmp)
            image_sizes.append(tmp)
            image_stacks.append(np.expand_dims(img2, axis=2))
            image_sizes_idx.append(np.array([ct]))
            
            if same_ff:
                if len(image_sizes) > 1:
                    warnings.warn('Multiple image resolutions incompatible ' +
                                  'with same_ff option')
                    return None
        else:
            idx = np.nonzero(np.all(np.all(tmp == image_sizes, axis=1), axis=1))[0][0]
            image_stacks[idx] = np.concatenate((image_stacks[idx], 
                                                np.expand_dims(img2, axis=2)),
                                                axis=2)
            image_sizes_idx[idx] = np.concatenate((image_sizes_idx[idx], 
                                                   np.array([ct])), axis=0)
        
        if (ct)%25 == 0:
                print(f'Initial Import {ct/num_import*100:.2f}% Complete')
        
        ct += 1
        
    num_imported = ct
    
    ## Clustering
    
    cluster_size_idx = []
    cluster_idx = []
    cluster_images = []
    cluster_FF = []
    tmp_idx = []
    FF2_dists = []
    cluster_center_dists = []
    cluster_center_img = []
    for idx in range(len(image_sizes)):
        FF = np.zeros_like(image_stacks[idx])
        FF2 = np.zeros_like(FF)
        tmp = rescale(FF[:,:,0], 0.25)
        FF2_feat = np.zeros((tmp.shape[0], tmp.shape[1], FF.shape[2]))
        
        cluster_dists = []
        cluster_dist_range = []
        for i in range(image_stacks[idx].shape[2]):
            FF[:,:,i] = filters.gaussian(np.squeeze(image_stacks[idx][:,:,i]),
                                         sigma=12)
            FF2[:,:,i] = FF[:,:,i]/FF[:,:,i].mean()
            FF2_feat[:,:,i] = rescale(FF2[:,:,i],0.25)
            if (i)%100 == 0:
                print(f'FF2 {i/image_stacks[idx].shape[2]*100:.2f}% Complete')
        
        
        # Calculate distances for clustering
        tmp = np.reshape(FF2_feat, 
                         (FF2_feat.shape[0]*FF2_feat.shape[1],FF2_feat.shape[2])).T
        FF2_dists.append(cdist(tmp, tmp))
        
        # Use mean shift clustering algorithm to assign clusters
        clustering = MeanShift(cluster_all=False).fit(tmp)
        num_clusts = clustering.cluster_centers_.shape[0]
        clust_delete = []
        for j in range(num_clusts):
            b = np.nonzero(clustering.labels_ == j)[0]
            if b.size < MIN_CLUSTER_SIZE:
                clustering.labels_[b] = -1
                clust_delete.append(j)
        
        clust_delete = np.array(clust_delete)
        mask = np.ones((num_clusts,), dtype=bool)
        mask[clust_delete] = False
        clustering.cluster_centers_ = clustering.cluster_centers_[mask,:]
                 
        num_clusts = clustering.cluster_centers_.shape[0]                              
                
        cluster_center_dists.append(cdist(clustering.cluster_centers_, 
                                          clustering.cluster_centers_))
        
        # Attempt to recluster unassigned images
        if num_clusts > 1:
            unclust_max = MERGE_OUTLIER_FACTOR \
                          * squareform(cluster_center_dists[-1]).min()
        else:
            unclust_max = OUTLIER_MAX_DIST
        
        for j in range(num_clusts):
            cluster_center_img.append(np.reshape(clustering.cluster_centers_[j,:], 
                                        (FF2_feat.shape[0], FF2_feat.shape[1])))
        
        for j in range(num_clusts):
            b = np.nonzero(clustering.labels_ == j)[0]
            tmp_dists = cdist(clustering.cluster_centers_[j:j+1,:], tmp[b])
            cluster_dist_range.append(tmp_dists.max()*1.4)
            cluster_dists.append(tmp_dists)
            
        cluster_dist_range = np.array(cluster_dist_range)
        cluster_dist_range[cluster_dist_range > unclust_max] = unclust_max
    
        b = np.nonzero(clustering.labels_ == -1 )[0]
        unclust_dists = cdist(clustering.cluster_centers_, 
                                              tmp[b])
        
        if unclust_dists.size > 0:
            for j in range(unclust_dists.shape[1]):
                c = np.argmin(unclust_dists[:,j])
                if unclust_dists[c,j] < cluster_dist_range[c]:
                    clustering.labels_[b[j]] = c
        
        # Stack clusters
        for i in np.unique(clustering.labels_):
            if i==-1:
                continue
            
            FF_idx = np.nonzero(clustering.labels_ == i)[0]
            
            tmp_idx.append(FF_idx)
            cluster_idx.append(image_sizes_idx[idx][FF_idx])
            cluster_images.append(np.average(image_stacks[idx][:,:,FF_idx], 
                                             axis=2))
            cluster_FF.append(FF2[:,:,FF_idx])
            cluster_size_idx.append(idx)
            
    tmp = cluster_idx[0]
    for i in range(1,len(cluster_idx)):
        tmp = np.concatenate((tmp, cluster_idx[i]), axis=0)

    unclustered_idx = np.setdiff1d(np.arange(ct),tmp)
    
    # Define clustering outputs
    FF_cluster = [np.array([])]*len(cluster_idx)
    idx_cluster = np.ones((num_imported,))*-1
    idx_cluster = idx_cluster.astype(int)
    for i in range(len(cluster_idx)):
        FF_cluster[i] = resize(cluster_images[i], 
                               (image_sizes[cluster_size_idx[i]][0,0], 
                                image_sizes[cluster_size_idx[i]][0,1])) 
        for j in range(cluster_idx[i].size):
            idx_cluster[cluster_idx[i][j]] = i
            
            
    cluster_sizes = []
    for i in range(len(cluster_idx)):
        cluster_sizes.append(cluster_idx[i].size)
        
    print(f'Number of flat field clusters: {len(cluster_idx)}')
    print(f'Number of images without clusters: {len(unclustered_idx)}')
    print(' ')
    
    if same_ff: 
        if len(cluster_idx) > 1:
            warnings.warn('Multiple clusters identified with same_ff ON')
         
            cluster_sizes = np.array(cluster_sizes)
            main_cluster_idx = np.argmin(cluster_sizes)
            
        else:
            main_cluster_idx = 0
    
    # Save the estimated cluster-based flat-fields
    if output_ff:
        out_file = output_directory + '\\cluster_assignments.npy'
        np.save(out_file, idx_cluster)
        
        out_file = output_directory + '\\cluster_FF.npy'
        np.save(out_file, FF_cluster)
        
    ## Correction
    
    id_labelvec = []
    sp_params = np.zeros((0,2195))
    ct = 0
    for file in file_list:
        
        
        I = plt.imread(file)
        img = rgb2gray(I)
        
        
        tmp_idx = idx_cluster[ct]
        if tmp_idx >= 0:
            if same_ff:
                FF = FF_cluster[tmp_idx]
            else:
                alpha = 1 - ALPHA_NUMERATOR/cluster_sizes[tmp_idx]
                
                if alpha > ALPHA_CUTOFF:
                    FF = FF_cluster[tmp_idx]
                else:
                    FF1 = filters.gaussian(img, sigma=120)
                    FF = alpha*FF_cluster[tmp_idx] + (1-alpha)*FF1
        
        else:
            if same_ff:
                FF = FF_cluster[main_cluster_idx]
            else:
                FF = filters.gaussian(img, sigma=120)

            
        FFmean = np.average(FF, axis=None)  
        imgC = np.divide(img,FF)*FFmean  
        
        imgout = standard_img_proc(imgC, ds=3)
    
    ## Steerable Pyramid Parameters
    
        if output_params:
            
            tiles = image_tiles(imgout,256,2,3)
                
            for tilei in tiles:
                tmpparams = spt.texture_analyze(tilei, 5, 4, 7, 
                                                vector_output=True)
                tmpparams = np.reshape(tmpparams,(1,2195))
                id_name = file.split('\\')[-1][:-4].split('_')[0]
                id_labelvec.append(id_name)
                sp_params = np.concatenate((sp_params, tmpparams), axis=0)
            
    
        
    ## Image file output
        if output_images:
            id_name = file.split('\\')[-1][:-4]
            tmpImage = Image.fromarray(imgout)
            out_file = output_directory + '\\' + id_name +'.tif'
            if not overwrite:
                repeat_ct = 1
                while glob.glob(out_file):
                    repeat_ct += 1
                    out_file = output_directory + '\\' + id_name \
                               + f'_{repeat_ct:2d}' + '.tif'
                
            tmpImage.save(out_file)
                
        if (ct)%25 == 0:
                print(f'Processing {ct/num_imported*100:.2f}% Complete')
        
        ct += 1
    
    if output_params:
        return id_labelvec, sp_params
    else:
        return FF_cluster
    
    
def sp_param_extract_folder(directory):
    """
    Calculate steerable pyramid features from a directory containing
    pre-processed texture .tifs

    Parameters
    ----------
    directory : str
        Location to import pre-processed images from.

    Returns
    -------
    id_labelvec : list of strings
        List of the names of each image (labeled per tile).
    sp_params : ndarray
        The feature vector describing each tile. Shape (n_tiles, n_parameters).

    """
    
    directory_list = [directory]
    
    file_list = []
    for i in range(len(directory_list)):
        file_list.extend(glob.glob(directory_list[i] + "\\*.tif"))
    
    num_import = len(file_list)
    if num_import == 0:
        print('No TIFF images found in directory')
    else:
        print(f'{num_import} images to import \n')
    
    id_labelvec = []
    sp_params = np.zeros((0,2195))
    ct = 0
    for file in file_list:
        
        img = np.asarray(Image.open(file))
        
        tiles = image_tiles(img,256,2,3)
                
        for tilei in tiles:
            tmpparams = spt.texture_analyze(tilei, 5, 4, 7, vector_output=True)
            tmpparams = np.reshape(tmpparams,(1,2195))
            id_name = file.split('\\')[-1][:-4].split('_')[0]
            id_labelvec.append(id_name)
            sp_params = np.concatenate((sp_params, tmpparams), axis=0)
            
            time.sleep(0.1)
            time.sleep(0.5)
            
        time.sleep(0.8)    
        
        ct += 1
        if (ct)%10 == 0:
            time.sleep(6) 
            print(f'Processing {ct/num_import*100:.2f}% Complete')
            
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
    """ 
    Linear Discriminant Anaysis with regularization towards Diagonal
    
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
    
class LinearDimReduce:
    """
    Class for save and applying linear (matrix multiplication) transforms.
    
    
    Parameters
    ----------
    keep_idx : array
        List of features to keep from array. If an array that is larger than
        the transform matrix is attempted to be transformed it will first
        be reduced to the correct size by keeping only the feature numbers
        included in this array.
    scalings : array (n_input_dims x n_output_dims)
        The transformation matrix
    xbar : array
        The average vector. Subtracted before transform, but may be handled 
        outside this class easily or ignored if not important by setting to 0s.
    max_components : int
        The maximum number of dimensions to keep. This will truncate the 
        output of transform() to this size if the scalings matrix 2nd dimension
        is larger than max_components.
    
    
    
    """
    
    def __init__(self, 
                 keep_idx=None, scalings=None, xbar=None, max_components=None):
        self.keep_idx = keep_idx
        self.scalings = scalings
        self.xbar = xbar
        self.max_components = max_components
    
    def clone_lda_rr(self, lda_instance):
        """
        Copy the transform from an instance of LDAReducedRank.

        Parameters
        ----------
        lda_instance : LDAReducedRank
            Class instance to copy from.

        """
        self.scalings = lda_instance.scalings_
        self.xbar = lda_instance.xbar_
        self.max_components = lda_instance._max_components
        
    def load(self, file_name):
        """
        Load the values from a saved instance.
        """
        file = open(file_name, 'rb')
        loaded_inst = pickle.load(file)
        
        self.keep_idx = loaded_inst.keep_idx
        self.scalings = loaded_inst.scalings
        self.xbar = loaded_inst.xbar
        self.max_components = loaded_inst.max_components   
        
    def transform(self, x):
        """
        Transform the input feature vectors, x.

        Parameters
        ----------
        x : array
            An array of feature vectors (n_samples x n_dims).

        Returns
        -------
        array
            Transformed features.

        """
        if not self.keep_idx is None and x.shape[1] != self.xbar.size:
            x = x[:, self.keep_idx]
            
        x_new = np.dot(x - self.xbar, self.scalings)

        return x_new[:, : self.max_components]   
        
    def save(self, file_name):
        """
       Save this instance to a pickle file.

        """
        file = open(file_name, 'wb')
        pickle.dump(self, file)
        
        


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

