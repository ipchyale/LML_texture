import glob 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from texture_tools import LinearDimReduce
from texture_tools import _class_means
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn import cluster

# %% Generate feature space to cluster on
# %% Load feature vectors and corresponding ID labels
allparams = np.load(r'C:\...\SPparams.npy')  
ldr = LinearDimReduce()
ldr.load(r'C:...\ldr01.pickle')
x = ldr.transform(allparams)

tmp = pd.read_csv(r'C:\...\IDvector.csv',
                  header=None)
y = np.squeeze(tmp.to_numpy(dtype=str))


# %% Create labels to identify which image each tile is from
y_persample = np.arange(y.size)//6

tmp, tmp1 = np.unique(y_persample, return_index=True)
Y = y[tmp1]


# %% Sample means
n_dim_keep = 25
class_means = _class_means(x[:,:n_dim_keep], y)


scaler = StandardScaler()
clust_features = scaler.fit_transform(class_means)

# %% Aggl. clustering with Ward linkage
ward_clust = cluster.AgglomerativeClustering(n_clusters=None, distance_threshold=7)

ward_clust.fit(clust_features)
print(ward_clust.n_clusters_)

cluster_labels = ward_clust.labels_
cluster_ids, cluster_inv, cluster_cts = np.unique(cluster_labels, 
                                                  return_inverse=True, 
                                                  return_counts=True)

# %% Histogram of the cluster sizes
plt.figure()
plt.hist(cluster_labels, ward_clust.n_clusters_)




# %% Generate file location list (in order) for importing images to plot
directory = r'C:\Users\ngr7\Documents\Data\derivedData\whole_collection\output_images_new2'  

file_list = []
file_list = glob.glob(directory + "\\*.tif")

ids = np.unique(y)

# %% Sort the clusters in descending order and plot
idx_cluster_size_sort = np.argsort(cluster_cts)[-1:0:-1]

idx_cluster_size_sort = idx_cluster_size_sort[:20]
for i_cl in idx_cluster_size_sort:
    plot_list = []
    
    tmp = np.nonzero(cluster_labels == cluster_ids[i_cl])[0]
    if len(tmp) > 2*5**2:
        tmp = np.random.choice(tmp, 2*5**2)
        
    for j in tmp:
        idx = np.nonzero(Y == ids[j])[0]
        plot_list.append(idx[0])
    
    
    if np.sqrt(len(plot_list)/2)%1 != 0:
       n_tmp = 2*np.ceil(np.sqrt(len(plot_list)/2))**2
       n_tmp = int(n_tmp)
       for j in np.random.permutation(tmp):
           idx = np.nonzero(Y == ids[j])[0]
           plot_list.append(idx[1])
           if len(plot_list) == n_tmp:
               break
        
    
    img_list = []
    for i in plot_list:
        img_list.append(np.asarray(Image.open(file_list[i])))
        
        
    num_plot = len(plot_list)
    n_y = np.ceil(np.sqrt(num_plot/2)).astype(int)
    n_x = 2*n_y
        
    
    fig = plt.figure()
    fig.dpi = 800
    ax = []
    
    gs_factor = 5
    gs = fig.add_gridspec(gs_factor*n_y+1, gs_factor*n_x)
    
    ax.append(fig.add_subplot(gs[0, np.floor(gs_factor*n_x/2).astype(int)]))
    ax[-1].set_title(f'Cluster {i_cl}')
    ax[-1].axis('off')
    for i in range(num_plot):
        j = gs_factor*(i//n_x) + 1
        k = gs_factor*(i%n_x)
           
        ax.append(fig.add_subplot(gs[j:j+gs_factor,k:k+gs_factor]))
        ax[-1].imshow(img_list[i][:int(640/n_y),:int(640/n_y)], cmap='gray', vmin=0, vmax=1)
        
        ax[-1].set_xticks([])
        ax[-1].set_yticks([])
               








# %% Generate alternate (cluster-based) labels
y_clust = np.zeros_like(y, dtype='int')
for i in range(len(lda2.classes_)):
    idx = np.where(y == lda2.classes_[i])
    y_clust[idx] = int(ward_clust.labels_[i])
    