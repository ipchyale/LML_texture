import glob

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from texture_tools import standard_img_proc
from texture_tools import _class_means
from texture_tools import LinearDimReduce
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from scipy.spatial.distance import squareform

from skimage.color import rgb2gray


# %% Load feature vectors and corresponding ID labels
allparams = np.load(r'C:\...\SPparams.npy')  
ldr = LinearDimReduce()
ldr.load(r'C:...\ldr01.pickle')
x = ldr.transform(allparams)

tmp = pd.read_csv(r'C:\.\IDvector.csv',
                  header=None)
y = np.squeeze(tmp.to_numpy(dtype=str))

# %% For labeling samples uniquely (6 tiles per image)
def create_new_label_per_n(label_list, n):
    label_set = set(label_list)
    label_new = np.zeros_like(label_list, dtype=int)
    for label in label_set:
        #print(label)
        idx = [i for i, x in enumerate(label_list) if x == label]
        for i in range(len(idx)):
            #label_new.append(i//n)
            label_new[idx[i]] = i//n
            
    return label_new

# For labeling a set of tiles as a unique sample
def create_unique_label_per_n(label_list, n):
    label_set = set(label_list)
    label_new = np.zeros_like(label_list, dtype=int)
    label_old = []
    ct = 1
    for label in label_set:
        #print(label)
        idx = [i for i, x in enumerate(label_list) if x == label]
        for i in range(len(idx)):
            label_new[idx[i]] = 1e4 + (ct*10 + i//n)
            if i%n == 0:
                label_old.append(label)
            
        ct += 1
            
    return label_new, label_old


# %% Generate new labels by image/sample
y_samp = np.array(create_new_label_per_n(y, 6))
y_unique_2, y_unique_orig = create_unique_label_per_n(y, 6)
y2, y_unique_orig = np.array(y_unique_2), np.array(y_unique_orig)


# %% Sample means
n_dim_dist = 100
x_means = _class_means(x[:, :n_dim_dist], y2)
y2_means, y2_inv = np.unique(y2, return_index=True)

y_samp_mean = y_samp[y2_inv]

y_means = y[y2_inv]
unique_ids, y_inv = np.unique(y_means, return_inverse=True)

# %% Imort catalog
catalog = pd.read_csv(r'C:\...\export_12_11_19.csv')
catalog['Catalog Number'] = catalog['Catalog Number'].fillna(value=0).astype(int)
catalog['ID'] = catalog['Catalog Number'].apply(str) \
                + catalog['Secondary Catalog Number'].fillna(value='').str.lower()


# %% Keep metadata relevant to texture
cols = ['ID', 'Year', 'Manufacturer', 'Brand', 'Texture2', 'Reflectance2', 
        'SurfaceDesignation2Definition', 'Finish']
for i in range(y_means.shape[0]):
    
    tmp = catalog[catalog['ID'] == y_means[i].lower()]
    tmp = tmp[0:1].fillna(value='')
    
    if len(tmp) < 1:
        print(i)
        tmp = pd.DataFrame(columns=cols)
        tmp.loc[0,:] = [y_means[i], 0] + ['']*(len(cols)-2)
        #tmp['ID'] = y_means[i] 
    
    if i == 0:
        y_meta = pd.DataFrame(tmp[cols])
    else:
        y_meta = y_meta.append(tmp[cols], ignore_index=True)
        
y_meta = y_meta.astype({'ID': str, 'Year': int, 'Manufacturer': str})

# %% Make a label with just texture labels
y_tmp = y_meta['Texture2'].astype(str).to_numpy()

texture_names, y_tmp_inv = np.unique(y_tmp, return_inverse=True)
y_tmp2 = texture_names[y_tmp_inv[y_tmp_inv > 0]]
texture_names2, y_tmp2_inv = np.unique(y_tmp2, return_inverse=True)
cnt = np.bincount(y_tmp2_inv)

idx_metalabel = np.nonzero(y_tmp_inv > 0)[0]

# %% Train LDA on the class labels
class_priors = np.ones(np.unique(y_tmp[idx_metalabel]).size)
class_priors = class_priors / np.sum(class_priors)
lda_metalabel = LinearDiscriminantAnalysis(store_covariance=True, 
                                           priors=class_priors )
lda_metalabel.fit(x_means[idx_metalabel,:], y_tmp[idx_metalabel])

acc = lda_metalabel.score(x_means[idx_metalabel,:], y_tmp[idx_metalabel])
preds = lda_metalabel.predict(x_means[idx_metalabel,:])
C = confusion_matrix(y_tmp[idx_metalabel], preds, normalize='true')

metalabel_dists = cdist(lda_metalabel.means_, lda_metalabel.means_)

# This loop changes the sorting of the labels from alphabetical to being
# based on the distances between the labels
idx = np.arange(metalabel_dists.shape[0])
for i in range(metalabel_dists.shape[0] - 1):
    if i < 2:
        tmp = np.argmin(metalabel_dists[idx[i],idx[i+1:]])
    elif i < 400:
        tmp = np.sum(metalabel_dists[np.ix_(idx[:i+1],idx[i+1:])], axis=0)
        tmp = np.argmin(tmp) 
    else:
        tmp = np.sum(metalabel_dists[np.ix_(idx[i-4:i+1],idx[i+1:])], axis=0)
        tmp = np.argmin(tmp) 
    #print(metalabel_dists[idx[i],idx[i+1:]])
    #print(i, tmp)
    idx[i+1], idx[tmp+i+1] = idx[tmp+i+1], idx[i+1]
    #print(idx)


# %% Show the confusion matrix of the trained LDA
fig, ax = plt.subplots()
fig.dpi = 150

im = ax.imshow(C[np.ix_(idx,idx)])
plt.colorbar(im, label='True/False Positive Rate')
plt.xticks(np.arange(metalabel_dists.shape[0]), lda_metalabel.classes_[idx],
           rotation=90, fontsize = 4)
plt.yticks(np.arange(metalabel_dists.shape[0]), lda_metalabel.classes_[idx],
           fontsize = 4)
pts = np.arange(C.shape[0]) - 0.5
for i in range(len(pts)):
    ax.plot([pts[i], pts[i]], [pts[0], pts[-1]+1], color='black', linewidth=0.25)
    ax.plot([pts[0], pts[-1]+1], [pts[i], pts[i]], color='black', linewidth=0.25)




# %% Show the distances between the label means
fig, ax = plt.subplots()
fig.dpi = 150

im = ax.imshow(np.log10(metalabel_dists[np.ix_(idx, idx)]))
fig.colorbar(im, label='Log$_{10}$ Distance')
plt.xticks(np.arange(metalabel_dists.shape[0]), lda_metalabel.classes_[idx],
           rotation=90, fontsize = 4)
plt.yticks(np.arange(metalabel_dists.shape[0]), lda_metalabel.classes_[idx],
           fontsize = 4)
#ax.grid(color='black', linestyle='-.', linewidth=.1)
pts = np.arange(metalabel_dists.shape[0]) - 0.5
for i in range(len(pts)):
    ax.plot([pts[i], pts[i]], [pts[0], pts[-1]+1], color='black', linewidth=0.25)
    ax.plot([pts[0], pts[-1]+1], [pts[i], pts[i]], color='black', linewidth=0.25)
    
# %% Predict and plot random samples
for i in range(5):

    idx_sample = np.random.randint(y_means.shape[0])
    print(y_means[idx_sample])
    print(y_samp_mean[idx_sample].astype(int))
    print(y_tmp[idx_sample])
    preds = lda_metalabel.predict(x_means[idx_sample:idx_sample+1,:])
    print(preds)
    
    
    if y_samp_mean[idx_sample].astype(int) == 0:
        directory = r'C:\...\from_camera'
        search_str = '\\' + str(y_means[idx_sample]) + '*'
    else:
        directory = r'C:\...\new_texturescope_images'
        search_str = '\\*\\' + str(y_means[idx_sample]) + '*' \
                      + str(y_samp_mean[idx_sample].astype(int)-1) + '.tif'
         
    print(directory + search_str)
        
    for file in glob.glob(directory + search_str):
            print(file)
            
            I = plt.imread(file)
            img = rgb2gray(I)
            imgout = standard_img_proc(img,3)

            fig, ax = plt.subplots()
            fig.dpi = 150
            
            implot = ax.imshow(imgout, cmap='gray') 
            ax.set_title(y_meta.iloc[idx_sample]['ID'] + '\n' 
                         + y_meta.iloc[idx_sample]['Manufacturer'] + ', '
                         + y_meta.iloc[idx_sample]['Brand'] + '\n' 
                         +  y_meta.iloc[idx_sample]['Texture2'], 
                         fontsize=8)
            plt.gca().set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.xlabel('Predicted texture: ' + preds[0], fontsize=10)
            
            break
     
      
# %% Histogram of predicted labels
preds = lda_metalabel.predict(x_means)
        
names, pred_inv = np.unique(preds, return_inverse=True)
cnt = np.bincount(pred_inv)    

fig, ax = plt.subplots()
fig.dpi = 150

ax.bar(names, cnt, width=0.9)
plt.xticks(rotation=90, fontsize=5)
plt.gca().set_aspect(3/1)
ax.set_yscale('log')
plt.title('Predictions')

            
# %% plot specific sample images
for idx_sample in np.nonzero(y_means == '2446')[0]:
    
    
    if y_samp_mean[idx_sample].astype(int) == 0:
        directory = r'C:\...\from_camera'
        search_str = '\\' + str(y_means[idx_sample]) + '*'
    else:
        directory = r'C:\...\new_texturescope_images'
        search_str = '\\*\\' + str(y_means[idx_sample]) + '*' \
                      + str(y_samp_mean[idx_sample].astype(int)-1) + '.tif'
         
    print(directory + search_str)
        
    for file in glob.glob(directory + search_str):
            print(file)
            
            I = plt.imread(file)
            img = rgb2gray(I)
            imgout = standard_img_proc(img,3)
    
            fig, ax = plt.subplots()
            fig.dpi = 150
            
            implot = ax.imshow(imgout, cmap='gray') 
            ax.set_title(y_meta.iloc[idx_sample]['ID'] + '\n' 
                         + y_meta.iloc[idx_sample]['Manufacturer'] + ', '
                         + y_meta.iloc[idx_sample]['Brand'] + '\n' 
                         +  y_meta.iloc[idx_sample]['Texture2'], 
                         fontsize=8)
            plt.gca().set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            
            
            break
        