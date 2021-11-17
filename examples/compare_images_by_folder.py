import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, cdist
from scipy.spatial.distance import squareform
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from texture_tools import LDAReducedRank
from texture_tools import LinearDimReduce
from texture_tools import _class_means, _class_cov



# %% Define functions for creating labels for tiles with the same ID
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

# %%  Import the first folder's data (Old)
allparams = np.load(r'C:\...\SPparams_old.npy')  
tmp = np.std(allparams, axis=0)
tmp = np.where(tmp > 1e-20)
keeplist = tmp[0]
x_old = allparams[:,keeplist]

tmp = pd.read_csv(r'C:\...\IDvector_old.csv',
                  header=None)
y_old = np.squeeze(tmp.to_numpy(dtype=str))

#  Import new image data
allparams = np.load(r'C:\...\SPparams_new.npy')  
x_new = allparams[:,keeplist]

tmp = pd.read_csv(r'C:\...\IDvector_new.csv',
                  header=None)
y_new = np.squeeze(tmp.to_numpy(dtype=str))

y_new_sample = np.array(create_new_label_per_n(y_new, 6))
y_unique_2, y_unique_orig = create_unique_label_per_n(y_new, 6)
y_unique_2, y_unique_orig = np.array(y_unique_2), np.array(y_unique_orig)
y_binder = np.zeros_like(y_new, dtype=int)

# Stitch the old and new data together
x = np.concatenate((x_old, x_new), axis=0)
y = np.concatenate((y_old, y_new), axis=0)

y_samp = np.concatenate(( np.zeros((np.shape(y_old))), y_new_sample+1), axis=0)



# %% Reduce the number of dimensions based on the labels down to 100

lda_dimred = LDAReducedRank(gamma=0.5, n_components=100)
lda_dimred.fit(x, y)

x2 = lda_dimred.transform(x)

# %% Transform the dimensions again to sphere the data based on the LDA 
#### within-class covariance
lda_sph = LDAReducedRank()
lda_sph.fit(x2, y)

x3 = lda_sph.transform(x2)

# Put the combined transform into an instance of LinearDimReduce
tmp = lda_dimred.scalings_[:, :lda_dimred.n_components] @ lda_sph.scalings_
ldr2 = LinearDimReduce(keep_idx = keeplist,
                       scalings = tmp,
                       xbar = lda_dimred.xbar_)

# = ldr2.transform(x)

# %% Group the labels by sample
y2 = np.concatenate( (y_old, y_unique_2) )
y_samp = np.concatenate(( np.zeros((np.shape(y_old))), y_new_sample+1), axis=0)

y2_means, y2_inv = np.unique(y2, return_index=True)
y_samp_mean = y_samp[y2_inv]
y_means = y[y2_inv]

unique_ids, y_inv = np.unique(y_means, return_inverse=True)

# %% Calculate distances by sample ID/pocket and old vs. new
n_dim_dist = 100
x_means = _class_means(x3[:, :n_dim_dist], y2)
dist_all = squareform(pdist(x_means))

dist_pocket = np.zeros((0,))
dist_old = np.zeros((0,))
for i in range(len(unique_ids)):
    idx = np.squeeze(np.argwhere(y_inv == i))
    idx_pocket = idx[y_samp_mean[idx] > 0]
    idx_old = idx[y_samp_mean[idx] == 0]
    #print(idx, idx_pocket)
    
    if idx_pocket.size > 0:
        tmp = squareform(dist_all[np.ix_(idx_pocket, idx_pocket)])
        
        dist_pocket = np.concatenate( (dist_pocket, tmp) )
    
    if idx_old.size > 0:
        dist_old = np.concatenate( (dist_old, 
                        dist_all[idx_old, idx_pocket]) )
    


# Plot the comparison of the distance histograms
fig, ax = plt.subplots()
fig.dpi = 250
tmp, tmp1 = np.histogram(squareform(dist_all), 
                         bins=np.linspace(0,50,301), density=False)
ax.bar(tmp1[:-1], tmp/tmp.max(), width=tmp1[1]-tmp1[0])

tmp, tmp1 = np.histogram(dist_pocket, tmp1, density=False)
ax.bar(tmp1[:-1], -1*tmp/tmp.max(), width=tmp1[1]-tmp1[0])

tmp, tmp1 = np.histogram(dist_old, tmp1, density=False)
ax.bar(tmp1[:-1], -1*tmp/tmp.max(), width=tmp1[1]-tmp1[0], alpha=0.4)

plt.xlim(tmp1.min()-10, 1.05*tmp1.max())
ax.set_yticks([])
plt.xlabel('Distance (Mahalanobis)')
plt.title(f'Within Pocket Texture Comparison ({n_dim_dist}-Dimensional)')

plt.legend(['All Papers', 'Within Pocket', 'Older Images'])






# %% Calculate distances by sample ID, both within sample and between samples
#### with the same ID.
y_means2, y_inv2 = np.unique(y, return_inverse=True)

dist_wtn_sample = np.zeros_like(y2_means, dtype=float)
dist_wtn_outlierval = np.zeros_like(y2_means, dtype=float)
dist_btn_sample = np.zeros_like(y2_means, dtype=object)
test = np.zeros_like(y2_means, dtype=int) - 1
ct = 0
for j in range(y_inv.max()+1):
    idx = np.nonzero(y_inv2 == j)[0]
    
    qda = QuadraticDiscriminantAnalysis(store_covariance=True)
    
    x_tmp = x3[idx, :]
    y_tmp = y_samp[idx]
    #qda.fit(x_tmp, y_tmp)
    
    qda.classes_ = np.unique(y_tmp)
    qda.means_ = _class_means(x_tmp, y_tmp)
     
    if x_tmp.shape[0] / qda.classes_.size != 6:
        print(j)
        print(y_tmp)
        print(y[idx])
    
    for i in range(qda.classes_.size):
        
        ct +=1
        class_idx = np.nonzero(y_tmp == qda.classes_[i])[0]   
        idx_tmp = int(np.nonzero(y2_means == y2[idx[class_idx[0]]])[0])  

        tmp = cdist(qda.means_[i:i+1,:], x_tmp)

        dist_wtn_sample[idx_tmp] = np.average(tmp[0, class_idx])
        dist_wtn_outlierval[idx_tmp] = np.max(tmp[0, class_idx]) \
                                        / dist_wtn_sample[idx_tmp]
        
        tmp = cdist(qda.means_[i:i+1,:], qda.means_) / dist_wtn_sample[idx_tmp]
        
        tmp2 = np.delete(tmp, i)
        
        
        if i==0:
            dist_btn_sample[idx_tmp] = tmp2
            test[idx_tmp] = 0
        else:
            dist_btn_sample[idx_tmp] = tmp2[1:]
            test[idx_tmp] = 1
        
        
    


# %%
fig, ax = plt.subplots()
fig.dpi = 150

ax.hist(dist_wtn_sample, 300)
plt.title('Within-sample distances')

# %%
print('Distance metrics')
print(f'Within-sample distance: {np.median(dist_wtn_sample):.2f}'
      f' ({np.percentile(dist_wtn_sample, 16):.2f}, '
      f'{np.percentile(dist_wtn_sample, 84):.2f})')
print(f'Within-pocket distance: {np.median(dist_pocket):.2f}'
      f' ({np.percentile(dist_pocket, 16):.2f}, '
      f'{np.percentile(dist_pocket, 84):.2f})')

#bootval = np.zeros((int(1e4),))
#for i in range(bootval.size):
#    tmp = np.random.choice(dist_wtn_sample, dist_wtn_sample.size, replace=True)
#    bootval[i] = np.median(tmp)


# %%
fig, ax = plt.subplots()
fig.dpi = 150

ax.hist(dist_wtn_outlierval, 200)
plt.title('Within-sample outliers')

# %%
idx = np.argsort(dist_wtn_outlierval)

tmp = int(-45)
print(dist_wtn_sample[idx[tmp]])
print(dist_wtn_outlierval[idx[tmp]])
print(y_means[idx[tmp]])
print(y_samp_mean[idx[tmp]])

# %%
fig, ax = plt.subplots()
fig.dpi = 150

ax.plot(dist_wtn_sample, np.multiply(dist_wtn_outlierval, 1)
        , 'x', markersize=0.8)
plt.title('Within-sample consistency')
plt.xlabel('Within-sample Average Distance')
plt.ylabel('Within-sample max/mean')

# %% Standard Histogram of ratio
fig, ax = plt.subplots()
fig.dpi = 150

idx = np.nonzero(y_samp_mean == 0)[0]
all_old_btn = np.zeros((0,))
for i in range(len(idx)):
    all_old_btn = np.append(all_old_btn, dist_btn_sample[idx[i]])

#ax.hist()
tmp, tmp1 = np.histogram(np.log2(all_old_btn), 120, density=False)
ax.bar(tmp1[:-1], tmp/tmp.max(), width=tmp1[1]-tmp1[0])


idx = np.nonzero(y_samp_mean > 0)[0]
all_new_btn = np.zeros((0,))
for i in range(len(idx)):
    all_new_btn = np.append(all_new_btn, dist_btn_sample[idx[i]])

tmp, tmp1 = np.histogram(np.log2(all_new_btn), tmp1, density=False)
ax.bar(tmp1[:-1], -tmp/tmp.max(), width=tmp1[1]-tmp1[0])

plt.xticks([0, 1, 2, 3, 4, 5, 6], ['1', '2', '4', '8', '16', '32', '64'])
plt.yticks([])
ax.legend(['Old Samples', 'New Samples'])
plt.title('Variation within images compared to between samples')
plt.xlabel('Ratio (between/within)')
plt.ylabel('Normalized Histogram')

# %% Cumulative Histogram of ratio
fig, ax = plt.subplots()
fig.dpi = 150

tmp, tmp1 = np.histogram(np.log2(all_new_btn), tmp1, density=True)
tmp = np.cumsum(tmp)
ax.bar(tmp1[:-1], tmp/tmp[-1], width=tmp1[1]-tmp1[0], color=u'#ff7f0e')
#ax.plot(tmp1[:-1], tmp/tmp[-1])

tmp, tmp1 = np.histogram(np.log2(all_old_btn), 300, density=True)
tmp = np.cumsum(tmp)
ax.bar(tmp1[:-1], tmp/tmp[-1], width=tmp1[1]-tmp1[0])
#ax.plot(tmp1[:-1], tmp/tmp[-1])

plt.xticks([0, 1, 2, 3, 4, 5], ['1', '2', '4', '8', '16', '32'])
plt.grid(color='k', linestyle='-', linewidth=.2)
ax.legend(['New Samples', 'Old Samples'])
plt.title('Variation within images compared to between samples')
plt.xlabel('Ratio (between/within)')
plt.ylabel('Cumulative Histogram')




