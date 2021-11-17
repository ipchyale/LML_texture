import numpy as np

from texture_tools import LDAReducedRank
from texture_tools import LinearDimReduce

# %% Use SP parameters as features and sample ID's as labels

y = id_labels

tmp = np.std(sp_params, axis=0)
keep_idx = np.where(tmp > 1e-20)[0]

x = sp_params[:, keep_idx]

# %% Find linear transform based on LDA separability to reduce to 100 or less
#### dimensions using the sample ID's as class labels.
lda_dimred = LDAReducedRank(gamma=0.5, n_components=100)
lda_dimred.fit(x, y)

# %% New data can be transformed either with the LDAReducedRank class or with
#### the LinearDimReduce class transform methods. They are equivalent, but the
#### latter is lighter and is used for saving the transform.
ldr = LinearDimReduce(keep_idx = keeplist)
ldr.clone_lda_rr(lda_dimred)

x2 = lda_dimred.transform(x)
x2 = ldr.transfrom(x)

# %% How to save and load the transformation matrix using LinearDimReduce
ldr.save(r'C:\...\ldr01.pickle')

ldr_2 = LinearDimReduce()
ldr_2.load(r'C:\...\ldr01.pickle')

