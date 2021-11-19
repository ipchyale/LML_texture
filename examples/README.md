## Viewing the bands of the steerable pyramid decomposition of an image (steerable_pyramid.py)

Steerable pyramid (SP) is a mutli-scale, oriented decomposition of the images. The image can be divided into frequency bands and orientation-specific subbands. This allows it to also be used as a filter for these qualities by reconstructing the image from the SP coefficients of each subband.

The first block uses the feature_vector_names function to output a list of names for the features that would be the output of the texture_analyze function called with the same input Nsc, Nor, Na arguments.

In the example a grayscale tiff is opened and decomposed into 5 levels and 4 orientations, the typical values used for feature selection. The end result is that there will be 5 + 2 = 7 levels and each of the 5 levels will have 4 oriented subbands. The additional 2 levels are the residual high and low pass above the highest level and below the lowest level. The residual highpass band will always be at a frequency on the order of the sampling frequency, but the residual lowpass may include multiple frequency octaves because the number of levels is an input parameter and can be less than the max levels that would fully decompose the image. In this example more than 5 levels could have been used, so it will be the case that the resdiual lowpass will in fact include what would have been levels 6, 7, etc.

To show the different bands it is more intuitive to reconstruct the image from a subset of the bands than to look at their coefficients. The reconstructed images will be the same size as the original which is easier to compare as the levels get downsmapled by a factor of 2 at each iteration.

In this example groups of bands are isolated and the image is reconstructed based on only their coefficients:

###### Levels (High = 4, 5, Low = 1, 2 ,3)

![image](https://user-images.githubusercontent.com/9450221/142087468-b9d0f141-fe39-45e2-a622-fcdea95ca2c6.png)

![image](https://user-images.githubusercontent.com/9450221/142087409-31c922af-4002-45c6-80b2-c5c49c475ffa.png)

###### Orientation (within level 3)

![image](https://user-images.githubusercontent.com/9450221/142087485-0926c96a-ae52-4f5c-9611-f7a77565bca3.png)

![image](https://user-images.githubusercontent.com/9450221/142087445-11822942-e968-4379-aa4e-ba03e046811f.png)

While these bands provide a detailed decomposition by frequency and orientation of the image, for use as texture descriptors we extract parameters that act as statistical summaries of the subbands and relationships between them with the goal being to avoid the texture description being senstitive to the particularities of a single image.


## Reducing the number of features (dim_red.py)

The default parameters used in the SP decomposition return 2,195 features, 1,712 of which contain variation and are useful. For many tasks it is better to work with a smaller set of features, and due to the features being adjacent cross-correlation matrix elements there is a degree of correlation built into the feature set that can be removed with linear combinations of the features. 

One way to choose linear combinations of the features is to find the ones that are most informative in terms of separating some kind of classes among the data. The only truly reliable classes we have are the sample ID's themselves because they represent different instances of the same texture. In this example the dimensions are reduced by finding the 100 most informative linear combinations of the original feature space in terms of separating the different classes from each other based on an assumption of normality of their feature distributions. It will return 100 features, but they are rank ordered so by only using the first *n* dimensions it will give the most informative *n*-dimensional feature vectors.

## Comparing two different sets of images (compare_images_by_folder.py)

This examples compares two sets of images with a high degree of overlap between the ID's of the samples contained in each. The goal is to compare how similar the samples from the 2nd (new) folder that has repeated measurements of each ID with the first (old) folder that only contains one image per ID.

It starts with the SP features already being extracted and their corresponding ID labels for each of the tiles for each image. The first step is to reduce the features down to 100 by using LDA to choose the most discriminative features. If the original features are *x*, we call the 100-dimensional discriminant coordinates *x2*. We can further transform into the vectors such that the LDA within-class covariance matrix is the identity matrix by repeating the LDA step with gamma=0. Using gamma > 0 in the first step is the reason the covariance is not identity after the first step.

Labels have to be created to distinguish not just what ID each tile has, but whether it is the 1st, 2nd, 3rd, etc. instance of that ID. Those labels can be used to compare different samples with the same ID to each other.

The first plot shows the distributions of distances between groups of images: between all images, between new images with the same ID, and between old images and new images with the same ID.

![image](https://user-images.githubusercontent.com/9450221/142299852-307a4c69-6892-4829-a541-0d6ab431a951.png)

Next stats of the distances between tiles from the same sample and between samples with the same ID will be compared to find how much the texture descriptors vary sample-to-sample. The basic stats of the between and within sample distributions are printed out and shown in histogram form

![image](https://user-images.githubusercontent.com/9450221/142300049-002cbadf-7ef1-4a5a-a18e-411891be7041.png)

In this data we can see there is some divergence between the two folders.

