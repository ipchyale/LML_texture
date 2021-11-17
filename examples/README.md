# Viewing the bands of the steerable pyramid decomposition of an image (steerable_pyramid.py)

Steerable pyramid (SP) is a mutli-scale, oriented decomposition of the images. The image can be divided into frequency bands and orientation-specific subbands. This allows it to also be used as a filter for these qualities by reconstructing the image from the SP coefficients of each subband.

In the example a grayscale tiff is opened and decomposed into 5 levels and 4 orientations, the typical values used for feature selection. The end result is that there will be 5 + 2 = 7 levels and each of the 5 levels will have 4 oriented subbands. The additional 2 levels are the residual high and low pass above the highest level and below the lowest level. The residual highpass band will always be at a frequency on the order of the sampling frequency, but the residual lowpass may include multiple frequency octaves because the number of levels is an input parameter and can be less than the max levels that would fully decompose the image. In this example more than 5 levels could have been used, so it will be the case that the resdiual lowpass will in fact include what would have been levels 6, 7, etc.

To show the different bands it is more intuitive to reconstruct the image from a subset of the bands than to look at their coefficients. The reconstructed images will be the same size as the original which is easier to compare as the levels get downsmapled by a factor of 2 at each iteration.

In the example groups of bands are isolated (high/low)

![image](https://user-images.githubusercontent.com/9450221/142087027-cd8b5a85-13f9-4dfb-8290-417149a58b5d.png)

and by orientation (within leve 4)

![image](https://user-images.githubusercontent.com/9450221/142087084-3c8ab873-5b50-48a8-b21f-42a87214343e.png)

While these bands provide a detailed decomposition by frequency and orientation of the image, for use as texture descriptors we extract parameters that act as statistical summaries of the subbands and relationships between them with the goal being to avoid the texture description being senstitive to the particularities of a single image.
