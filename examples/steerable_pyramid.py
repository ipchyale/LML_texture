from PIL import Image
import matplotlib.pyplot as plt

from steerablepyrtexture.pyramids import SteerablePyramidFreq

Nsc = 5
Nor = 4

file = r'C:\...\image02.tif'
img = np.asarray(Image.open(file))

pyr0 = SteerablePyramidFreq(img, height=Nsc, order=Nor-1, is_complex=True)
im0 = pyr0.recon_pyr(levels=[0, 1, 2], bands=[0, 1, 2, 3])
imL = pyr0.recon_pyr(levels=[3, 4], bands=[0, 1, 2, 3])
# Use these to see the highest and lowest frequency residual components
#imL = pyr0.recon_pyr(levels=['residual_lowpass'], bands='all')
#im0 = pyr0.recon_pyr(levels=['residual_highpass'], bands='all')

fig, ax = plt.subplots(1,3)
fig.dpi = 300
ax[0].imshow(img, cmap='gray')
ax[0].set_xlabel('Original Image', fontsize=5)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].imshow(im0, cmap='gray')
ax[1].set_xlabel('Reconstructed Image (Highpass)', fontsize=5)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[2].imshow(imL, cmap='gray')
ax[2].set_xlabel('Reconstructed Image (Lowpass)', fontsize=5)
ax[2].set_xticks([])
ax[2].set_yticks([])

im1 = pyr0.recon_pyr(levels=[3], bands=[0])
im2 = pyr0.recon_pyr(levels=[3], bands=[2, 3])
fig, ax = plt.subplots(1,3)
fig.dpi = 300
ax[0].imshow(img, cmap='gray')
ax[0].set_xlabel('Original Image', fontsize=5)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].imshow(im1, cmap='gray')
ax[1].set_xlabel('Reconstructed Image (Orientation 1)', fontsize=5)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[2].imshow(im2, cmap='gray')
ax[2].set_xlabel('Reconstructed Image (Orientations 3 and 4)', fontsize=5)
ax[2].set_xticks([])
ax[2].set_yticks([])