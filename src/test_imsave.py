
import numpy as np
import sys
import skimage
import imageio
import scipy
import matplotlib.pyplot as plt
import PIL

from timeit import timeit

img = (np.random.rand(1024,1024,4)*255).astype(np.uint8)

@profile
def run():
	skimage.io.imsave('test-skimage.png',img)
	skimage.io.imsave('test-skimage-no_contrast_check.png',img, check_contrast=False)
	imageio.imwrite('test-imageio.png', img)
	plt.imsave('test-matplot.png',img)
	PIL.Image.fromarray(img).save('test-pillow.png')

for _ in range(10): run()

#!diff test-imageio.png test-skimage.png

