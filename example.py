import numpy as np
import matplotlib.pyplot as plt
from skimage.data import lena
import slic

im = lena()
lena_argb = np.dstack([im[:, :, :1], im]).copy("C")
region_labels = slic.slic_n(lena_argb, 1000, 10)
slic.contours(lena_argb, region_labels, 10)
plt.imshow(lena_argb[:, :, 1:])
plt.show()
