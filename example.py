import numpy as np
import Image
import matplotlib.pyplot as plt
import slic

im = np.array(Image.open("grass.jpg"))
image_argb = np.dstack([im[:, :, :1], im]).copy("C")
region_labels = slic.slic_n(image_argb, 1000, 10)
slic.contours(image_argb, region_labels, 10)
plt.imshow(image_argb[:, :, 1:].copy())
plt.show()
