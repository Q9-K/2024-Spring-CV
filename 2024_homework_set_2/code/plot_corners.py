"""
This code adapted from scikit-image documentation
http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_corner.html

Edited by Lu Sheng, Beihang University (lsheng@buaa.edu.cn)
"""

"""
================
Corner detection
================

Detect corner points using the Harris corner detector and determine the
subpixel position of corners ([1]_, [2]_).

.. [1] https://en.wikipedia.org/wiki/Corner_detection
.. [2] https://en.wikipedia.org/wiki/Interest_point_detection

"""


# load in different images to see where the Harris Corner Detector finds corners
from matplotlib import pyplot as plt
from skimage import io, color, transform
from skimage.feature import corner_harris, peak_local_max

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
file_name = 'Chase1'
file_path = os.path.join(current_dir, '../data/matching/'+file_name+'.jpg')

image = transform.rescale(color.rgb2gray(io.imread(file_path)), 0.25)

harris_response = corner_harris(image)
# Note: Feel free to play with these parameters to investigate their effects
coords = peak_local_max(harris_response, min_distance=5, threshold_rel=0.05)

plt.imshow(image, cmap=plt.cm.gray)
plt.plot(coords[:, 1], coords[:, 0], '+r', markersize=15)
plt.axis((-100, image.shape[1]+100, image.shape[0]+100, -100))

# 保存结果
plt.savefig('../data/matching/'+file_name+'_corner.jpg')
plt.show()
