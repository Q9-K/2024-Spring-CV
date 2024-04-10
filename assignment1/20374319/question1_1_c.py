'''
Date: 2024-04-10 00:57:38
Author: Q9K
Description: 1.1.c
'''

from scipy.ndimage import convolve, correlate
from skimage import io, img_as_ubyte
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)

kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
image = io.imread(os.path.join(current_dir, 'img/image.jpg'), as_gray=True)
io.imsave(os.path.join(current_dir, 'img/image_gray.jpg'), image.copy())


convolve_res = convolve(image, kernel)
correlate_res = correlate(image, kernel)

io.imsave(os.path.join(current_dir, 'img/convolve_res.jpg'), convolve_res.copy())
io.imsave(os.path.join(current_dir, 'img/correlate_res.jpg'), correlate_res.copy())

difference = convolve_res + correlate_res
io.imsave(os.path.join(current_dir, 'img/difference.bmp'), img_as_ubyte(difference))
