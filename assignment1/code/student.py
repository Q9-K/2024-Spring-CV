# Spring 2021: Image Processing and Computer Vision
# Beihang Univeristy
# Homework set 1
# Lu Sheng (lsheng@buaa.edu.cn)
#
# Implement my_imfilter() and gen_hybrid_image()
#
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech and
# by Donsuk Lee (donlee90@stanford.edu) for CS131 @ Stanford
import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale


def correlate_fast(image, kernel):
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros(image.shape)
    padding_height = Hk//2
    padding_width = Wk//2
    image = np.pad(image, ((padding_height, padding_height),
                   (padding_width, padding_width)))
    for m in range(Hi):
        for n in range(Wi):
            out[m, n] = np.sum(kernel*image[m:m+Hk, n:n+Wk])
    return out


def cross_correlation(f, g):
    """ Cross-correlation of image f and template g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    # YOUR CODE HERE
    out = correlate_fast(f, g)
    # END YOUR CODE

    return out


def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of image f and template g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    # YOUR CODE HERE
    g_mean = g - np.mean(g)
    out = cross_correlation(f, g_mean)
    # END YOUR CODE

    return out


def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of image f and template g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    # YOUR CODE HERE
    g = (g - np.mean(g)) / np.std(g)

    Hi, Wi = f.shape
    Hk, Wk = g.shape
    out = np.zeros(f.shape)
    padding_height = Hk//2
    padding_width = Wk//2
    pad_f = np.pad(f, ((padding_height, padding_height),
                   (padding_width, padding_width)))

    for m in range(Hi):
        for n in range(Wi):
            sub_f = pad_f[m:m+Hk, n:n+Wk]
            sub_f_mean = np.mean(sub_f)
            sub_f_std = np.std(sub_f)
            out[m, n] = np.sum(g * (sub_f - sub_f_mean)/sub_f_std)
    # END YOUR CODE

    return out


def my_imfilter(image, kernel):
    """
    Your function should meet the requirements laid out on the project webpage.
    Apply a filter (using kernel) to an image. Return the filtered image. To
    achieve acceptable runtimes, you MUST use numpy multiplication and summation
    when applying the kernel.
    Inputs
    - image: numpy nd-array of dim (m,n) or (m, n, c)
    - kernel: numpy nd-array of dim (k, l)
    Returns
    - filtered_image: numpy nd-array of dim of equal 2D size (m,n) or 3D size (m, n, c)
    Errors if:
    - filter/kernel has any even dimension -> raise an Exception with a suitable error message.
    """
    filtered_image = np.zeros(image.shape)

    ##################
    # Your code here #
    # print('my_imfilter function in student.py needs to be implemented')

    # Pad the input image with zeros.
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape

# (i) (4 pts) Pad the input image with zeros.
# (ii) (4 pts) Support grayscale and color images. Note that grayscale images will be 2D
# numpy arrays.
# (iii) (4 pts) Support arbitrary shaped odd-dimension filters (e.g., 7x9 filters but not 4x5
# filters).
# (iv) (4 pts) Raise an Exception with an error message for even filters, as their output is
# undefined.
# (v) (4 pts) Return a filtered image which is the same resolution as the input image.

    if Hk % 2 == 0 or Wk % 2 == 0:
        raise Exception('卷积核边为偶数!')

    padding_height = Hk//2
    padding_width = Wk//2
    pad_image = np.pad(image, ((padding_height, padding_height),
                               (padding_width, padding_width)))
    filtered_image = cross_correlation(pad_image, kernel)
    ##################

    return filtered_image


"""
EXTRA CREDIT placeholder function
"""


def my_imfilter_fft(image, kernel):
    """
    Your function should meet the requirements laid out in the extra credit section on
    the project webpage. Apply a filter (using kernel) to an image. Return the filtered image.
    Inputs
    - image: numpy nd-array of dim (m,n) or (m, n, c)
    - kernel: numpy nd-array of dim (k, l)
    Returns
    - filtered_image: numpy nd-array of dim of equal 2D size (m,n) or 3D size (m, n, c)
    Errors if:
    - filter/kernel has any even dimension -> raise an Exception with a suitable error message.
    """
    filtered_image = np.zeros(image.shape)

    ##################
    # Your code here #
    print('my_imfilter_fft function in student.py is not implemented')
    ##################

    return filtered_image


def gen_hybrid_image(image1, image2, cutoff_frequency):
    """
     Inputs:
     - image1 -> The image from which to take the low frequencies.
     - image2 -> The image from which to take the high frequencies.
     - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                           blur that will remove high frequencies.

     Task:
     - Use my_imfilter to create 'low_frequencies' and 'high_frequencies'.
     - Combine them to create 'hybrid_image'.
    """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]

    # Steps:
    # (1) Remove the high frequencies from image1 by blurring it. The amount of
    #     blur that works best will vary with different image pairs
    # generate a 1x(2k+1) gaussian kernel with mean=0 and sigma = s, see https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    s, k = cutoff_frequency, cutoff_frequency*2
    probs = np.asarray([exp(-z*z/(2*s*s))/sqrt(2*pi*s*s)
                       for z in range(-k, k+1)], dtype=np.float32)
    kernel = np.outer(probs, probs)

    # Your code here:
    # Replace with your implementation
    low_frequencies = np.zeros(image1.shape)

    # (2) Remove the low frequencies from image2. The easiest way to do this is to
    #     subtract a blurred version of image2 from the original version of image2.
    #     This will give you an image centered at zero with negative values.
    # Your code here #
    # Replace with your implementation
    high_frequencies = np.zeros(image1.shape)

    # (3) Combine the high frequencies and low frequencies
    # Your code here #
    hybrid_image = np.zeros(image1.shape)  # Replace with your implementation

    # (4) At this point, you need to be aware that values larger than 1.0
    # or less than 0.0 may cause issues in the functions in Python for saving
    # images to disk. These are called in proj1_part2 after the call to
    # gen_hybrid_image().
    # One option is to clip (also called clamp) all values below 0.0 to 0.0,
    # and all values larger than 1.0 to 1.0.

    return low_frequencies, high_frequencies, hybrid_image
