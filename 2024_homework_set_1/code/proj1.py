from student import cross_correlation
from student import zero_mean_cross_correlation
from student import normalized_cross_correlation

from helpers import save_image

import os
from skimage import io
import numpy as np
from matplotlib import pyplot as plt


resultsDir = '..' + os.sep + 'results'
if not os.path.exists(resultsDir):
    os.mkdir(resultsDir)


def naive_cross_correlation(img, img_gray, temp, temp_gray):
    # Perform cross-correlation between the image and the template
    out = cross_correlation(img_gray, temp_gray)
    save_image(f'{resultsDir}/naive_out.jpg', out)

    # Find the location with maximum similarity
    y,x = (np.unravel_index(out.argmax(), out.shape))

    # Display product template
    plt.figure(figsize=(25,20))
    plt.subplot(3, 1, 1)
    plt.imshow(temp)
    plt.title('Template')
    plt.axis('off')

    # Display cross-correlation output
    plt.subplot(3, 1, 2)
    plt.imshow(out)
    plt.title('Cross-correlation (white means more correlated)')
    plt.axis('off')

    # Display image
    plt.subplot(3, 1, 3)
    plt.imshow(img)
    plt.title('Result (blue marker on the detected location)')
    plt.axis('off')

    # Draw marker at detected location
    plt.plot(x, y, 'bx', ms=40, mew=10)
    plt.show()



def simple_zero_mean_cross_correlation(img, img_gray, temp, temp_gray):
    # Perform cross-correlation between the image and the template
    out = zero_mean_cross_correlation(img_gray, temp_gray)
    save_image(f'{resultsDir}/simple_out.jpg', out)

    # Find the location with maximum similarity
    y,x = np.unravel_index(out.argmax(), out.shape)

    # Display product template
    plt.figure(figsize=(30,20))
    plt.subplot(3, 1, 1)
    plt.imshow(temp)
    plt.title('Template')
    plt.axis('off')

    # Display cross-correlation output
    plt.subplot(3, 1, 2)
    plt.imshow(out)
    plt.title('Cross-correlation (white means more correlated)')
    plt.axis('off')

    # Display image
    plt.subplot(3, 1, 3)
    plt.imshow(img)
    plt.title('Result (blue marker on the detected location)')
    plt.axis('off')

    # Draw marker at detected location
    plt.plot(x, y, 'bx', ms=40, mew=10)
    plt.show()

def check_product_on_shelf(shelf, product, threshold=0.025):
    out = zero_mean_cross_correlation(shelf, product)
    
    # Scale output by the size of the template
    out = out / float(product.shape[0]*product.shape[1])
    
    # Threshold output
    out = out > threshold
    
    if np.sum(out) > 0:
        print('The product is on the shelf')
    else:
        print('The product is not on the shelf')

def you_can_check_product_on_shelf(img, img_gray, img2, img2_gray, temp, temp_gray, threshold=0.025):
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    check_product_on_shelf(img_gray, temp_gray, threshold)

    plt.imshow(img2)
    plt.axis('off')
    plt.show()
    check_product_on_shelf(img2_gray, temp_gray, threshold)



def change_in_lighting_condition(img, img_gray, temp, temp_gray):
    # Perform cross-correlation between the image and the template
    out = zero_mean_cross_correlation(img_gray, temp_gray)
    save_image(f'{resultsDir}/change_in_lighting_out.jpg', out)

    # Find the location with maximum similarity
    y, x = np.unravel_index(out.argmax(), out.shape)

    # Display image
    plt.imshow(img)
    plt.title('Result (red marker on the detected location)')
    plt.axis('off')

    # Draw marker at detcted location
    plt.plot(x, y, 'rx', ms=25, mew=5)
    plt.show()

def final_normalized_cross_correlation(img, img_gray, temp, temp_gray):
    # Perform normalized cross-correlation between the image and the template
    out = normalized_cross_correlation(img_gray, temp_gray)
    save_image(f'{resultsDir}/final_out.jpg', out)

    # Find the location with maximum similarity
    y, x = np.unravel_index(out.argmax(), out.shape)

    # Display image
    plt.imshow(img)
    plt.title('Result (red marker on the detected location)')
    plt.axis('off')

    # Draw marker at detcted location
    plt.plot(x, y, 'rx', ms=25, mew=5)
    plt.show()

if __name__ == "__main__":
    # Load template and image in grayscale
    img = io.imread('../data/shelf.jpg')
    img_gray = io.imread('../data/shelf.jpg', as_gray=True)
    temp = io.imread('../data/template.jpg')
    temp_gray = io.imread('../data/template.jpg', as_gray=True)
    
    ### Hint: If you use python in headless environment, matplotlib.pyplot may not work 
    # properly. Instead you can use save_img in helpers to display the image.

    # Implement cross_correlation function in stduent.py and run the code below.
    ### Hint: implement your correlate_fast in student.py first.
    naive_cross_correlation(img, img_gray, temp, temp_gray)

    # Implement zero_mean_cross_correlation function in stduent.py and run the code below.
    simple_zero_mean_cross_correlation(img, img_gray, temp, temp_gray)

    # Load image of the shelf without the product
    img2 = io.imread('../data/shelf_soldout.jpg')
    img2_gray = io.imread('../data/shelf_soldout.jpg', as_gray=True)

    # Threshold is arbitrary, you would need to tune the threshold for a real application.
    you_can_check_product_on_shelf(img, img_gray, img2, img2_gray, temp, temp_gray, threshold=0.025)

    # Load image
    img = io.imread('../data/shelf_dark.jpg')
    img_gray = io.imread('../data/shelf_dark.jpg', as_gray=True)
    change_in_lighting_condition(img, img_gray, temp, temp_gray)

    # Implement normalized_cross_correlation function in stduent.py and run the code below.
    final_normalized_cross_correlation(img, img_gray, temp, temp_gray)