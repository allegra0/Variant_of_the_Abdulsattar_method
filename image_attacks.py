from numpy import ceil, float16
from numpy.lib.function_base import quantile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde
import PIL
import random
from PIL import Image

def gaussian_noise(adresse,var):
# original image
    f = cv2.imread(adresse, 0)
    f = f/255 
#print(f)

# create gaussian noise
    x, y = f.shape
#print(x,y)
    mean = 0
#var = 0.005
    sigma = np.sqrt(var)
    n = np.random.normal(loc=mean, 
                     scale=sigma, 
                     size=(x,y))



# display the probability density function (pdf)
    kde = gaussian_kde(n.reshape(int(x*y)))
    dist_space = np.linspace(np.min(n), np.max(n), 100)
#plt.plot(dist_space, kde(dist_space))
    plt.xlabel('Noise pixel value'); plt.ylabel('Frequency')

# add a gaussian noise
    g = f + n
    return g*255

def SaltAndPepper(adresse, density):
    image = cv2.imread(adresse)
    # create an empty array with same size as input image
    output = np.zeros(image.shape, np.uint8)

    # parameter for controlling how much salt and paper are added
    threshhold = 1 - density

    # loop every each pixel and decide add the noise or not base on threshhold (density)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            possibility = random.random()
            if possibility < density:
                output[i][j] = 0
            elif possibility > threshhold:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def specklenoise (adresse,var):
    img = cv2.imread(adresse)
    sigma=np.sqrt(var)
    gauss = np.random.normal(0,sigma,img.size)
    gauss=gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
    noise=img+img*gauss
    return noise

def MedianFilter(adresse, filter_size):
    image = cv2.imread(adresse)
    # create an empty array with same size as input image
    output = np.zeros(image.shape, np.uint8)

    # create the kernel array of filter as same size as filter_size
    filter_array = [image[0][0]] * filter_size

    # deal with filter size = 3x3
    if filter_size == 9:
        for j in range(1, image.shape[0]-1):
            for i in range(1, image.shape[1]-1):
                filter_array[0] = image[j-1, i-1]
                filter_array[1] = image[j, i-1]
                filter_array[2] = image[j+1, i-1]
                filter_array[3] = image[j-1, i]
                filter_array[4] = image[j, i]
                filter_array[5] = image[j+1, i]
                filter_array[6] = image[j-1, i+1]
                filter_array[7] = image[j, i+1]
                filter_array[8] = image[j+1, i+1]

                # sort the array
                filter_array.sort()

                # put the median number into output array
                output[j][i] = filter_array[4]

    # deal with filter size = 5x5
    elif filter_size == 25:
        for j in range(2, image.shape[0]-2):
            for i in range(2, image.shape[1]-2):
                filter_array[0] = image[j-2, i-2]
                filter_array[1] = image[j-1, i-2]
                filter_array[2] = image[j, i-2]
                filter_array[3] = image[j+1, i-2]
                filter_array[4] = image[j+2, i-2]
                filter_array[5] = image[j-2, i-1]
                filter_array[6] = image[j-1, i-1]
                filter_array[7] = image[j, i-1]
                filter_array[8] = image[j+1, i-1]
                filter_array[9] = image[j+2, i-1]
                filter_array[10] = image[j-2, i]
                filter_array[11] = image[j-1, i]
                filter_array[12] = image[j, i]
                filter_array[13] = image[j+1, i]
                filter_array[14] = image[j+2, i]
                filter_array[15] = image[j-2, i+1]
                filter_array[16] = image[j-1, i+1]
                filter_array[17] = image[j, i+1]
                filter_array[18] = image[j+1, i+1]
                filter_array[19] = image[j+2, i+1]
                filter_array[20] = image[j-2, i+2]
                filter_array[21] = image[j-1, i+2]
                filter_array[22] = image[j, i+2]
                filter_array[23] = image[j+1, i+2]
                filter_array[24] = image[j+2, i+2]

                # sort the array
                filter_array.sort()

                # put the median number into output array
                output[j][i] = filter_array[12]
    return output

def MeanFilter(image, filter_size):
    #image = cv2.imread(adresse,0)
    #image = SaltAndPepper(adresse, 0.01)
    # create an empty array with same size as input image
    output = np.zeros(image.shape, np.uint8)

    # creat an empty variable
    result = 0

    # deal with filter size = 3x3
    if filter_size == 9:
        for j in range(1, image.shape[0]-1):
            for i in range(1, image.shape[1]-1):
                for y in range(-1, 2):
                    for x in range(-1, 2):
                        result = result + image[j+y, i+x]
                       
                output[j][i] = int(result / filter_size)
                result = 0

    # deal with filter size = 5x5
    elif filter_size == 25:
        for j in range(2, image.shape[0]-2):
            for i in range(2, image.shape[1]-2):
                for y in range(-2, 3):
                    for x in range(-2, 3):
                        result = result + image[j+y, i+x]
                print(result)
                output[j][i] = int(result / filter_size)
                result = 0
    
    return output

def gaussian_formula(x,y,sigma):
    from sklearn.preprocessing import normalize
    result = (1/(2*np.pi*(sigma**2)))*np.exp(-(x**2+y**2)/(2*(sigma**2)))
    # normalize the kernel to avoid cut off with
    # large sigma values
    return result/result.sum()

def gaussian_filter(adresse, filter_size):
    """
    Gaussian filter implementation using HSV color space
    """
    from itertools import product
    image = cv2.imread(adresse)
    sigma=1
    #sigma=np.sqrt(var)
    img = image[:,:,2]
    center = filter_size//2
    x,y = np.mgrid[0-center:filter_size-center, 0-center:filter_size-center]
    filter_ = gaussian_formula(x,y,sigma)

    # calculate the resulting image size
    # after applying gaussian filter, y coordinate
    new_img_height = image.shape[0] - filter_size + 1
    new_img_width = image.shape[1] - filter_size + 1

    # stack all possible windows in the image vertically
    # to apply the filter later on
    new_image = np.empty((new_img_height*new_img_width, filter_size**2))

    row = 0
    for i,j in product(range(new_img_height), range(new_img_width)):
        new_image[row,:] = np.ravel(img[i:i+filter_size, j:j+filter_size])
        row += 1

    filter_ = np.ravel(filter_)
    filtered_image = np.dot(new_image, filter_).reshape(new_img_height, new_img_width).astype(np.uint8)
    image[center:new_img_height+center,center:new_img_width+center,2] = filtered_image

    return image



