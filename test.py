from matplotlib import cm
import numpy as np
from scipy.signal import convolve2d
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
# print(1)
def sobel_filter(im, k_size):

    im = im.astype(np.float)
    # print(im.shape)
    # width, height, c = im.shape
    # if im.shape
    if len(im.shape) > 2:
        width, height, c = im.shape
        if c > 1:
            img = 0.2126 * im[:,:,0] + 0.7152 * im[:,:,1] + 0.0722 * im[:,:,2]
        else:
            img = im
    else: img = im
    assert(k_size == 3 or k_size == 5);

    if k_size == 3:
        kh = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float)
        kv = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype = np.float)
    else:
        kh = np.array([[-1, -2, 0, 2, 1], 
                   [-4, -8, 0, 8, 4], 
                   [-6, -12, 0, 12, 6],
                   [-4, -8, 0, 8, 4],
                   [-1, -2, 0, 2, 1]], dtype = np.float)
        kv = np.array([[1, 4, 6, 4, 1], 
                   [2, 8, 12, 8, 2],
                   [0, 0, 0, 0, 0], 
                   [-2, -8, -12, -8, -2],
                   [-1, -4, -6, -4, -1]], dtype = np.float)
    # print(img.shape)
    # print(kh.shape)
    gx = convolve2d(img, kh, mode='same', boundary = 'symm', fillvalue=0)
    gy = convolve2d(img, kv, mode='same', boundary = 'symm', fillvalue=0)

    g = np.sqrt(gx * gx + gy * gy)
    g *= 255.0 / np.max(g)

    #plt.figure()
    #plt.imshow(g, cmap=plt.cm.gray)      

    return g
import numpy as np

def gaussian_kernel(size, sigma=1):

    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def noise_reduce(image, gaussian_ker):
    im = image.astype(np.float)
    # print(im.shape)
    # width, height, c = im.shape
    # if im.shape
    if len(im.shape) > 2:
        width, height, c = im.shape
        if c > 1:
            img = 0.2126 * im[:,:,0] + 0.7152 * im[:,:,1] + 0.0722 * im[:,:,2]
        else:
            img = im
    else: img = im
    return convolve2d(img, gaussian_ker, mode='same', boundary = 'symm', fillvalue=0)
    ...
image = mpimg.imread('images/BUD78DD.jpg')
gray_image = image[:,:,0]
plt.imshow(gray_image, cmap='gray')
plt.show()
# test1 =  convolve2d(image, gaussian_kernel(5), mode='same', boundary = 'symm', fillvalue=0)
# test2 = sobel_filter(gray_image,5)
test1 = noise_reduce(gray_image, gaussian_kernel(5))
# print(image.shape)
# print(np.array(gaussian_kernel(5)).shape)
# # print(test)
# invert = cv2.bitwise_not(test)

plt.imshow(test1, cmap='gray')
plt.show()
# print(len(image.shape))
# print(image.shape)
# print(test.shape)
# print(test)
# print(1+1)