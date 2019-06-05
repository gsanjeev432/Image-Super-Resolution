import skimage
import numpy as np
from matplotlib import pyplot as plt
from scipy import misc
from skimage.transform import rescale, resize
from skimage.filters import gaussian

def crop_image(img, scale):
    h = img.shape[0]
    w = img.shape[1]
    img = img[0:h - np.mod(h, scale), 0:w - np.mod(w, scale), :]
    return img

# Function for reading in image as array from file path and converting to a float type
def image_to_float(file_name):
    img = misc.imread(file_name)
    return skimage.img_as_float(img)

# Function for applying a gaussian blur kernel and downscaling the image
def downscale_image(img, blur_sigma=1, scale_factor=2):
    img_blur = gaussian(img, sigma=blur_sigma, multichannel=True, mode="reflect")
    return rescale(img_blur, 1.0/scale_factor)

# Function for upscaling the image with bicubic interpolation
def upscale_image(img, interpolation_order=3, scale_factor=2):
    return resize(img,
        output_shape=[img.shape[0]*scale_factor,
                      img.shape[1]*scale_factor,
                      img.shape[2]
                      ],
            order=interpolation_order
    )

def save_image(path, image):
    misc.imsave(path, image)

