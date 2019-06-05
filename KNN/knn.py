import numpy as np
import os
from sklearn.feature_extraction import image
from sklearn.neighbors import NearestNeighbors
from scipy import misc
import utils
SAVE_DATA_DIR = "./saved_data/" # Directory for saving data
# Implementation of K-Nearest-Neighbors algorithm for Image Super Resolution
class KNN():
    def __init__(self, training_dir=None, test_file=None, lr_patch_size=5, hr_patch_size=10, num_neighbors=3, scale=3):
        self.training_set = [] # Training examples
        self.training_dir = training_dir # Training directory path
        self.test_file = test_file # Test file path
        self.lr_patch_size = lr_patch_size # Low resolution image patch size
        self.hr_patch_size = hr_patch_size # High resolution image patch size
        self.num_neighbors = num_neighbors # Number of neighbors for knn algorithm
        self.scale = scale # Scale of resizing image
        self.knn = None # Trained KNN model

    # Function for preprocessing data and compiling training set
    def preprocess_data(self):
        # Initialize empty lists to store patch from LR image and difference between
        # corresponding patches in HR image and interpolated image
        patch_list = []
        dif_list =[]

        print("processing training set...")

        # Iterate through training directory
        for file_name in os.listdir(self.training_dir):
            file_name = self.training_dir + file_name
            # Read in image as array and convert to float type
            img = utils.image_to_float(file_name)
            img = utils.crop_image(img, scale=self.scale)
            
            # Apply blur kernel and downscale
            img_LR = utils.downscale_image(img, scale_factor=self.scale)
            
            # Upscale with bicubic interpolation
            img_interpolate = utils.upscale_image(img_LR, scale_factor=self.scale)
            

            # Extract patches from images, using corresponding patch sizes
            original_patches = image.extract_patches_2d(img, (self.hr_patch_size, self.hr_patch_size))
            upscale_patches = image.extract_patches_2d(img_interpolate, (self.hr_patch_size, self.hr_patch_size))
            downscale_patches = image.extract_patches_2d(img_LR, (self.lr_patch_size, self.lr_patch_size))

            nrow = img_LR.shape[0] - self.lr_patch_size + 1
            ncol = img_LR.shape[1] - self.lr_patch_size + 1

            # Iterate over patches in the LR image
            low_idx = 0
            for i in range(nrow):
                for j in range(ncol):
                    # Find the corresponding patches in the upscaled images and the original image
                    high_idx = i*(img_interpolate.shape[1]-self.hr_patch_size) + low_idx*2
                    # Compute the difference between the patches in the original image and the upscaled image
                    dif = original_patches[high_idx] - upscale_patches[high_idx]

                    low_res = downscale_patches[low_idx]
                    # Save the LR patch and computed differences for the training set
                    patch_list.append(low_res.flatten())
                    dif_list.append(dif.flatten())
                    low_idx += 1

        # Save the list of LR patches and the list of differences to the training set
        self.training_set = [patch_list, dif_list]
        print("finished processing...")

    # Function for fitting the knn model to the training set using the sci-kit learn implementation
    def train_knn(self):
        print("training...")
        # Fit training set to KNN algorithm
        neighbors = NearestNeighbors(n_neighbors=self.num_neighbors)
        self.knn = neighbors.fit(self.training_set[0])

    # Function for using K-Nearest-Neighbors algorithm for image super resolution
    def test_knn(self):
        print("testing...")
        # Read in low resolution image as dataset
        img = utils.image_to_float(self.test_file)
        img = utils.crop_image(img, scale=self.scale)
        misc.imsave(SAVE_DATA_DIR+"result/ground_truth.png",img)
        # Apply blur kernel and downscale
        img_LR = utils.downscale_image(img, scale_factor=self.scale)
        misc.imsave(SAVE_DATA_DIR+"result/input_LR.png",img_LR)
        # Upscale with bicubic interpolation
        img_interpolate = utils.upscale_image(img_LR, scale_factor=self.scale)
        misc.imsave(SAVE_DATA_DIR+"result/bicubic.png",img_interpolate)
        # Extract patches from images
        downscale_patches = image.extract_patches_2d(img_LR, (self.lr_patch_size, self.lr_patch_size))
        upscale_patches = image.extract_patches_2d(img_interpolate, (self.hr_patch_size, self.hr_patch_size))

        # Iterate over low resolution patches
        for patch_index in range(len(downscale_patches)):
            if patch_index%1000 == 0:
                print("on patch %d of %d" %(patch_index, len(downscale_patches)))
            # Record the indices for the closest "neighbors" of the patch
            distances, indices = self.knn.kneighbors([downscale_patches[patch_index].flatten()])
            # Retrieve differences from training set and calculate average
            differences = np.zeros((self.hr_patch_size, self.hr_patch_size, 3))
            for dif in indices[0]:
                differences += np.asarray(self.training_set[1][dif]).reshape((self.hr_patch_size, self.hr_patch_size, 3))
            ave_differences = differences/len(indices[0])
            # Add the average of the differences to the upscaled patches
            upscale_patches[patch_index] = upscale_patches[patch_index] + ave_differences

            # Convert floats that are less than zero to zero and greater than one to one
            upscale_patches[patch_index][upscale_patches[patch_index]<0] = 0
            upscale_patches[patch_index][upscale_patches[patch_index]>1] = 1
        # Reconstruct image from patches
        reconstructed = image.reconstruct_from_patches_2d(upscale_patches, img_interpolate.shape)
        return reconstructed







