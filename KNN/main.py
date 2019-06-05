from knn import KNN
from scipy import misc
import os
import utils
import cv2
from scipy import misc

TRAINING_DIR = "./data/91-image/" # Directory containing training images
TEST_FILE = "./data/Set5/baby_GT.bmp" # File path of test image
SAVE_DATA_DIR = "./saved_data/" # Directory for saving data
knn = KNN(training_dir=TRAINING_DIR, test_file=TEST_FILE)
knn.preprocess_data()
knn.train_knn()
image = knn.test_knn()
misc.imsave(SAVE_DATA_DIR+"result/knn_result.png",image)

''' PSNR Calculation '''

im1 = misc.imread('./saved_data/result/ground_truth.png')
im2 = misc.imread('./saved_data/result/knn_result.png')
psnr = cv2.PSNR(im1,im2)
print ('PSNR: ',psnr)