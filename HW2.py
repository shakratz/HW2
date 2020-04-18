import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io as sio
from ex2_funcsions import *

# FILL IN YOUR ID
# ID1 = 200940500
# ID2 = 204251144


#####################################PART 1: Lucas-Kanade Optical Flow################################################

# Load images I1,I2
IMG = sio.loadmat('HW2_PART1_IMAGES.mat')
I1 = IMG['I1']
I2 = IMG['I2']

# Choose parameters
WindowSize = 5  # Add your value here!
MaxIter = 10  # Add your value here!
NumLevels = 4  # Add your value here!

# Compute optical flow using LK algorithm
(u, v) = LucasKanadeOpticalFlow(I1, I2, WindowSize, MaxIter, NumLevels)

# Warp I2
I2_warp = WarpImage(I2, u, v)

# The RMS should decrease as the warped image (I2_warp) should be more similar to I1
print('RMS of original frames: ' + str(np.sqrt(np.average(np.power(I2 - I1, 2)))))
print('RMS of processed frames: ' + str(np.sqrt(np.average(np.power(I2_warp - I1, 2)))))
"""print('RMS of original frames: ' + str(np.sum(np.sum(np.abs((I1 - I2) ** 2)))))
print('RMS of processed frames: ' + str(np.sum(np.sum(np.abs((I1 - I2_warp) ** 2)))))"""

# Plot I1,I2,I2_warp
plt.subplot(1, 3, 1)
plt.imshow(I1, cmap='gray')
plt.title('I1'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 2)
plt.imshow(I2, cmap='gray')
plt.title('I2'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 3)
plt.imshow(I2_warp, cmap='gray')
plt.title('I2_warp'), plt.xticks([]), plt.yticks([])
plt.show()

###########################################3PART 2: Video Stabilization################################################
""""
# Choose parameters
WindowSize = None  # Add your value here!
MaxIter = None  # Add your value here!
NumLevels = None  # Add your value here!

# Load video file
InputVidName = 'input.avi'

# Stabilize video - save the stabilized video inside the function
StabilizedVid = LucasKanadeVideoStabilization(InputVidName, WindowSize, MaxIter, NumLevels)
"""
