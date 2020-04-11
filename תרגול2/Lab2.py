import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def plot_two_by_two_subplot(first_img_name, second_img_name, third_img_name, fourth_img_name,
                            first_title="", second_title="", third_title="", fourth_title=""):
    """
    Plot four images (with corresponding titles if given) on a 2X2 grid.
    - - - - -
    - 1 - 2 -
    - - - - -
    - 3 - 4 -
    - - - - -
    :param first_img_name:
    :param second_img_name:
    :param third_img_name:
    :param fourth_img_name:
    :param first_title:
    :param second_title:
    :param third_title:
    :param fourth_title:
    :return: void
    """
    first_img = mpimg.imread(first_img_name)
    second_img = mpimg.imread(second_img_name)
    third_img = mpimg.imread(third_img_name)
    fourth_img = mpimg.imread(fourth_img_name)
    fig = plt.figure()
    first_img_plt = fig.add_subplot(2, 2, 1)
    first_img_plt.set_title(first_title)
    plt.imshow(first_img)
    second_img_plt = fig.add_subplot(2, 2, 2)
    second_img_plt.set_title(second_title)
    plt.imshow(second_img)
    third_img_plt = fig.add_subplot(2, 2, 3)
    third_img_plt.set_title(third_title)
    plt.imshow(third_img)
    fourth_img_plt = fig.add_subplot(2, 2, 4)
    fourth_img_plt.set_title(fourth_title)
    plt.imshow(fourth_img)
    plt.show()



# ##########################################Image_Blending########################################################
#

NUM_OF_LEVELS = 6
IMAGE_SIZE = (10 * 2 ** (NUM_OF_LEVELS - 1), 10 * 2 ** (NUM_OF_LEVELS - 1))

apple = cv2.imread('apple.jpg')
orange = cv2.imread('orange.jpg')

apple = cv2.resize(apple, IMAGE_SIZE)
orange = cv2.resize(orange, IMAGE_SIZE)


# generate Gaussian pyramid for apple
apple_pyramid_level = apple.copy()
apple_gaussian_p = [apple_pyramid_level]
for i in range(NUM_OF_LEVELS):
    apple_pyramid_level = cv2.pyrDown(apple_pyramid_level)
    apple_gaussian_p.append(apple_pyramid_level)

# generate Gaussian pyramid for orange
orange_pyramid_level = orange.copy()
orange_gaussian_p = [orange_pyramid_level]
for i in range(NUM_OF_LEVELS):
    orange_pyramid_level = cv2.pyrDown(orange_pyramid_level)
    orange_gaussian_p.append(orange_pyramid_level)

# generate Diff Pyramid for apple
apple_diff_p = [apple_gaussian_p[NUM_OF_LEVELS - 1]]
for i in range(NUM_OF_LEVELS - 1, 0, -1):
    apple_diff_level = cv2.pyrUp(apple_gaussian_p[i])
    apple_diff_level = cv2.subtract(apple_gaussian_p[i-1], apple_diff_level)
    apple_diff_p.append(apple_diff_level)

# generate Diff Pyramid for orange
orange_diff_p = [orange_gaussian_p[NUM_OF_LEVELS - 1]]
for i in range(NUM_OF_LEVELS - 1, 0, -1):
    orange_diff_level = cv2.pyrUp(orange_gaussian_p[i])
    L = cv2.subtract(orange_gaussian_p[i-1], orange_diff_level)
    orange_diff_p.append(L)

# Now add left and right halves of images in each level
blend_diff_p = []
for la, lb in zip(apple_diff_p, orange_diff_p):
    rows, cols, dpt = la.shape
    blend_diff_level = np.hstack((la[0:rows, 0:int(cols/2), :], lb[0:rows, int(cols/2):cols, :]))
    blend_diff_p.append(blend_diff_level)

# now reconstruct
blend_reconstructed = blend_diff_p[0]
for i in range(1, NUM_OF_LEVELS):
    blend_reconstructed = cv2.pyrUp(blend_reconstructed)
    blend_reconstructed = cv2.add(blend_reconstructed, blend_diff_p[i])

# image with direct connecting each half
direct_blend_img = np.hstack((apple[:, :int(cols/2)], orange[:, int(cols/2):]))

cv2.imwrite('Pyramid_blending.jpg', blend_reconstructed)
cv2.imwrite('Direct_blending.jpg', direct_blend_img)

apple = mpimg.imread('apple.jpg')
orange = mpimg.imread('orange.jpg')
pyramid_blending_img = mpimg.imread('Pyramid_blending.jpg')
direct_blending_img = mpimg.imread('Direct_blending.jpg')

plot_two_by_two_subplot('apple.jpg', 'orange.jpg',  'Direct_blending.jpg', 'Pyramid_blending.jpg',
                        'Original apple', 'Original orange', 'Direct', 'Pyramid')



