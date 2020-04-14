import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io as sio


def LucasKanadeStep(I1, I2, WindowSize):
    num_rows, num_cols = I2.shape[:2]
    du = np.zeros((num_rows, num_cols))
    dv = np.zeros((num_rows, num_cols))

    # Shifting x by 1
    translation_matrix = np.float32([[1, 0, 1], [0, 1, 0]])
    x_shifted = cv2.warpAffine(I2, translation_matrix, (num_cols, num_rows))

    # Shifting y by 1
    translation_matrix = np.float32([[1, 0, 0], [0, 1, 1]])
    y_shifted = cv2.warpAffine(I2, translation_matrix, (num_cols, num_rows))

    # Derivatives
    Ix = I2 - x_shifted
    Iy = I2 - y_shifted

    for i in range(num_rows):
        for j in range(num_cols):
            # Splitting to WindowSize - DIDNT IMPLEMENT IT YET
            # split Ix, Iy, I2, It

            # B = [Ix Iy] as CS
            B = [np.column_stack(Ix), np.column_stack(Iy)]

            # It is the difference between frames
            It = I2 - I1

            # Delta p
            Dp = -np.linalg.inv((cv2.transpose(B) * B)) * cv2.transpose(B) * It
            du[j, i] = Dp[0]
            dv[j, i] = Dp[1]

    return du, dv


def WarpImage(I, u, v):
    num_rows, num_cols = I.shape[:2]
    I_warp = np.zeros((num_rows, num_cols))

    for y in range(num_rows):
        for x in range(num_cols):
            du = u[x, y]
            dv = v[x, y]
            # du dv need to be round or bi-linear or something
            I_warp[x + du, y + dv] = I[x, y]

            # Why do we need meshgrid?
    return I_warp


def LucasKanadeOpticalFlow(I1, I2, WindowSize, MaxIter, NumLevels):
    I1_pyr = []
    I2_pyr = []
    I1_pyr.append(I1)
    I2_pyr.append(I2)

    # Creating image pyramids
    for pyrLevel in range(NumLevels):
        dst = cv2.pyrDown(I1)
        I1_pyr.append(dst)
        dst = cv2.pyrDown(I2)
        I2_pyr.append(dst)

    # creating u and v
    num_rows, num_cols = dst.shape[:2]
    u = np.zeros((num_rows, num_cols))  # Starting assumption: there is no motion
    v = np.zeros((num_rows, num_cols))

    # Main Algorithm
    for pyrLevel in range(NumLevels)[::-1]:  # starting with the smallest images
        # Using the down sampled image
        I1down = I1_pyr[pyrLevel]
        I2down = I2_pyr[pyrLevel]

        for inter in range(MaxIter):
            # Move the image I2 according to Delta p
            I_warp = WarpImage(I2down, u, v)

            # One step - Calculate Delta p
            (du, dv) = LucasKanadeStep(I1down, I_warp, WindowSize)

            # Update u and v
            u = u + du
            v = v + dv

        if pyrLevel != 0:   # not the last level
            # going up 1 pyramid level, increasing u,v
            u = cv2.pyrUp(u)
            v = cv2.pyrUp(v)
            u = u * 2
            v = v * 2

    return u, v


#################### PART 3 ###############################

def LucasKanadeVideoStabilization(InputVid, WindowSize, MaxIter, NumLevels):
    u = 1
