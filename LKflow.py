import cv2
import numpy as np

def LucasKanadeOpticalFlow(I1, I2, WindowSize, MaxIter, NumLevels):

    I1_pyr = []
    I2_pyr = []
    I1_pyr.append(I1)
    I2_pyr.append(I2)
    odd_flag = np.zeros((NumLevels, 2))
    # Creating image pyramids
    for pyrLevel in range(0, NumLevels - 1):
        dst = cv2.pyrDown(I1_pyr[pyrLevel])
        I1_pyr.append(dst)
        dst = cv2.pyrDown(I2_pyr[pyrLevel])
        I2_pyr.append(dst)

    # creating u and v
    num_rows, num_cols = dst.shape[:2]
    u = np.zeros((num_rows, num_cols))  # Starting assumption: there is no motion
    v = np.zeros((num_rows, num_cols))

    # Main Algorithm
    for pyrLevel in range(NumLevels)[::-1]:  # starting with the smallest images
        print('Started pyr: ' + str(pyrLevel))
        # Using the down sampled image
        I1curPyr = I1_pyr[pyrLevel]
        I2curPyr = I2_pyr[pyrLevel]


        for it in range(MaxIter):
            # Move the image I2 according to Delta p
            I2_warp = WarpImage(I2curPyr, u, v)

            # One step - Calculate Delta p
            (du, dv) = LucasKanadeStep(I1curPyr, I2_warp, WindowSize)

            # Update u and v
            u = u + du
            v = v + dv

        if pyrLevel != 0:  # not the First level
            dims = (I1_pyr[pyrLevel - 1].shape[1], I1_pyr[pyrLevel - 1].shape[0])
            # going up 1 pyramid level, increasing u,v
            u = cv2.pyrUp(u, dstsize=dims)
            v = cv2.pyrUp(v, dstsize=dims)
            u = u * 2
            v = v * 2

    return u, v