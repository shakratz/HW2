import math
import cv2
import numpy as np


def LucasKanadeStep(I1, I2, WindowSize):
    hw = round(WindowSize / 2)  # hw = half window size
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

    for i in range(hw, num_rows - hw):
        for j in range(hw, num_cols - hw):
            # Splitting to WindowSize
            I1seg = I1[i - hw:i + hw + 1, j - hw:j + hw + 1].flatten()
            I2seg = I2[i - hw:i + hw + 1, j - hw:j + hw + 1].flatten()
            Ixseg = Ix[i - hw:i + hw + 1, j - hw:j + hw + 1].flatten()
            Iyseg = Iy[i - hw:i + hw + 1, j - hw:j + hw + 1].flatten()

            # B = [Ix Iy] as CS
            B = np.array([np.column_stack(Ixseg), np.column_stack(Iyseg)])
            B = np.reshape(B, (25, 2))
            # It is the difference between frames
            It = I2seg - I1seg

            # Delta p
            BBT = np.matmul(cv2.transpose(B), B)
            BIt = np.matmul(cv2.transpose(B), It)

            # In case det(A)=0 it is not invertible
            if np.linalg.det(BBT) == 0:
                Dp = [0, 0]
            else:
                Dp = -np.matmul(np.linalg.inv(BBT), BIt)

            # Store the du and dv in the array
            du[i, j] = Dp[0]
            dv[i, j] = Dp[1]

    return du, dv


def WarpImage(I, u, v):
    num_rows, num_cols = I.shape[:2]
    I_warp = np.ones((num_rows, num_cols)) * (-1)  # holes will be marked as "-1"

    # Move pixels according to u and v
    for y in range(num_cols):
        for x in range(num_rows):
            du = u[x, y]
            dv = v[x, y]

            # Keep the new pixel index in the array indices
            ymov = int(round(y + dv))
            if ymov >= num_cols:
                ymov = num_cols - 1

            xmov = int(round(x + du))
            if xmov >= num_rows:
                xmov = num_rows - 1

            # move the value to its new position
            I_warp[xmov, ymov] = I[x, y]

    # Fill holes
    for y in range(num_cols):
        for x in range(num_rows):
            if I_warp[x, y] == -1:
                I_warp[x, y] = I[x, y]

    return I_warp


def LucasKanadeOpticalFlow(I1, I2, WindowSize, MaxIter, NumLevels):
    I1_pyr = []
    I2_pyr = []
    I1_pyr.append(I1)
    I2_pyr.append(I2)
    odd_flag = np.zeros((NumLevels,2))
    # Creating image pyramids
    for pyrLevel in range(1, NumLevels):
        dst = cv2.pyrDown(I1_pyr[pyrLevel - 1])
        I1_pyr.append(dst)
        dst = cv2.pyrDown(I2_pyr[pyrLevel - 1])
        I2_pyr.append(dst)

        # Create odd flag array for later u/v dim fix
        if math.ceil(dst.shape[1]) % 2 == 1:
            odd_flag[pyrLevel,1] = 1
        if math.ceil(dst.shape[0]) % 2 == 1:
            odd_flag[pyrLevel,0] = 1

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

        if pyrLevel != 0:  # not the First level

            # Because pyrUp rounds up, need to change it for odd picture
            if odd_flag[pyrLevel-1,0] == 1:
                dim0 = math.ceil(I1down.shape[0] * 2) - 1
            else:
                dim0 = math.ceil(I1down.shape[0] * 2)
            if odd_flag[pyrLevel-1, 1] == 1:
                dim1 = math.ceil(I1down.shape[1] * 2) - 1
            else:
                dim1 = math.ceil(I1down.shape[1] * 2)
            dims = (dim1, dim0)

            # going up 1 pyramid level, increasing u,v
            u = cv2.pyrUp(u, dstsize=dims)
            v = cv2.pyrUp(v, dstsize=dims)
            u = u * 2
            v = v * 2
    return u, v


#################### PART 3 ###############################

def LucasKanadeVideoStabilization(InputVid, WindowSize, MaxIter, NumLevels):
    u = 1
