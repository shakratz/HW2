import math
import cv2
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

"""
def LucasKanadeStep(I1, I2, WindowSize):
    hw = round(WindowSize / 2)  # hw = half window size
    num_rows, num_cols = I2.shape[:2]
    du = np.zeros((num_rows, num_cols))
    dv = np.zeros((num_rows, num_cols))

    Ix = cv2.Sobel(I2, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(I2, cv2.CV_64F, 0, 1, ksize=5)

    for i in range(hw, num_rows - hw):
        for j in range(hw, num_cols - hw):
            B = []
            # Splitting to WindowSize
            I1seg = I1[i - hw:i + hw + 1, j - hw:j + hw + 1].flatten('F')
            I2seg = I2[i - hw:i + hw + 1, j - hw:j + hw + 1].flatten('F')
            Ixseg = Ix[i - hw:i + hw + 1, j - hw:j + hw + 1].flatten('F')
            Iyseg = Iy[i - hw:i + hw + 1, j - hw:j + hw + 1].flatten('F')

            # B = [Ix Iy] as CS
            B = np.array([Ixseg, Iyseg]).T
            # It is the difference between frames
            It = I2seg - I1seg

            # Delta p
            BTB = np.matmul(cv2.transpose(B), B)

            # In case det(BTB)=0 it is not invertible - if close to 0 their is no motion (background)
            if np.linalg.det(BTB) != 0:
                BTIt = np.matmul(cv2.transpose(B), It)
                Dp = np.matmul(np.linalg.inv(BTB), BTIt)  # Without minus since Sobel contain minus

            # Store the du and dv in the array
            du[i, j] = Dp[0]
            dv[i, j] = Dp[1]

    return du, dv
"""
"""def WarpImage(I, u, v):
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
    for x in range(num_rows):
        for y in range(num_cols):
            if I_warp[x, y] == -1:
                I_warp[x, y] = I[x, y]

    return I_warp"""


def LucasKanadeStep(I1, I2, WindowSize):
    height = I1.shape[0]
    width = I2.shape[1]
    u = np.zeros((height, width))
    v = np.zeros((height, width))

    # Creating Matrices for Calculation #
    Ix = cv2.Sobel(I1, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(I1, cv2.CV_64F, 0, 1, ksize=5)
    It = I2 - I1

    for i in range(0, height - WindowSize, WindowSize):
        for j in range(0, width - WindowSize, WindowSize):
            B = np.zeros((WindowSize ** 2, 2))
            It_window = np.zeros((WindowSize ** 2, 1))
            # Creating the Vectors #
            for n in range(WindowSize):
                for m in range(WindowSize):
                    B[n * WindowSize + m, 0] = Ix[i + n, j + m]
                    B[n * WindowSize + m, 1] = Iy[i + n, j + m]
                    It_window[n * WindowSize + m] = It[i + n, j + m]

            # In case det(A)=0 it is not invertible - if close to 0 their is no motion (background)
            if np.linalg.det(np.matmul(B.T, B)) != 0:
                dp = np.matmul(np.matmul(np.linalg.inv(np.matmul(B.T, B)), B.T), It_window)

            u[i:i + WindowSize, j:j + WindowSize] = dp[0] * np.ones((WindowSize, WindowSize))
            v[i:i + WindowSize, j:j + WindowSize] = dp[1] * np.ones((WindowSize, WindowSize))

    return u, v


def WarpImage(I, u, v):
    num_rows, num_cols = I.shape[:2]
    u_vec = np.tile(np.linspace(0, num_cols - 1, num_cols), (num_rows, 1))
    v_vec = np.tile(np.linspace(0, num_rows - 1, num_rows).reshape(-1, 1), (1, num_cols))
    umov = np.matrix.flatten(u_vec + u).reshape(-1, 1)
    vmov = np.matrix.flatten(v_vec + v).reshape(-1, 1)

    grid_x, grid_y = np.mgrid[0:(num_rows - 1):num_rows * 1j, 0:(num_cols - 1):num_cols * 1j]
    points = np.concatenate((vmov, umov), axis=1)
    values = np.matrix.flatten(I)
    I_warp = griddata(points, values, (grid_x, grid_y), method='linear')

    # Fill holes
    I_warp[np.isnan(I_warp)] = I[np.isnan(I_warp)]
    return I_warp


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
        # Using the down sampled image
        I1down = I1_pyr[pyrLevel]
        I2down = I2_pyr[pyrLevel]

        for it in range(MaxIter):
            # Move the image I2 according to Delta p
            I2_warp = WarpImage(I2down, u, v)

            # One step - Calculate Delta p
            (du, dv) = LucasKanadeStep(I1down, I2_warp, WindowSize)

            if du.any() == 0 and dv.any() == 0:  # If there is no more change in u and v - break
                break

            # Update u and v
            u = u + du
            v = v + dv

        if pyrLevel != 0:  # not the First level

            dims = (I1_pyr[pyrLevel - 1].shape[1], I1_pyr[pyrLevel - 1].shape[0])
            # dim1=I1_pyr[pyrLevel-1].shape[1]
            # going up 1 pyramid level, increasing u,v
            u = cv2.pyrUp(u, dstsize=dims)
            v = cv2.pyrUp(v, dstsize=dims)
            u = u * 2
            v = v * 2
    return u, v


#################### PART 3 ###############################

def LucasKanadeVideoStabilization(InputVid, WindowSize, MaxIter, NumLevels):
    orig_video = cv2.VideoCapture(InputVid)
    HalfWt = (WindowSize ** 2) // 2
    HalfWb = (WindowSize ** 2) // 2 + (WindowSize ** 2) % 2
    # video output parameters
    fourcc = orig_video.get(cv2.CAP_PROP_FOURCC)
    fps = orig_video.get(cv2.CAP_PROP_FPS)
    frameSize = ((int(orig_video.get(4)) - WindowSize ** 2), (int(orig_video.get(3)) - WindowSize ** 2))
    frameSize = (int(orig_video.get(4)), int(orig_video.get(3)))
    hasFrames, I1 = orig_video.read()

    # define video output
    output_video = cv2.VideoWriter('StabilizedVid_200940500_204251144.avi', int(fourcc), fps, frameSize)

    # write first frame to output video
    #output_video.write(I1[HalfWb:frameSize[0] - HalfWt, HalfWb:frameSize[1] - HalfWt])
    output_video.write(I1)

    # stabilization using LK with first frame
    I_gray = cv2.cvtColor(I1, cv2.COLOR_RGB2GRAY)
    numFrames = int(orig_video.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(orig_video.get(4)), int(orig_video.get(3)))

    v = np.zeros(size)
    u = np.zeros(size)

    for i in range(1, 3):
        print('Frame number ' + str(i + 1) + ' from ' + str(numFrames) + ' is in process')

        hasFrames, frame = orig_video.read()
        if hasFrames:  # hasFrames returns a bool, if frame is read correctly - it will be True
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            (du, dv) = LucasKanadeOpticalFlow(I_gray, gray_frame, WindowSize, MaxIter, NumLevels)
            u += np.mean(du) * np.ones(size)
            v += np.mean(dv) * np.ones(size)
            # Move the frame according to Delta p
            frame_warp = WarpImage(gray_frame, u, v)

            # update the frame for next LK step
            I_gray = gray_frame.copy()

            # write frame to output video
            frame_warp = cv2.cvtColor(frame_warp.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            #output_video.write(frame_warp[HalfWb:frameSize[0] - HalfWt, HalfWb:frameSize[1] - HalfWt])
            output_video.write(frame_warp)
        else:
            break
    orig_video.release()
    output_video.release()
