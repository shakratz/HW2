import math
import sys
import scipy
import cv2
import numpy as np
from scipy import linalg
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


def LucasKanadeStep(I1, I2, WindowSize):
    hw = round(WindowSize / 2)  # hw = half window size
    Dp = [0, 0]
    num_rows, num_cols = I2.shape[:2]
    du = np.zeros((num_rows, num_cols))
    dv = np.zeros((num_rows, num_cols))

    Ix = cv2.Sobel(I1, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(I1, cv2.CV_64F, 0, 1, ksize=5)

    for i in range(0, num_rows - WindowSize, WindowSize):
        for j in range(0, num_cols - WindowSize, WindowSize):

            # Splitting to WindowSize
            I1seg = I1[i:i + WindowSize, j:j + WindowSize].flatten('F')
            I2seg = I2[i:i + WindowSize, j:j + WindowSize].flatten('F')
            Ixseg = Ix[i:i + WindowSize, j:j + WindowSize].flatten('F')
            Iyseg = Iy[i:i + WindowSize, j:j + WindowSize].flatten('F')

            # B = [Ix Iy] as CS
            B = np.array([Ixseg, Iyseg]).T
            # It is the difference between frames
            It = I2seg - I1seg

            # Delta p
            BTB = np.matmul(cv2.transpose(B), B)
            BTIt = np.matmul(cv2.transpose(B), It)
            try:
                Dp = np.matmul(np.linalg.inv(BTB), BTIt)  # Without minus since Sobel contain minus
            except np.linalg.LinAlgError:
                Dp = np.matmul(np.linalg.pinv(BTB), BTIt)  # Without minus since Sobel contain minus

            # Store the du and dv in the array
            du[i:i + WindowSize, j:j + WindowSize] = Dp[0] * np.ones((WindowSize, WindowSize))
            dv[i:i + WindowSize, j:j + WindowSize] = Dp[1] * np.ones((WindowSize, WindowSize))

    return du, dv


def WarpImage(I, u, v):
    num_rows, num_cols = I.shape[:2]
    # Finding the movement of pixel
    grid_x, grid_y = np.mgrid[0:(num_rows - 1):num_rows * 1j, 0:(num_cols - 1):num_cols * 1j]

    umov = np.matrix.flatten(grid_y + u).reshape(-1, 1)
    vmov = np.matrix.flatten(grid_x + v).reshape(-1, 1)

    # Parameters for interpolation
    points = np.concatenate((vmov, umov), axis=1)
    values = np.matrix.flatten(I)

    # Interpolation
    I_warp = griddata(points, values, (grid_x, grid_y), method='linear')

    # Fill holes
    I_warp[np.isnan(I_warp)] = I[np.isnan(I_warp)]
    return I_warp


def LucasKanadeOpticalFlow(I1, I2, WindowSize, MaxIter, NumLevels):
    du_dv_th = 0.00001  # init threshold - should not effect much
    I1_pyr = []
    I2_pyr = []
    I1_pyr.append(I1)
    I2_pyr.append(I2)

    # Creating image pyramids
    for pyrLevel in range(0, NumLevels - 1):
        dst = cv2.pyrDown(I1_pyr[pyrLevel])
        I1_pyr.append(dst)
        dst = cv2.pyrDown(I2_pyr[pyrLevel])
        I2_pyr.append(dst)

    # creating u and v
    num_rows, num_cols = I1_pyr[-1].shape[:2]
    u = np.zeros((num_rows, num_cols))  # Starting assumption: there is no motion
    v = np.zeros((num_rows, num_cols))
    u_last = u
    v_last = v
    # Main Algorithm
    for pyrLevel in range(NumLevels)[::-1]:  # starting with the smallest images
        # print('Started pyr: ' + str(pyrLevel))
        # Using the down sampled image
        I1curPyr = I1_pyr[pyrLevel]
        I2curPyr = I2_pyr[pyrLevel]

        # Setting threshold to stop iteration when u and v doesnt change much
        TotalPixels = I1curPyr.shape[1] * I1curPyr.shape[0]
        PixelTh = TotalPixels * 0.80

        for it in range(MaxIter):
            # Move the image I2 according to Delta p
            I2_warp = WarpImage(I2curPyr, u, v)

            # One step - Calculate Delta p
            (du, dv) = LucasKanadeStep(I1curPyr, I2_warp, WindowSize)

            # Update u and v
            u = u + du
            v = v + dv

            duSmall = (np.absolute(du) < du_dv_th).sum()
            dvSmall = (np.absolute(dv) < du_dv_th).sum()
            if (duSmall > PixelTh and dvSmall > PixelTh and it > 4) or (np.array_equal(u,u_last) and np.array_equal(v,v_last) and it > 4):
                #print('break at iter {0} pyr {1}'.format(it, pyrLevel))
                break
            u_last = u
            v_last = v

        if 4 > pyrLevel > 0:
            MaxIter = int(MaxIter / 2)
        if pyrLevel < 2 and MaxIter > 5:
            MaxIter = 5

        if pyrLevel != 0:  # not the First level

            dims = (I1_pyr[pyrLevel - 1].shape[1], I1_pyr[pyrLevel - 1].shape[0])

            # going up 1 pyramid level, increasing u,v
            u = cv2.pyrUp(u, dstsize=dims)
            v = cv2.pyrUp(v, dstsize=dims)
            u = u * 2
            v = v * 2

            # update dynamically the threshold
            du_dv_th = 0.2 * (np.mean(u) + np.mean(v)) / 2
    return u, v


#################### PART 3 ###############################

def LucasKanadeVideoStabilization(InputVid, WindowSize, MaxIter, NumLevels):
    orig_video = cv2.VideoCapture(InputVid)
    edge = 30

    # video output parameters
    fourcc = orig_video.get(6)
    fps = orig_video.get(5)
    frameSize = (int(orig_video.get(3)) - edge, int(orig_video.get(4)) - edge)
    hasFrames, I1 = orig_video.read()

    # define video output
    output_video = cv2.VideoWriter('output_WS{0}_It{1}.avi'.format(WindowSize, MaxIter), int(fourcc), fps, frameSize)

    # stabilization using LK with first frame
    size = (int(orig_video.get(4)), int(orig_video.get(3)))
    numFrames = int(orig_video.get(cv2.CAP_PROP_FRAME_COUNT))
    v = np.zeros(size)
    u = np.zeros(size)

    # write first frame to output video
    output_video.write(I1)

    # set the first frame
    I1_gray = cv2.cvtColor(I1, cv2.COLOR_RGB2GRAY)
    print('Frame number 1 from ' + str(numFrames) + ' is in process')
    I_gray = I1_gray

    for i in range(1, numFrames):
        print('Frame number ' + str(i + 1) + ' from ' + str(numFrames) + ' is in process')

        hasFrames, frame = orig_video.read()
        if hasFrames:  # hasFrames returns a bool, if frame is read correctly - it will be True
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            (du, dv) = LucasKanadeOpticalFlow(I_gray, gray_frame, WindowSize, MaxIter, NumLevels)

            # clean outliers
            du[du > (0.85 * np.max(du))] = 0
            dv[dv > (0.85 * np.max(du))] = 0

            # because all the frame move in the same way
            u += np.mean(du) * np.ones(size)
            v += np.mean(dv) * np.ones(size)

            # Move the frame according to Delta p
            frame_warp = WarpImage(gray_frame, u, v)

            # Second iteration for better results
            (du, dv) = LucasKanadeOpticalFlow(I_gray, frame_warp, WindowSize, MaxIter, NumLevels)

            # because all the frame move in the same way
            u += np.mean(du) * np.ones(size)
            v += np.mean(dv) * np.ones(size)

            # clean outliers
            du[du > (0.85 * np.max(du))] = 0
            dv[dv > (0.85 * np.max(du))] = 0

            # Move the frame according to Delta p
            frame_warp = WarpImage(gray_frame, u, v)

            # update the frame for next LK step
            I_gray = gray_frame.copy()

            # write frame to output video
            frame_warp = cv2.cvtColor(frame_warp.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            output_video.write(frame_warp[0:size[0] - edge, 0:size[1] - edge])

        else:
            break
    orig_video.release()
    output_video.release()
