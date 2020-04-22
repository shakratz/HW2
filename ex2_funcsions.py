import math
import sys
import scipy
import cv2
import numpy as np
from scipy import linalg
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


def LucasKanadeStep(I1, I2, WindowSize):
    hw = round(WindowSize / 2)  # hw = half window size
    Dp = [0, 0]
    num_rows, num_cols = I2.shape[:2]
    du = np.zeros((num_rows, num_cols))
    dv = np.zeros((num_rows, num_cols))

    Ix = cv2.Sobel(I1, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(I1, cv2.CV_64F, 0, 1, ksize=5)

    # set a threshold to see when Ix and Iy are very small
    #   IxIyTh = 0.5 * np.absolute(np.median(Ix)+np.median(Iy)/2)
    #   IxIyPixTh = 1   # 0.9  # % of pixels to reach threshold

    for i in range(hw, num_rows - hw,WindowSize):
        for j in range(hw, num_cols - hw,WindowSize):
            # Checking threshold of Ix and Iy
            #   SmallElementsY = (np.absolute(Iy[i - hw:i + hw + 1, j - hw:j + hw + 1]) < IxIyTh).sum()
            #   SmallElementsX = (np.absolute(Ix[i - hw:i + hw + 1, j - hw:j + hw + 1]) < IxIyTh).sum()
            #   if SmallElementsX < ((WindowSize ** 2) * IxIyPixTh) or SmallElementsY < ((WindowSize ** 2) * IxIyPixTh):

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
            BTIt = np.matmul(cv2.transpose(B), It)
            try:
                Dp = -np.matmul(np.linalg.pinv(BTB), BTIt)  # Without minus since Sobel contain minus
            except np.linalg.LinAlgError:
                print('Non invertible matrix!')
                Dp = [0, 0]

            # Store the du and dv in the array
            du[i - hw:i + hw + 1, j - hw:j + hw + 1] = Dp[0]
            dv[i - hw:i + hw + 1, j - hw:j + hw + 1] = Dp[1]

    return du, dv


"""
def bilinear_interpolate(I, y, x):
    num_rows, num_cols = I.shape[:2]
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, num_cols - 1)
    x1 = np.clip(x1, 0, num_cols - 1)
    y0 = np.clip(y0, 0, num_rows - 1)
    y1 = np.clip(y1, 0, num_rows - 1)

    Ia = I[y0, x0]
    Ib = I[y1, x0]
    Ic = I[y0, x1]
    Id = I[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def WarpImage(I, u, v):
    num_rows, num_cols = I.shape[:2]
    I_warp = np.zeros((num_rows, num_cols))

    # Move pixels according to u and v
    for i in range(num_rows):
        for j in range(num_cols):
            # Move pixel and make bi-linear interpolation to find value
            I_warp[i, j] = bilinear_interpolate(I, i + u[i, j], j + v[i, j])

    return I_warp

"""

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
    # I_warp = scipy.interpolate.LinearNDInterpolator(points, values)

    # Fill holes
    I_warp[np.isnan(I_warp)] = I[np.isnan(I_warp)]
    return I_warp




def LucasKanadeOpticalFlow(I1, I2, WindowSize, MaxIter, NumLevels):
    # du_dv_th = 0.0001  # init threshold - should not effect much
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

    # Main Algorithm
    for pyrLevel in range(NumLevels)[::-1]:  # starting with the smallest images
        print('Started pyr: ' + str(pyrLevel))
        # Using the down sampled image
        I1curPyr = I1_pyr[pyrLevel]
        I2curPyr = I2_pyr[pyrLevel]

        # Setting threshold to stop iteration when u and v doesnt change much
        # TotalPixels = I1curPyr.shape[1] * I1curPyr.shape[0]
        # PixelTh = TotalPixels  # * 0.95

        for it in range(MaxIter):
            # Move the image I2 according to Delta p
            I2_warp = WarpImage(I2curPyr, u, v)

            # One step - Calculate Delta p
            (du, dv) = LucasKanadeStep(I1curPyr, I2_warp, WindowSize)

            # If there is no more change in u and v - break
            # SmallElementsY = (np.absolute(du) < du_dv_th).sum()
            # SmallElementsX = (np.absolute(dv) < du_dv_th).sum()
            #if SmallElementsX > PixelTh and SmallElementsY > PixelTh:
            #    print('Break on iter: ' + str(it) + '      pyr: ' + str(pyrLevel))
            #    break

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

            # update threshold of du dv
            # du_dv_th = 0.5 * np.absolute((np.mean(u) + np.mean(v)) / 2)
    return u, v


#################### PART 3 ###############################
def LucasKanadeVideoStabilization(InputVid, WindowSize, MaxIter, NumLevels):
    orig_video = cv2.VideoCapture(InputVid)
    edge = WindowSize ** 2
    if edge % 2 != 0:
        edge += 1
    HalfEdge = int(edge / 2)

    # video output parameters
    fourcc = orig_video.get(6)
    fps = orig_video.get(5)
    # frameSize = (int(orig_video.get(3)) - edge, int(orig_video.get(4)) - edge)
    frameSize = (int(orig_video.get(3)) - edge, int(orig_video.get(4)) - edge)
    hasFrames, I1 = orig_video.read()

    # define video output
    output_video = cv2.VideoWriter('StabilizedVid_200940500_204251144.avi', int(fourcc), fps, frameSize)

    # write first frame to output video
    output_video.write(I1)

    # stabilization using LK with first frame
    I1_gray = cv2.cvtColor(I1, cv2.COLOR_RGB2GRAY)
    numFrames = int(orig_video.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(orig_video.get(4)), int(orig_video.get(3)))

    print('Frame number 1 from ' + str(numFrames) + ' is in process')

    v = np.zeros(size)
    u = np.zeros(size)
    RMS_thr = 3  # what is close between frames?
    I_gray = I1_gray

    for i in range(1, numFrames):
        print('Frame number ' + str(i + 1) + ' from ' + str(numFrames) + ' is in process')

        hasFrames, frame = orig_video.read()
        if hasFrames:  # hasFrames returns a bool, if frame is read correctly - it will be True
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            if np.sqrt(np.average(
                    np.power(gray_frame - I1_gray, 2))) < RMS_thr:  # write frame to output video if close to I1
                frame_warp = gray_frame

            elif np.sqrt(np.average(
                    np.power(gray_frame - I_gray, 2))) > RMS_thr:  # calculate change if frame not close to last frame
                (du, dv) = LucasKanadeOpticalFlow(I_gray, gray_frame, WindowSize, MaxIter, NumLevels)
                u += np.median(du) * np.ones(size)  # because all the frame move in the same way
                v += np.median(dv) * np.ones(size)  # because all the frame move in the same way

                # Move the frame according to Delta p
                frame_warp = WarpImage(gray_frame, u, v)
            else:  # if frame close to last frame we use the same u , v from last frame
                frame_warp = WarpImage(gray_frame, u, v)

            RMS1 = np.sqrt(np.average(np.power(gray_frame - I1_gray, 2)))
            RMS2 = np.sqrt(np.average(np.power(gray_frame - I_gray, 2)))
            print('RMS to I1 {:.02f} RMS to last frame {:.02f}\n'.format(RMS1, RMS2))

            # update the frame for next LK step
            I_gray = gray_frame.copy()

            # write frame to output video
            frame_warp = cv2.cvtColor(frame_warp.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            output_video.write(frame_warp[HalfEdge:size[0] - HalfEdge, HalfEdge:size[1] - HalfEdge])

        else:
            break
    orig_video.release()
    output_video.release()
