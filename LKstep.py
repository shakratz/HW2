import cv2
import numpy as np

def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

def LucasKanadeStep(I1, I2, WindowSize):
    hw = round(WindowSize / 2)  # hw = half window size
    Dp = [0, 0]
    num_rows, num_cols = I2.shape[:2]
    du = np.zeros((num_rows, num_cols))
    dv = np.zeros((num_rows, num_cols))

    Ix = cv2.Sobel(I2, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(I2, cv2.CV_64F, 0, 1, ksize=5)


    for i in range(hw, num_rows - hw):
        for j in range(hw, num_cols - hw):

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
            if is_invertible(BTB):
                BTIt = np.matmul(cv2.transpose(B), It)
                Dp = -np.matmul(np.linalg.inv(BTB), BTIt)

            # Store the du and dv in the array
            du[i, j] = Dp[0]
            dv[i, j] = Dp[1]

    return du, dv