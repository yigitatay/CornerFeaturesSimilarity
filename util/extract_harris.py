###########################################
### UNNECESSARY, BUT STILL FINE TO KEEP ###
###########################################

import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndi
import cv2

# Harris corner detector
def extract_harris(img, sigma = 1.0, k = 0.05, thresh = 1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0


    # Compute image gradients

    ## Apparently Sobel filters are generally used to calculate image gradients
    ## which takes into account the diagonal directions as well:
    filter_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    ## But to get the results of the formula on the slides, we could use these filters instead too:
    # filter_X = np.array([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]])
    # filter_Y = np.array([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]])

    grads_X = signal.convolve2d(img, filter_X, mode='same')
    grads_Y = signal.convolve2d(img, filter_Y, mode='same')

    ## To visualize the gradients
    # cv2.imwrite("X_grads.png", (grads_X*255.0).astype(int))
    # cv2.imwrite("Y_grads.png", (grads_Y*255.0).astype(int))

    # Compute local auto-correlation matrix

    # First we need to get the values for the 2x2 matrix in the sum
    grads_X_squared = grads_X**2
    grads_Y_squared = grads_Y**2
    grads_XY = grads_X * grads_Y

    ## Apply Gaussian blur to the gradient matrices
    grads_X_blurred = cv2.GaussianBlur(src=grads_X_squared, ksize=(3,3), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    grads_Y_blurred = cv2.GaussianBlur(src=grads_Y_squared, ksize=(3,3), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    grads_XY_blurred = cv2.GaussianBlur(src=grads_XY, ksize=(3,3), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)


    ## Calculate the M matrix for each pixel location

    # First we initialize the matrices to zero
    Mx = np.zeros((img.shape[1], img.shape[0]))
    My = np.zeros((img.shape[1], img.shape[0]))
    Mxy = np.zeros((img.shape[1], img.shape[0]))

    # Now we do the summation with a window size of 3 (i-1:i+2, j-1:j+2)
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            Mx[j][i] = np.sum(grads_X_blurred[i-1:i+2, j-1:j+2])
            My[j][i] = np.sum(grads_Y_blurred[i-1:i+2, j-1:j+2])
            Mxy[j][i] = np.sum(grads_XY_blurred[i-1:i+2, j-1:j+2])


    # Compute Harris response function
    # that is (determinant - k*trace^2)
    det = Mx * My - Mxy*Mxy
    trace = Mx + My
    C = det - k * trace**2


    # Detection with threshold
    corners = []
    # get whether a position is maximum in its 3x3 window as a boolean matrix
    maxes = ndi.maximum_filter(C, size=3) == C

    # The given default threshold was way too low for me, and while I'm not
    # sure why this could be, I think it might be because of the Sobel filters?
    threshold = thresh*(10**5)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # if harris response larger than threshold and it's the maximum in its window
            if (C[j, i]>threshold and maxes[j, i]):
                corners.append([j, i])

    return np.array(corners), C
