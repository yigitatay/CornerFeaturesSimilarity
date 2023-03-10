import numpy as np
import scipy.spatial as spat
import time

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    ssdist = spat.distance_matrix(desc1, desc2)
    return ssdist

def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = []
    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        for i, row in enumerate(distances):
            matches.append([i, np.argmin(row)])
    elif method == "mutual":
        for i, row in enumerate(distances):
            argmn = np.argmin(row)
            if np.argmin(distances[:, argmn]) == i:
                matches.append([i, argmn])
    elif method == "ratio":
        for i, row in enumerate(distances):
            min = np.min(row)
            argmn = np.argmin(row)
            # we can partition to get the nearest neighbor to the 0th index of the array
            # so that we can get the 2nd nearest neighbor from array[1:]
            second_min = np.min(np.partition(row, 0)[1:])
            if (min / second_min < ratio_thresh):
                matches.append([i, argmn])
    else:
        raise NotImplementedError
    return np.array(matches)
