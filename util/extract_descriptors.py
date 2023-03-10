import numpy as np

def filter_keypoints(img, keypoints, patch_size = 9):
    filtered_keypoints = []
    for keypoint in keypoints:
        if ((keypoint[0]-patch_size)<0 or (keypoint[0]+patch_size)>=img.shape[0] or \
            (keypoint[1]-patch_size)<0 or (keypoint[1]+patch_size)>=img.shape[1]):
            continue
        filtered_keypoints.append(keypoint)
    return np.array(filtered_keypoints)

# Could change this to histogram of oriented gradients
def extract_patches(img, keypoints, patch_size = 9):
    '''
    Extract local patches for each keypoint
    Inputs:
    - img:          (h, w) gray-scaled images
    - keypoints:    (q, 2) numpy array of keypoint locations [x, y]
    - patch_size:   size of each patch (with each keypoint as its center)
    Returns:
    - desc:         (q, patch_size * patch_size) numpy array. patch descriptors for each keypoint
    '''
    h, w = img.shape[0], img.shape[1]
    img = img.astype(float) / 255.0
    img_r = img[:, :, 2]
    img_g = img[:, :, 1]
    img_b = img[:, :, 0]
    offset = int(np.floor(patch_size / 2.0))
    ranges = np.arange(-offset, offset + 1)
    desc_r = np.take(img_r, ranges[:,None] * w + ranges + (keypoints[:, 1] * w + keypoints[:, 0])[:, None, None]) # (q, patch_size, patch_size)
    desc_r = desc_r.reshape(keypoints.shape[0], -1) # (q, patch_size * patch_size)
    desc_g = np.take(img_g, ranges[:,None] * w + ranges + (keypoints[:, 1] * w + keypoints[:, 0])[:, None, None]) # (q, patch_size, patch_size)
    desc_g = desc_g.reshape(keypoints.shape[0], -1) # (q, patch_size * patch_size)
    desc_b = np.take(img_b, ranges[:,None] * w + ranges + (keypoints[:, 1] * w + keypoints[:, 0])[:, None, None]) # (q, patch_size, patch_size)
    desc_b = desc_b.reshape(keypoints.shape[0], -1) # (q, patch_size * patch_size)
    return np.concatenate((desc_r, desc_g, desc_b), axis=1)
