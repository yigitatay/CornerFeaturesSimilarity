import os
import cv2
import tqdm

import numpy as np
import sys
# from util.extract_harris import extract_harris
from util.extract_descriptors import filter_keypoints, extract_patches
from util.match_descriptors import match_descriptors
from util.util import plot_image_pair_with_matches, reshape_imgs
import time

# constants
# HARRIS_SIGMA = 2
# HARRIS_K = 0.04
# HARRIS_THRESH = 1e-5
MATCHING_RATIO_TEST_THRESHOLD = 0.8

def get_img_paths(DIR):
    lst = [(os.path.join(DIR, path) if 'png' in path or 'jpg' in path else print()) for path in os.listdir(DIR)]
    return [i for i in lst if i is not None]

def get_harris(img):
    return np.transpose(cv2.cornerHarris(img, 2, 9, 0.2))

def main_matching(IMG1, IMG2):

    # Corner Detection
    img1_gray = cv2.imread(IMG1, cv2.IMREAD_GRAYSCALE)
    img2_gray = cv2.imread(IMG2, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(IMG1)
    img2 = cv2.imread(IMG2)
    img1, img2 = reshape_imgs(img1, img2) ##reshapes to the larger dimension of the smaller image (in terms of number of pixels)
    img1_gray, img2_gray = reshape_imgs(img1_gray, img2_gray)

    # HARRIS EXTRACTION IS THE MAIN TIME-CONSUMING PART
    # corners1_prev, C1 = extract_harris(img1, HARRIS_SIGMA, HARRIS_K, HARRIS_THRESH)
    # corners2, C2 = extract_harris(img2, HARRIS_SIGMA, HARRIS_K, HARRIS_THRESH)

    corners1 = get_harris(np.float32(img1_gray))
    corners2 = get_harris(np.float32(img2_gray))

    corners1 = np.transpose((corners1 > 0.004 * corners1.max()).nonzero())
    corners2 = np.transpose((corners2 > 0.004 * corners2.max()).nonzero())

    # print(corners1.shape)
    # print(corners2.shape)

    # Extract descriptors
    corners1 = filter_keypoints(img1_gray, corners1, patch_size=9)
    desc1 = extract_patches(img1, corners1, patch_size=9)

    corners2 = filter_keypoints(img2_gray, corners2, patch_size=9)
    desc2 = extract_patches(img2, corners2, patch_size=9)

    # Matching
    # matches_mutual = match_descriptors(desc1, desc2, "mutual")
    
    # tm = time.time()
    matches_ratio = match_descriptors(desc1, desc2, "ratio", ratio_thresh=MATCHING_RATIO_TEST_THRESHOLD)
    # print(time.time()-tm)

    ## FIND MATCHES WHERE THE SLOPE IS VERY DIFFERENT FROM THE MEAN SLOPE (SO TAKE OUT MISMATCHES)
    slopes = []
    for _, match in enumerate(matches_ratio):
        corner1 = corners1[match[0]]
        corner2 = corners2[match[1]]
        slope = (corner2[1]-corner1[1])/(corner2[0]/corner1[0])
        slopes.append(slope)
    mean_slope = sum(slopes)/len(slopes)
    slopes = np.array(slopes)
    matches_ratio = matches_ratio[np.reshape(np.argwhere(slopes<(mean_slope+15.0)), (-1))]
    slopes = slopes[slopes<(mean_slope+15.0)]
    matches_ratio = matches_ratio[np.reshape(np.argwhere(slopes>=(mean_slope-15.0)), (-1))]    
    # plot_image_pair_with_matches("test.png", img1_gray, corners1, img2_gray, corners2, matches_ratio)
    return (matches_ratio, IMG1, IMG2)



if __name__ == "__main__":
    DATASET_PATHS = ['../Datasets/ZuBuD_Dataset/1000city/qimage/', '../Datasets/window_instance_segmentation_datasets/val/images', '../Datasets/window_instance_segmentation_datasets/train/images',
                     '../Datasets/ZuBuD_Dataset/png_ZuBuD/', '../Datasets/CMP_Dataset/testA/', '../Datasets/CMP_Dataset/trainA/', '../Datasets/Kuba_Dataset/images/']
    PATH_DATA = DATASET_PATHS[0]
    PATH_COMP = DATASET_PATHS[1]
    imgs_data = get_img_paths(PATH_DATA)
    imgs_comp = get_img_paths(PATH_COMP)

    imgs_data.sort()
    imgs_comp.sort()

    with open('qimage--wisd_val.txt', 'a') as f:
        count = 0
        for img_data in tqdm.tqdm(imgs_data):
        # img_data = './qimg0052.jpg'
            similars = []
            for img_comp in imgs_comp:
                try:
                    matches = main_matching(img_data, img_comp)
                except:
                    continue
                if len(matches[0]) > 5:
                    count += 1
                    similars.append((matches[2][61:], len(matches[0])))
            f.write(f"img 1: {matches[1][42:]}, img2: {similars}\n")
    f.close()



    # TEST WITH SAMPLE IMAGES
    # sample_data = "./qimg0004.jpg"
    # sample_comp = "./cmp_x0196.jpg"
    # matches = main_matching(sample_data, sample_comp)
    # print(len(matches[0]))
