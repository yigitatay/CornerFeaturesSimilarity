import cv2
import numpy as np
import os
import sys
import ast
from util.util import reshape_imgs

if __name__ == "__main__":
    first_dir = '../Datasets/ZuBuD_Dataset/1000city/qimage/'
    second_dir = '../Datasets/window_instance_segmentation_datasets/val/images/'
    move_to = '../Datasets/duplicates/qimage--wisd_val/'
    with open('qimage--wisd_val.txt', 'r') as file:
        lines = file.readlines()
        total_same = 0
        for line in lines:
            line = line.split(':') 
            img_1 = first_dir + line[1].split(',')[0][1:]
            img_2_list = ast.literal_eval(line[2][1:])
            img2_names = [item[0] for item in img_2_list]
            img2_sims = [item[1] for item in img_2_list]
            img_2_list = [second_dir+item[0] for item in img_2_list]

            try:
                img1 = cv2.imread(img_1)
            except:
                continue
            img_lst = []
            for img in img_2_list:
                try:
                    img_read = cv2.imread(img)
                    _img_1, _img_2 = reshape_imgs(img1, img_read)
                    img_lst.append((_img_1, _img_2))
                except:
                    continue
            for (img1, img2), img2_name, img2_sim in zip(img_lst, img2_names, img2_sims):
                hor_imgs = np.hstack((img1, img2))
                cv2.imshow(f'COMPARE__{img2_sim}', hor_imgs)
                key = cv2.waitKey(0)
                if key == 27 or key == 113: ## esc or q, exit
                    sys.exit(0)
                elif key == 100: ### PRESS d TO MOVE THE SECOND IMAGE FROM ITS DIRECTORY TO A DIFFERENT ONE
                    os.rename(second_dir+img2_name, move_to+img2_name)
                    total_same += 1
                else: ## Any other key, keep going
                    continue
        print(f"TOTAL NUMBER OF SAME OR SIMILAR IMAGES: {total_same}. This is {total_same/len(lines)*100.0}% of images in the dataset.")

        