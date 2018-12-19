import os
import numpy as np
import cv2

from deploy_frozen_yolov3_model import DiscDetection 


def compute_iou(image_size, box1, box2):
    """compute the iou of two boxes
    
    args:
        image_size - (height, width)
        box1, box2 - [top, left, bottom, right]

    return:
        intersection over union
    """
 
    image = np.zeros(image_size, np.int)
    image[box1[0]:box1[2], box1[1]:box1[3]] += 1
    image[box2[0]:box2[2], box2[1]:box2[3]] += 1
    #cv2.imwrite('union.png', image*255)

    intersec= image.flatten().tolist().count(2)
    union = np.count_nonzero(image>0)
    print intersec, union

    return intersec * 1.0 / union

    





def main(val_list_file):

    detector = DiscDetection()
    detector.load_network()
    with open(val_list_file, 'r') as vf:
        val_list = vf.readlines()

    ious = []
    no_num = 0
    yes_num = 0
    for line in val_list:
        imgfile, label = line.split()
        left, top, right, bottom, _ = map(int, label.split(','))
        box_truth = [top, left, bottom, right]

        box = detector.get_disc_box(imgfile)
        if box is None:
            no_num += 1
        else:
            iou = compute_iou(detector.image_size, box_truth, box)
            ious.append(iou)

            if iou > 0.5:
                yes_num += 1
            else:
                no_num += 1

        print('running iou: {}'.format(iou))

    print('mean iou: {}'.format(np.mean(ious)))
    print('detection rate: {} ({}/{})'.format(float(yes_num)/(no_num+yes_num), yes_num, no_num+yes_num))


if __name__ == '__main__':

    main('../lijian_work/test.txt')






