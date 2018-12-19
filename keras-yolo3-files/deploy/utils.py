#!/usr/bin/env python
## -*- coding: utf-8 -*-

#from Crypto.Cipher import AES
#from uuid import getnode
import cv2
import numpy as np
import time


#def get_current_time():
#    return int(round(time.time() * 1000))
#
#
#def get_elapsed_time(start, end):
#    return end - start
#
#
#def mac_address():
#    mac_string = hex(getnode()).replace('0x', '').replace('L', '')
#    pad_zero_mac_string = '0' * (12 - len(mac_string)) + mac_string
#    return ':'.join(pad_zero_mac_string[i:i + 2] for i in range(0, 11, 2))
#
#
#def encrypt(data, key, alg='AES', padding='*'):
#    if alg == 'AES':
#        pad_data = data + (16 - len(data) % 16) * padding
#        pad_key = key + (16 - len(key) % 16) * padding
#        cipher = AES.new(pad_key)
#        return cipher.encrypt(pad_data)
#    return data
#
#
#def decrypt(data, key, alg='AES', padding='*'):
#    if alg == 'AES':
#        pad_key = key + (16 - len(key) % 16) * padding
#        cipher = AES.new(pad_key)
#        return cipher.decrypt(data).rstrip(padding)
#    return data


def regularize_cor_cut(ori_image, xmin, ymin, xmax, ymax):
    img_height = ori_image.shape[0]
    img_width = ori_image.shape[1]
    # Mask sure all coordinates in right area

    if xmin > img_width:
        xmin = img_width
    elif xmin < 0:
        xmin = 0

    if xmax > img_width:
        xmax = img_width
    elif xmax < 0:
        xmax = 0

    if ymin > img_height:
        ymin = img_height
    elif ymin < 0:
        ymin = 0

    if ymax > img_height:
        ymax = img_height
    elif ymax < 0:
        ymax = 0

    return xmin, ymin, xmax, ymax


def ave_edge_cal(ori_image):
    gray = cv2.cvtColor(ori_image, cv2.COLOR_BGR2GRAY)
    ul = gray[:5, :5]
    ur = gray[:5, ori_image.shape[1] - 5:]
    ll = gray[ori_image.shape[0] - 5:, :5]
    lr = gray[ori_image.shape[0] - 5:, ori_image.shape[1] - 5:]
    mean = (np.sum(ul) / 25 + np.sum(ur) / 25 + np.sum(ll) / 25 + np.sum(lr) / 25) / 4
    return mean


def remove_black_edge(ori_image):
    thresh = 15
    b, g, r = cv2.split(ori_image)
    img_width = ori_image.shape[1]
    img_height = ori_image.shape[0]

    mean_pixel = ave_edge_cal(ori_image)
    if mean_pixel < 127:
        img_mask = np.array(((b < mean_pixel+thresh) *
                             (g < mean_pixel+thresh) *
                             (r < mean_pixel+thresh)) * 1, dtype=np.uint8)
    else:
        img_mask = np.array(((b > mean_pixel-thresh) *
                             (g > mean_pixel-thresh) *
                             (r > mean_pixel-thresh)) * 1, dtype=np.uint8)

    halfv_h_start = np.sum(img_mask[:, :img_width / 2], 1).reshape(img_height, 1)
    half_h_end = img_width - np.sum(img_mask[:, img_width / 2:], 1).reshape(img_height, 1)
    half_h_len = max(img_width / 2 - np.min(halfv_h_start), np.max(half_h_end) - img_width / 2)

    half_v_start = np.sum(img_mask[:img_height / 2, :], 0).reshape(img_width, 1)
    half_v_end = img_height - np.sum(img_mask[img_height / 2:, :], 0).reshape(img_width, 1)
    half_v_len = max(img_height / 2 - np.min(half_v_start), np.max(half_v_end) - img_height / 2)

    xmin, ymin, xmax, ymax = regularize_cor_cut(ori_image,
                                                int(img_width / 2 - half_h_len),
                                                int(img_height / 2 - half_v_len),
                                                int(img_width / 2 + half_h_len),
                                                int(img_height / 2 + half_v_len))
    cut_image = ori_image[ymin: ymax, xmin: xmax]
    #resize_size = (961, 961)
    #shrink_image = cv2.resize(cut_image, resize_size, interpolation=cv2.INTER_AREA)

    #return shrink_image
    return cut_image


def image_preprocess(raw_image_list):
    transformed_image_list = []
    for idx, image in enumerate(raw_image_list):
        edge_removed_image = remove_black_edge(image)  # remove the black edges
        edge_removed_image = np.asarray(edge_removed_image, dtype=np.float32)
        # transposed_image = edge_removed_image.transpose(2, 0, 1)
        # print np.sum(transposed_image)
        # transformed_image_list.append(transposed_image.copy())
        transformed_image_list.append(edge_removed_image.copy())
    return transformed_image_list


def convert_code(code):
    return code + 200


if __name__ == '__main__':
    data = 'this is a message'
    data = encrypt(data, mac_address())
    print decrypt(data, mac_address())
