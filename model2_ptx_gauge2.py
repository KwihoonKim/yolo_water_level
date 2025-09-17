# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 23:17:02 2023
version 1: 241008 작업
주요내용: 원본이미지로부터 변환판 detect

input: 48 MP 이미지
output: water level (삼각형 수위표)

@author: USESR
"""

import cv2
import numpy as np
from image_check import locate_bbox, plot_bbox, revised_bbox, sanity_check
from post_process import nms, iou, confu_mat, pr_score

from ultralytics import YOLO, checks
from torchvision import transforms
from square_transform_matrix import pts1_coord, matrix_det, edge_coord, knee_finder, revised_list, linest, intersection


def delete_gauge3 (bbox):
    
    box1 = [5153, 3385, 5257, 3589]
# =============================================================================
#     box2 = [5320, 3318, 5412, 5026]
# =============================================================================
    box3 = [4770, 3347, 5003, 4727]

    GTbbox = [box1,box3]
    
    final_bbox = []
    
    for i, box in enumerate(bbox):
        score = 0
        for j in GTbbox:
            score += iou(box, j)
            
        if score == 0:
            final_bbox.append(box)
            
    return final_bbox

def level_est (result):
    
    def dist (a,b):
        aa = np.power(a[0][0]-b[0][0], 2)
        bb = np.power(a[0][1]-b[0][1], 2)
        cc = np.power(aa+bb, 0.5)
        return cc
    
    result = np.array(result, dtype = 'uint8')
    corners = cv2.goodFeaturesToTrack(result, 50, 0.01, 5, blockSize = 3, useHarrisDetector=True, k=0.03)
    
    if corners is None:
        level = -999

    else:       
        adds = []
        diffs = []

        for i in corners:
            i = i[0]
            add = i[0] + i[1]
            diff = i[0] - i[1]
            adds.append(add)
            diffs.append(diff)

        a3 = adds.index(max(adds))
        a1 = adds.index(min(adds))
        a2 = diffs.index(max(diffs))
        a4 = diffs.index(min(diffs))

        a = corners[a1]
        b = corners[a2]
        c = corners[a3]
        d = corners[a4]
            
        dist1 = dist(a, b)
        dist2 = dist(c, d)
    
        dist_w = dist1 + dist2
    
        dist3 = dist(b, c)
        dist4 = dist(a, d)
        dist_h = dist3 + dist4
    
        level = round(1 - 0.1 * (dist_h / dist_w), 2)
    
    return level

def level_ptx_gauge2 (img_path):
    transform_img = transforms.ToPILImage()
    
    img = cv2.imread(img_path)

    '1.detect'
    model1_detect = YOLO('model1_detect/best52.pt')
    results_detect = model1_detect.predict(img)
    box  = results_detect[0].boxes.xyxy.tolist()
    conf = results_detect[0].boxes.conf.tolist()
    
    answer, new_bbox, new_conf = nms(box, conf)
    final_bbox = delete_gauge3(new_bbox)

    if final_bbox:
    
        region = revised_bbox(final_bbox)[0]
    
        '2.crop image'
        img_crop = img[region[1]:region[3], region[0]:region[2], :]

        if img_crop.size == 0:
            level = -999
            return level

        h11, w11, z11 = np.shape(img_crop)
        new_crop = cv2.resize(img_crop, (640, 640))
        
        max_value = np.max(new_crop)
        new_crop = np.array(new_crop / max_value*255, dtype = 'uint8')
        
        gray = cv2.cvtColor(new_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        gray = cv2.bitwise_not(gray)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        '4. segment'
        model1_segment = YOLO('model2_segment/best.pt')
        segment = model1_segment.predict(gray, conf = 0.8)
        mask_crop = segment[0].masks
        
        if mask_crop == None:
            level = -999

        else:
            mask_crop = mask_crop.data[0]
            mask_crop = transform_img(mask_crop)
            mask_crop = np.array(mask_crop)

            '5. resize image'
            crop_mask = cv2.resize(mask_crop, (w11, h11))
    
            final_bbox_in = final_bbox[0]
            space_height = (final_bbox_in[3] - final_bbox_in[1]) * 0.5
            space_width = (final_bbox_in[2] - final_bbox_in[0]) * 0.5
            
            new_x1 = int(final_bbox_in[0] - space_width)
            new_y1 = int(final_bbox_in[1] - space_height)
            new_x2 = int(final_bbox_in[2] + space_width)
            new_y2 = int(final_bbox_in[3] + space_height)
            
            '6. insert in raw image'
            h2, w2, z2 = np.shape(img)
            img_new = np.zeros((h2, w2))
            img_new[new_y1:new_y2, new_x1:new_x2] = crop_mask
            img_new = np.array(img_new, dtype = 'uint8')
    
            level = level_est(img_new)
    
    else:
        level = -999

    return level

# =============================================================================
# "ploting 5.3.2"
# plt.imshow(result)
# test = mask_crop2[:,:163]
# test = np.expand_dims(test, axis = 2)
# test_rgb = np.append(test,test,2)
# test_rgb = np.append(test_rgb,test,2)
# new_img = np.array(img)
# new_img[region2[1]:region2[3], region2[0]:region2[2], :] = test_rgb
# cv2.imwrite('test1.jpg', result)
# 
# mtrx = matrix(img_path)
# result = cv2.warpPerspective(new_img, mtrx, (2000, 2000))
# =============================================================================

