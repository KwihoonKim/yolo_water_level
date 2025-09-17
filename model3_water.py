# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 23:17:02 2023
version 1: 240722 작업

@author: USESR
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from image_check import locate_bbox, plot_bbox, revised_bbox, sanity_check
from post_process import nms, iou, confu_mat, pr_score

from ultralytics import YOLO, checks
from torchvision.utils import draw_bounding_boxes
from torchvision import transforms
from square_transform_matrix import pts1_coord, matrix_det, edge_coord, knee_finder, revised_list, linest, intersection
import os
from torchvision import transforms
from model1_matrix import matrix_gt

def delete_gauge1 (bbox):
    
# =============================================================================
#     box1 = [5153, 3385, 5257, 3589]
# =============================================================================
    box2 = [5320, 3318, 5412, 5026]
    box3 = [4770, 3347, 5003, 4727]

    GTbbox = [box2,box3]
    
    final_bbox = []
    
    for i, box in enumerate(bbox):
        score = 0
        for j in GTbbox:
            score += iou(box, j)
            
        if score == 0:
            final_bbox.append(box)
            
    return final_bbox

def level_water (img_path, mtrx):
    
    transform_img = transforms.ToPILImage()
    
    img = cv2.imread(img_path)

    '### Plate Detection and Segmentation ###'
    '========================================================================='
    '1.detect'
    model1_detect = YOLO('model1_detect/best52.pt')
    results_detect = model1_detect.predict(img, conf = 0.2)
    box  = results_detect[0].boxes.xyxy.tolist()
    conf = results_detect[0].boxes.conf.tolist()

    answer, new_bbox, new_conf = nms(box, conf)
    
    final_bbox = delete_gauge1(new_bbox)
    
    if final_bbox:
    
        region = revised_bbox(final_bbox)[0]
    
        '1. crop image'
        img_crop = img[region[1]:region[3], region[0]:region[2], :]
        h11, w11, z11 = np.shape(img_crop)
        new_crop = cv2.resize(img_crop, (640, 640))

        max_value = np.max(new_crop)
        new_crop = np.array(new_crop / max_value*255, dtype = 'uint8')
        
        gray = cv2.cvtColor(new_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        gray = cv2.bitwise_not(gray)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        '2. segment'
        model1_segment = YOLO('model1_segment/best.pt')
        results_segment = model1_segment.predict(gray, conf = 0.7)
        

        if results_segment[0].masks == None:
            mtrx = np.array([])
        
        else:
            mask_crop = results_segment[0].masks.data
            mask_crop = mask_crop[0]
            mask_crop = transform_img(mask_crop)
            mask_crop = np.array(mask_crop)

            crop_mask = cv2.resize(mask_crop, (w11, h11))
            '========================================================================='

            '4. coordinates'
            contours, hierachy = cv2.findContours(crop_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            largest_areas = sorted(contours, key = cv2.contourArea, reverse = True)
            contour = largest_areas[0]

            hull = cv2.convexHull(contour)

            new_mask = np.zeros((h11, w11))
            cv2.fillPoly(new_mask,[hull],1)            
            new_mask = np.array(new_mask*255, dtype = 'uint8')

    else:
        level = -999
        return level

    '========================================================================='
    
    '### Water Segmentation'
    '========================================================================='
    
    h21, w21, _ = img.shape
    
    '2. resize (raw_to_640)'
    img640 = cv2.resize(img, (640, 384))

    '3. segment'
    model3_segment = YOLO('model3_water/best56.pt')
    results_segment3 = model3_segment.predict(img640, conf = 0.3)
    
    if results_segment3[0].masks == None:
        level = -200
    
    else:
        mask_water = results_segment3[0].masks.data[0]
        mask_water = transform_img(mask_water)
        mask_water = np.array(mask_water)
        
        '4. resize (640_to_raw)'
        mask_water = cv2.resize(mask_water, (w21, h21))

        
        '====================================================================='
    
        '### Merge image'
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
        img_new[new_y1:new_y2, new_x1:new_x2] = new_mask
        img_new = np.where(mask_water == 0, img_new, mask_water)

        '5. perspective_transformation'
        result = cv2.warpPerspective(img_new, mtrx, (2000, 2000))

        '6. ROI set'
        result_roi = result[1200:, 1000:1100]
        result_roi = np.where(result_roi > 127, 255, 0)
        result_roi = np.transpose(result_roi)
        
        '7. level estimate'
        jj = []

        for i, array in enumerate(result_roi):
    
            y_coords = np.where(array == 255)[0]
            
            if list(y_coords):
                y_coord = min(y_coords)
                jj.append(y_coord)
            else:
                jj.append(800)
                
        level = (800 - sum(jj) / len(jj)) / 1000
        level = round(level, 3)
        
    return level

# =============================================================================
# '5.3.3 figure'
# plt.imshow(show_img)
# show_img = np.where(test_rgb == 255, test_rgb, img)
# 
# test = final_img
# test = np.expand_dims(test, axis = 2)
# test_rgb = np.append(test,test,2)
# test_rgb = np.append(test_rgb,test,2)
# 
# result = cv2.warpPerspective(show_img, mtrx, (2000, 2000))
# cv2.imwrite('test.jpg', result)
# =============================================================================



