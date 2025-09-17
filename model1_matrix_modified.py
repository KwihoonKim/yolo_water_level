# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 23:17:02 2023
version 1: 241008 작업
주요내용: 원본이미지로부터 변환판 detect

***model1_matrix_modified에서 300*100 변환판으로 좌표를 바꿔준 파일임***

input: 48 MP 이미지
output: water level (삼각형 수위표)

@author: USESR
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from image_check import locate_bbox, plot_bbox, revised_bbox, sanity_check
from post_process import nms, nms_seg, iou, confu_mat, pr_score

from ultralytics import YOLO, checks
from torchvision.utils import draw_bounding_boxes
from torchvision import transforms
import os

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

def delete_gauge2 (bbox):
    
    box1 = [5153, 3385, 5257, 3589]
# =============================================================================
#     box2 = [5320, 3318, 5412, 5026]
#     box3 = [4770, 3347, 5003, 4727]
# =============================================================================

    GTbbox = [box1]
    
    final_bbox = []
    
    for i, box in enumerate(bbox):
        score = 0
        for j in GTbbox:
            score += iou(box, j)
            
        if score == 0:
            final_bbox.append(box)
            
    return final_bbox

def matrix_gt ():
    '4. perspective transform'
    coord1 = (5149, 3401)
    coord2 = (5254, 3392)
    coord3 = (5260, 3579)
    coord4 = (5155, 3593)
    
    pts1 = np.float32([[coord1, coord2, coord3, coord4]])
    pts2 = np.float32([[1000, 1000], [1100, 1000], [1100, 1100], [1000, 1100]]) # 변환 후 4개 좌표
    mtrx = cv2.getPerspectiveTransform(pts1, pts2) # 변환 행렬 계산 
    
    return mtrx

def matrix_gt300 ():
    '4. perspective transform'
    coord1 = (5146, 3397)
    coord2 = (5454, 3372)
    coord3 = (5458, 3567)
    coord4 = (5148, 3602)
    
    pts1 = np.float32([[coord1, coord2, coord3, coord4]])
    pts2 = np.float32([[1000, 1000], [1300, 1000], [1300, 1100], [1000, 1100]]) # 변환 후 4개 좌표
    mtrx = cv2.getPerspectiveTransform(pts1, pts2) # 변환 행렬 계산 

    return mtrx

def matrix_c3_m (img_path):
    transform_img = transforms.ToPILImage()
    
    img = cv2.imread(img_path)

    '### Plate Detection and Segmentation ###'
    '========================================================================='
    '1.detect'
    model1_detect = YOLO('model1_detect/best52.pt')
    results_detect = model1_detect.predict(img, conf = 0.3)
    box  = results_detect[0].boxes.xyxy.tolist()
    conf = results_detect[0].boxes.conf.tolist()

    answer, new_bbox, new_conf = nms(box, conf)
    
    final_bbox = delete_gauge1(new_bbox)

    if final_bbox:
        
        region = revised_bbox(final_bbox)[0]

        '1. crop image'
        img_crop = img[region[1]:region[3], region[0]:region[2], :]

        if img_crop.size == 0:
            mtrx = np.array([])
            return mtrx

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
        results_segment = model1_segment.predict(gray, conf = 0.8)

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

            corners = cv2.goodFeaturesToTrack(new_mask, 200, 0.01, 5, blockSize = 3, useHarrisDetector=True, k=0.03)
            corners = np.array(corners, dtype = 'int')
            
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

            ref_w = region[0]
            ref_h = region[1]
# =============================================================================
#             (-5, -4)
#             (+201, -20)
#             (+198, -12)
#             (-9, +9)
# =============================================================================
            coord1 = (a[0][0] + ref_w   -5, a[0][1] + ref_h  -4) 
            coord2 = (b[0][0] + ref_w +201, b[0][1] + ref_h -20) 
            coord3 = (c[0][0] + ref_w +198, c[0][1] + ref_h -12) 
            coord4 = (d[0][0] + ref_w   -9, d[0][1] + ref_h  +9)

            '5. perspective transform'
            pts1 = np.float32([[coord1, coord2, coord3, coord4]])
            pts2 = np.float32([[1000, 1000], [1300, 1000], [1300, 1100], [1000, 1100]]) # 변환 후 4개 좌표
            mtrx = cv2.getPerspectiveTransform(pts1, pts2) # 변환 행렬 계산 

    else:
        mtrx = np.array([])

    return mtrx

def mat_coord_c3 (img_path):
    transform_img = transforms.ToPILImage()
    
    img = cv2.imread(img_path)

    '### Plate Detection and Segmentation ###'
    '========================================================================='
    '1.detect'
    model1_detect = YOLO('model1_detect/best52.pt')
    results_detect = model1_detect.predict(img, conf = 0.3)
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
        results_segment = model1_segment.predict(gray, conf = 0.5)

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

            corners = cv2.goodFeaturesToTrack(new_mask, 200, 0.01, 5, blockSize = 3, useHarrisDetector=True, k=0.03)
            corners = np.array(corners, dtype = 'int')
            
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
            
            ref_w = region[0]
            ref_h = region[1]
            
            coord1 = (a[0][0] + ref_w, a[0][1] + ref_h) 
            coord2 = (b[0][0] + ref_w, b[0][1] + ref_h) 
            coord3 = (c[0][0] + ref_w, c[0][1] + ref_h) 
            coord4 = (d[0][0] + ref_w, d[0][1] + ref_h)
            mat_coords = [coord1, coord2, coord3, coord4]
    else:
        mat_coords = []
        
    return mat_coords

def matrix_c3_alt (img_path):
    
    img = cv2.imread(img_path)

    '### Plate Detection and Segmentation ###'
    '========================================================================='
    '1.detect'
    model1_detect = YOLO('model1_detect/best52.pt')
    results_detect = model1_detect.predict(img, conf = 0.4)
    box  = results_detect[0].boxes.xyxy.tolist()
    conf = results_detect[0].boxes.conf.tolist()

    answer, new_bbox, new_conf = nms(box, conf)
    
    final_bbox = delete_gauge2(new_bbox)

    if new_bbox:

        coord1 = (int(final_bbox[0][0]), int(final_bbox[0][1]))
        coord2, _, _, coord3 = mat_coord_c3(img_path)
        coord4 = (int(final_bbox[1][0]), int(final_bbox[1][1]))
    
        '4. perspective transform'

        coord21 = [ 380,  925]
        coord22 = [1000, 1000]
        coord23 = [1000, 1200]
        coord24 = [1250,  925]
    
        pts1 = np.float32([[coord1, coord2, coord3, coord4]])
        pts2 = np.float32([coord21, coord22, coord23, coord24]) # 변환 후 4개 좌표
        mtrx = cv2.getPerspectiveTransform(pts1, pts2) # 변환 행렬 계산 
    
    else:
        mtrx = np.array([])

    return mtrx


