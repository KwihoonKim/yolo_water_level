# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 22:00:30 2024

0. 본 코드는 정사각형 표척으로부터 projective transform matrix를 추출하기 위한
   좌표를 추출해줌
   input: 
   output:
       
@author: Kwihoon
"""

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import math

def edge_coord (seg_img):
    'input: segmented mask image'
    'output: edge coordinates'
    'ex) right_coords: [82, 275], [83, 275], [84, 275], ...'
    'ex) left_coords: [82, 230], [83, 227], [84, 224], ...'
    
    right_coords = []
    left_coords = []
    
    row_seg_img = seg_img
    
    for i, array in enumerate(row_seg_img):
        j = np.where(array == 255)
        
        jj = list(j[0])
        
        if not jj:
            pass
        
        else:
            right = max(j[0])
            right_coord = [right, i]
            right_coords.append(right_coord)
            
            left = min(j[0])
            left_coord = [left, i]
            left_coords.append(left_coord)
        
    upper_coords = []
    lower_coords = []
    
    col_seg_img = seg_img.T

    for i, array in enumerate(col_seg_img):
        j = np.where(array == 255)
        
        jj = list(j[0])
        
        if not jj:
            pass
        
        else:
            right = max(j[0])
            lower_coord = [i,right]
            lower_coords.append(lower_coord)
            
            left = min(j[0])
            upper_coord = [i,left]
            upper_coords.append(upper_coord)
    
    return right_coords, left_coords, upper_coords, lower_coords
    


def knee_finder (coords):
    # 첫 포인트와 마지막 포인트의 직선에서 가장 멀리 위치한 곳이 knee point
    x1, y1 = coords[0]
    x2, y2 = coords[-1]
    
    m = (y2 - y1) / (x2 - x1)
    n = y1 - m * x1
    
    del coords[-1]
    del coords[0]
    
    dist = []

    for i in coords:
        x3, y3 = i
        d = abs(x3 - y3/m + n / m) / math.sqrt(1+1/math.pow(m,2))
        dist.append(d)
        
    knee_pos = dist.index(max(dist))
    
    return knee_pos


def revised_list (array):
    # 양쪽 10% 잘라서 버리기
    cut = int(len(array)/10)
    
    for i in range(cut):
        del array[-1]
        del array[0]
        
    return array


def linest (array):
    array = np.array(array)
    array1 = array.T[0]
    array1 = np.vstack([array1, np.ones(len(array1))]).T
    
    array2 = array.T[1]
    results = np.linalg.lstsq(array1, array2, rcond = None)
    
    return results[0]


def intersection (array1, array2):
    a, b = array1
    c, d = array2
    
    x = (-b + d) / (a - c)
    y = a * (-b + d) / (a - c) + b
    
    x = round(x, 0)
    y = round(y, 0)

    x = int(x)
    y = int(y)
    
    return (x, y)

'main 부분'
'============================================================================'
# =============================================================================
# coords_right, coords_left, coords_upper, coords_lower = edge_coord(test_img)
# 
# x_right = knee_finder(right_coords)
# x_left = knee_finder(left_coords)
# 
# pre_right = right_coords[:x_right]
# post_right = right_coords[x_right:]
# pre_left = left_coords[:x_left]
# post_left = left_coords[x_left:]
# 
# pre_right = revised_list(pre_right)
# post_right = revised_list(post_right)
# pre_left = revised_list(pre_left)
# post_left = revised_list(post_left)
# 
# array_pre_right = linest(pre_right)
# array_post_right = linest(post_right)
# array_pre_left = linest(pre_left)
# array_post_left = linest(post_left)
# 
# intersection1 = intersection(array_pre_right, array_pre_left)
# intersection2 = intersection(array_pre_left, array_post_left)
# intersection3 = intersection(array_pre_right, array_post_right)
# intersection4 = intersection(array_post_left, array_post_right)
# 
# color = (255, 255, 255)
# cv2.circle(img_seg, intersection1, 5, color)
# cv2.circle(img_seg, intersection2, 5, color)
# cv2.circle(img_seg, intersection3, 5, color)
# cv2.circle(img_seg, intersection4, 5, color)
# =============================================================================

def divide_coord (coords):
    x1 = knee_finder(coords)
    x2 = np.shape(coords)[0] - x1
    
    if x2 > x1:
        re_coord = coords[x1:]
    else:
        re_coord = coords[:x1]

    return re_coord
    
def draw_coord (y_coord, w):

    y_coord_new = []
    for i in y_coord:
        x = i[0]
        y = i[1]
    
        if 0 < (x and y) < w:
            y_coord_new.append(i)
            
    a1 = tuple(y_coord_new[0])
    b1 = tuple(y_coord_new[-1])
    
    return [a1, b1]

def pts1_coord (seg_img):
    
    coords_right, coords_left, coords_upper, coords_lower = edge_coord(seg_img)

    re_right = divide_coord(coords_right)
    re_left = divide_coord(coords_left)
    re_upper = divide_coord(coords_upper)
    re_lower = divide_coord(coords_lower)
    re_lower = divide_coord(re_lower)

    r_right = revised_list(re_right)
    r_left = revised_list(re_left)
    r_upper = revised_list(re_upper)
    r_lower = revised_list(re_lower)

    l_right = linest(r_right)
    l_left = linest(r_left)
    l_upper = linest(r_upper)
    l_lower = linest(r_lower)
    
    intersection1 = intersection(l_left, l_upper) 
    intersection2 = intersection(l_upper, l_right)
    intersection3 = intersection(l_right, l_lower)
    intersection4 = intersection(l_lower, l_left)
    
    coord = [intersection1, intersection2, intersection3, intersection4]
    
    return coord 

def matrix_det (seg_img):
    
    matrix_coord = pts1_coord (seg_img)
    
    size = 200
    ref_x = 5142 - (5257 - 5142) *0.1
    ref_x = 0    
    ref_y = 3346 - (3623 - 3346) *0.1
    ref_y = 0
    
    pts1 = np.float32(matrix_coord)
    pts2 = np.float32([[ref_x, ref_y], [ref_x + size, ref_y], [ref_x + size, ref_y + size], [ref_x, ref_y + size]]) # 변환 후 4개 좌표

    mtrx = cv2.getPerspectiveTransform(pts1, pts2) # 변환 행렬 계산 
    
# =============================================================================
#     result = cv2.warpPerspective(seg_img, mtrx, (500, 500))
#     plt.imshow(result)
# =============================================================================

    return mtrx


    
    



