# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 22:02:42 2024

@author: USESR
"""

from model1_matrix import matrix_gt, mat_coord_c3, matrix_c3, matrix_c3_alt
from model1_matrix_modified import matrix_gt, mat_coord_c3, matrix_c3_alt, matrix_c3_m
from model2_gauge1 import level_gauge1
from model2_gauge2 import level_gauge2
from model2_ptx_gauge2 import level_ptx_gauge2
from model2_ptx_gauge1 import level_ptx_gauge1
from model3_water import level_water

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


mtrx_gt = matrix_gt()

path = 'c3/'
data_list = os.listdir(path)
data_list = data_list[:1861]
i = data_list[0]
img_path = path + i
img = cv2.imread(img_path)
plt.imshow(img)

mtrx = matrix_c3_m(img_path)
result = cv2.warpPerspective(img, mtrx, (2000, 3000))
plt.imshow(result)


'============================================================================'
levels = []
for i in data_list:
    img_path = path + i
    
    mtrx = matrix_c3_m(img_path)
    
    if mtrx.size == 0:
        level12 = -999
        level22 = -999
        level32 = -999
        level = (level12, level22, level32)
        levels.append(level)

    else:
        level12 = level_gauge1 (img_path, mtrx)
        level22 = level_gauge2 (img_path, mtrx)
        level32 = level_water(img_path, mtrx)
        level = (level12, level22, level32)
        
        levels.append(level)

        

    if mtrx.size == 0:
        level11 = level_gauge1 (img_path, mtrx_gt)
        level12 = -999
        level13 = level_ptx_gauge1 (img_path)
        
        level21 = level_gauge2 (img_path, mtrx_gt)
        level22 = -999
        level23 = level_ptx_gauge2 (img_path)
        
        level31 = level_water(img_path, mtrx_gt)
        level32 = -999
        level = (level11, level12, level13, level21, level22, level23, level31, level32)

        levels.append(level)

    else:
        level11 = level_gauge1 (img_path, mtrx_gt)
        level12 = level_gauge1 (img_path, mtrx)
        level13 = level_ptx_gauge1 (img_path)
    
        level21 = level_gauge2 (img_path, mtrx_gt)
        level22 = level_gauge2 (img_path, mtrx)
        level23 = level_ptx_gauge2 (img_path)
    
        level31 = level_water(img_path, mtrx_gt)
        level32 = level_water(img_path, mtrx)
        level = (level11, level12, level13, level21, level22, level23, level31, level32)

        levels.append(level)
    
    
    
    
    
    
    
mtrx = matrix_c3(img_path)
result = cv2.warpPerspective(img, mtrx, (2000, 3000))
plt.imshow(result)
cv2.imwrite('test1.jpg', result)
    
path = 'model1_detect/test/C4/'
data_list = os.listdir(path)
data_list = data_list[125:]
'matrix check'

mrtx_coords = []
for i in data_list:
    img_path = path + i
    mtrx = mat_coord_c3(img_path)
    
    model1_detect = YOLO('model1_detect/best52.pt')
    results_detect = model1_detect.predict(img)
    box = results_detect[0].boxes.xyxy.tolist()  
    conf = results_detect[0].boxes.conf.tolist()
    answer, new_bbox, new_conf = nms(box, conf)
    final_bbox = delete_gauge2(new_bbox)
    bbox_return = [int(final_bbox[0][0]), int(final_bbox[0][1])]
    
    mrtx_coords.append((mtrx, bbox_return))

        
            
i = data_list[190]
img_path = path + i
img = cv2.imread(img_path)
img_crop = img[region[1]:region[3], region[0]:region[2], :]

plt.imshow(img)

for i in data_list:
    img_path = path + i
    img = cv2.imread(img_path)
    
    model1_detect = YOLO('model1_detect/best52.pt')
    results_detect = model1_detect.predict(img, conf = 0.4)

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
# =============================================================================
#             
#             '4. coordinates'
#             contours, hierachy = cv2.findContours(crop_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
#             largest_areas = sorted(contours, key = cv2.contourArea, reverse = True)
#             contour = largest_areas[0]
# 
#             hull = cv2.convexHull(contour)
# 
#             new_mask = np.zeros((h11, w11))
#             cv2.fillPoly(new_mask,[hull],1)            
#             new_mask = np.array(new_mask*255, dtype = 'uint8')
# 
#             contours, hierachy = cv2.findContours(new_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
#             largest_areas = sorted(contours, key = cv2.contourArea, reverse = True)
#             contour = largest_areas[0]
#             
#             adds = []
#             diffs = []
#             
#             for j in contour:
#                 j = j[0]
#                 add = j[0] + j[1]
#                 diff = j[0] - j[1]
#                 adds.append(add)
#                 diffs.append(diff)
#                 
#             a31 = [index for index, val in enumerate(adds) if val == max(adds)]
#             a32 = [contour[index][0] for index in a31]
#             a33 = [contour[index][0][0] for index in a31]
#             a34 = [a32[a33.index(max(a33))]]
#                    
#             a11 = [index for index, val in enumerate(adds) if val == min(adds)]
#             a12 = [contour[index][0] for index in a11]
#             a13 = [contour[index][0][0] for index in a11]
#             a14 = [a12[a13.index(min(a13))]]
# 
#             a21 = [index for index, val in enumerate(diffs) if val == max(diffs)]
#             a22 = [contour[index][0] for index in a21]
#             a23 = [contour[index][0][0] for index in a21]
#             a24 = [a22[a23.index(max(a23))]]
#             
#             a41 = [index for index, val in enumerate(diffs) if val == min(diffs)]
#             a42 = [contour[index][0] for index in a41]
#             a43 = [contour[index][0][0] for index in a41]
#             a44 = [a42[a43.index(min(a43))]]
#             
#             a = a14
#             b = a24
#             c = a34
#             d = a44
#                 
#     for k in contour:
#         k = tuple(k[0])
#         cv2.circle(img_crop, k, 1, (255,0,0), 1)
#       
#     coords = [a,b,c,d]
#     
#     for k in coords:
#         k = tuple(k[0])
#         cv2.circle(img_crop, k, 5, (0,0,255), 1)
# =============================================================================
    
    cv2.imwrite('seg_plate/' + i, crop_mask)
    
    
    
    result = cv2.warpPerspective(img, mtrx, (2000, 3000))
    plt.imshow(gray)       
        
        
i = data_list[33]
img_path = path + i
img = cv2.imread(img_path)
img_crop = img[region[1]:region[3], region[0]:region[2], :]

plt.imshow(crop_mask)        
        
coord = mat_coord_c3(img_path)
        
        
"plate size 300*100"
'============================================================================='
coord1 = (5151, 3401)
coord2 = (5253, 3392)
coord3 = (5260, 3579)
coord4 = (5157, 3593)

coord1 = (5146, 3397)
coord2 = (5454, 3372)
coord3 = (5458, 3567)
coord4 = (5148, 3602)

(-5, -4)
(+201, -20)
(+198, -12)
(-9, +9)

'4. perspective transform'
# 100 * 100 plate
coord1 = (5151, 3401)
coord2 = (5253, 3392)
coord3 = (5260, 3579)
coord4 = (5157, 3593)
pts2 = np.float32([[1000, 1000], [1100, 1000], [1100, 1100], [1000, 1100]]) # 변환 후 4개 좌표

# 200 * 100 plate
coord1 = (5146, 3397)
coord2 = (5354, 3382)
coord3 = (5360, 3575)
coord4 = (5151, 3599)

pts2 = np.float32([[1000, 1000], [1200, 1000], [1200, 1100], [1000, 1100]]) # 변환 후 4개 좌표

# 300 * 100 plate
coord1 = (5146, 3397)
coord2 = (5454, 3372)
coord3 = (5458, 3567)
coord4 = (5148, 3602)
pts2 = np.float32([[1000, 1000], [1300, 1000], [1300, 1100], [1000, 1100]]) # 변환 후 4개 좌표

# 100 * 200 plate
coord1 = (5149, 3401)
coord2 = (5254, 3392)
coord3 = (5268, 3778)
coord4 = (5161, 3798)
pts2 = np.float32([[1000, 1000], [1100, 1000], [1100, 1200], [1000, 1200]]) # 변환 후 4개 좌표

# 100 * 300 plate
coord1 = (5149, 3401)
coord2 = (5253, 3392)
coord3 = (5280, 3992)
coord4 = (5176, 4016)
pts2 = np.float32([[1000, 1000], [1100, 1000], [1100, 1300], [1000, 1300]]) # 변환 후 4개 좌표

pts1 = np.float32([[coord1, coord2, coord3, coord4]])
mtrx = cv2.getPerspectiveTransform(pts1, pts2) # 변환 행렬 계산 
result = cv2.warpPerspective(img, mtrx, (2000, 3000))
plt.imshow(result)     


levels = []
for i in range(7):
    i = i - 3

    for j in range(7):
        j = j - 3
        
        coord1 = (5151, 3401)
        coord2 = (5253, 3392)
        coord3 = (5260+i, 3579+j)
        coord4 = (5157, 3593)
        
        pts2 = np.float32([[1000, 1000], [1100, 1000], [1100, 1100], [1000, 1100]]) # 변환 후 4개 좌표

        pts1 = np.float32([[coord1, coord2, coord3, coord4]])
        mtrx = cv2.getPerspectiveTransform(pts1, pts2) # 변환 행렬 계산
        
        level12 = level_gauge1 (img_path, mtrx)
        level22 = level_gauge2 (img_path, mtrx)
        level32 = level_water (img_path, mtrx)

        level = (i, j, level12, level22, level32)
        levels.append(level)


coord1 = (5151, 3401)
coord2 = (5253, 3392)
coord3 = (5260, 3579)
coord4 = (5157, 3593)
pts2 = np.float32([[1000, 1000], [1100, 1000], [1100, 1100], [1000, 1100]]) # 변환 후 4개 좌표

pts1 = np.float32([[coord1, coord2, coord3, coord4]])
mtrx = cv2.getPerspectiveTransform(pts1, pts2) # 변환 행렬 계산
level12 = level_gauge1 (img_path, mtrx)
level22 = level_gauge2 (img_path, mtrx)
result = cv2.warpPerspective(img, mtrx, (2000, 3000))
plt.imshow(result)    




img = cv2.imread(img_path)

plt.imshow(img[3200:4200, 5000:6000, :])        
cv2.line(img, coord1, coord2, (255,0,0), thickness = 3)
cv2.line(img, coord2, coord3, (255,0,0), thickness = 3)
cv2.line(img, coord3, coord4, (255,0,0), thickness = 3)
cv2.line(img, coord4, coord1, (255,0,0), thickness = 3)


crop_img2 = img[3200:4200, 5000:6000, :]
cv2.imwrite('test1.jpg', img)        
        
'============================================================================='
'24. 12. 24'
img = cv2.imread(img_path)
'### Plate Detection and Segmentation ###'
'1.detect'
model1_detect = YOLO('model1_detect/best52.pt')
results_detect = model1_detect.predict(img, conf = 0.3)
box  = results_detect[0].boxes.xyxy.tolist()
conf = results_detect[0].boxes.conf.tolist()

answer, new_bbox, new_conf = nms(box, conf)
    
final_bbox = delete_gauge1(new_bbox)

region = revised_bbox(final_bbox)[0]

'1. crop image'
img_crop = img[region[1]:region[3], region[0]:region[2], :]

from torchvision import transforms
transform_img = transforms.ToPILImage()
        
mask = results_detect[0].masks.data        
mask = mask[2]
mask = transform_img(mask)
mask = np.array(mask)

h11, w11, z11 = np.shape(img)
mask = cv2.resize(mask, (w11, h11))

mask1 = np.expand_dims(mask, axis = 2)
mask2 = np.append(mask1, mask1, axis = 2)
mask3 = np.append(mask2, mask1, axis = 2)
new_img = np.where(mask3 < 100, img, 0)

plt.imshow(new_img)

img_crop = img[3325:3644, 5167:5322, :]
plt.imshow(img_crop1)

img_crop_raw = new_img[3325:3644, 5167:5322, :]
plt.imshow(img_crop_raw)

cv2.imwrite('test1.jpg', new_img)
cv2.imwrite('test.jpg', img_crop_raw)

new_mask1 = np.expand_dims(new_mask, axis = 2)
new_mask2 = np.append(new_mask1, new_mask1, axis = 2)
new_mask3 = np.append(new_mask2, new_mask1, axis = 2)

new_mask_img = np.where(new_mask3 < 128, img_crop, 0)

plt.imshow(new_mask_img)
cv2.imwrite('test2.jpg', new_mask_img)
'============================================================================='

'============================================================================='
'24. 12. 24'
img = cv2.imread(img_path)
'### Plate Detection and Segmentation ###'
'1.detect'
model1_detect = YOLO('model1_detect/best52.pt')
results_detect = model1_detect.predict(img, conf = 0.3)
box  = results_detect[0].boxes.xyxy.tolist()
conf = results_detect[0].boxes.conf.tolist()

answer, new_bbox, new_conf = nms(box, conf)
    
final_bbox = delete_gauge2(new_bbox)

region = revised_bbox(final_bbox)[0]

'1. crop image'
img_crop = img[region[1]:region[3], region[0]:region[2], :]

img_crop1 = img[3200:, 4687:5176, :]
img_crop2 = img[3200:, 5290:5550, :]

from torchvision import transforms
transform_img = transforms.ToPILImage()
        
mask = results_detect[0].masks.data        
mask = mask[1]
mask = transform_img(mask)
mask = np.array(mask)


h11, w11, z11 = np.shape(img)
mask = cv2.resize(mask, (w11, h11))
plt.imshow(mask)

mask1 = np.expand_dims(mask, axis = 2)
mask2 = np.append(mask1, mask1, axis = 2)
mask3 = np.append(mask2, mask1, axis = 2)

plt.imshow(mask3)
img_crop2 = img[3200:, 5290:5550, :]
mask_crop2 = mask3[3200:, 5290:5550, :]


new_img = np.where(mask_crop2 < 100, img_crop2, 0)
cv2.imwrite('test2.jpg', new_img)


plt.imshow(new_img)

new_mask_img = np.where(new_mask3 < 128, img_crop, 0)


new_mask_img = np.where(new_mask3 < 128, img_crop, 0)


img_crop_raw = new_img[3325:3644, 5167:5322, :]

new_mask = img_new
new_mask1 = np.expand_dims(new_mask, axis = 2)
new_mask2 = np.append(new_mask1, new_mask1, axis = 2)
new_mask3 = np.append(new_mask2, new_mask1, axis = 2)

new_img = np.where(new_mask3 < 128, img, 0)

plt.imshow(new_img)
cv2.imwrite('test2.jpg', new_mask_img)
cv2.imwrite('test1.jpg', new_img)
cv2.imwrite('test.jpg', new_img)
'============================================================================='        
crop_mask1 = crop_mask
img_crop1 = img_crop      
crop_mask2 = crop_mask
img_crop2 = img_crop      
plt.imshow(crop_mask2)
new_mask_img = np.where(new_mask3 < 128, img_crop, 0)

new_mask
        