# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 00:35:07 2024

@author: USESR
"""


import cv2
import matplotlib.pyplot as plt


def sanity_check(image, mask):
# =============================================================================
#     i = random.randrange(test_image.shape[0])
#     test_img = test_image[i]
#     predicted_mask = predicted_test[i]
# =============================================================================
    plt.figure(figsize=(16, 8))
    
    plt.subplot(121)
    plt.title('Testing Image')
    plt.imshow(image/256.)
    
    plt.subplot(122)
    plt.title('Predicted Label')
    plt.imshow(mask, cmap='gray')
    

def locate_bbox (results):
    
    boxes = results[0].boxes.xyxy.tolist()
    
    bbox = []
    for i, box in enumerate(boxes):
        globals()['box{}'.format(i)] = []
        
        for j in box:
            new_coord = int(j)
            globals()['box{}'.format(i)].append(new_coord)
        
        bbox.append(globals()['box{}'.format(i)])
        
    return bbox

def plot_bbox (img, bbox):
    
    # num = 0: circle
    num = 1  #: rectangle
    
    size = 10
    color = (255, 0, 0)
    thickness = 5
    
    boxes = bbox
    
    for i, box in enumerate(boxes):
        globals()['box{}'.format(i)] = []
        
        for j in box:
            new_coord = int(j)
            globals()['box{}'.format(i)].append(new_coord)
                
        temp_box = globals()['box{}'.format(i)]
        
        globals()['box{}_topLeft'.format(i)] = (temp_box[0], temp_box[1])
        globals()['box{}_bottomLeft'.format(i)] = (temp_box[0], temp_box[3])
        globals()['box{}_topRight'.format(i)] = (temp_box[2], temp_box[1])
        globals()['box{}_bottomRight'.format(i)] = (temp_box[2], temp_box[3])
        
        temp_topLeft = globals()['box{}_topLeft'.format(i)]
        temp_bottomLeft = globals()['box{}_bottomLeft'.format(i)]
        temp_topRight = globals()['box{}_topRight'.format(i)]
        temp_bottomRight = globals()['box{}_bottomRight'.format(i)]
        
        if num == 0:
            cv2.circle(img, temp_topLeft, size, color, thickness)
            cv2.circle(img, temp_bottomLeft, size, color, thickness)
            cv2.circle(img, temp_topRight, size, color, thickness)
            cv2.circle(img, temp_bottomRight, size, color, thickness)

        elif num == 1:
            cv2.rectangle(img, temp_topLeft, temp_bottomRight, color, thickness)
            
    plt.imshow(img)
    

def revised_bbox (bbox_in):
    
    bbox_coord = []
    
    for i, box in enumerate(bbox_in):

        space_height = (box[3] - box[1]) * 0.5
        space_width = (box[2] - box[0]) * 0.5
        
        new_x1 = int(box[0] - space_width)
        new_y1 = int(box[1] - space_height)
        new_x2 = int(box[2] + space_width)
        new_y2 = int(box[3] + space_height)
        
        temp_array = []
        
        temp_array.append(new_x1)
        temp_array.append(new_y1)
        temp_array.append(new_x2)
        temp_array.append(new_y2)
        
        bbox_coord.append(temp_array)
    
    return bbox_coord



