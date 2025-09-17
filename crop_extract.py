# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:47:33 2024

@author: USESR
"""

from model1_matrix import matrix_gt, matrix, mat_coord, mat_coord_c4
from model2_gauge1 import level_gauge1
from model2_gauge2 import level_gauge2
from model2_ptx_gauge2 import level_ptx_gauge2
from model2_ptx_gauge1 import level_ptx_gauge1
from model3_water import level_water

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

'1. crop plate'
'============================================================================='
path = 'c3/'
data_list = os.listdir(path)
data_list = data_list[:100]

for i in data_list:
    img_path = path + i
    img = cv2.imread(img_path)
    img_crop = img[3325:3644, 5167:5322, :]

    max_value = np.max(img_crop)
    img_crop = np.array(img_crop / max_value*255, dtype = 'uint8')

    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)    
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    gray = cv2.bitwise_not(gray)

    cv2.imwrite('crop_plate/' + i, gray)
'============================================================================='

'2. crop gauge'
'============================================================================='
for i in data_list:
    img_path = path + i
    img = cv2.imread(img_path)
    img_crop1 = img[3200:, 4687:5176, :]
    img_crop2 = img[3200:, 5290:5550, :]

    max_value1 = np.max(img_crop1)
    max_value2 = np.max(img_crop2)

    img_crop1 = np.array(img_crop1 / max_value1*255, dtype = 'uint8')
    img_crop2 = np.array(img_crop2 / max_value2*255, dtype = 'uint8')

    gray1 = cv2.cvtColor(img_crop1, cv2.COLOR_BGR2GRAY)    
    gray1 = cv2.GaussianBlur(gray1, (5,5), 0)
    gray1 = cv2.bitwise_not(gray1)
    
    gray2 = cv2.cvtColor(img_crop2, cv2.COLOR_BGR2GRAY)    
    gray2 = cv2.GaussianBlur(gray2, (5,5), 0)
    gray2 = cv2.bitwise_not(gray2)
    
    cv2.imwrite('crop_gauge/' + 'gauge1_' + i, gray1)
    cv2.imwrite('crop_gauge/' + 'gauge2_' + i, gray2)








