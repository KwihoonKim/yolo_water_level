# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 02:00:18 2024

1. iou: 지표 값 계산

2. nms: 겹치는 detect 있으면 모두 없애버리고 confidence score 제일 큰거만 남김
   -> model.predict 바로 다음에 해줘야 하고, nms의 결과가 최종 결과임
   -> 모델의 성능평가는 nms를 거친 데이터로 하기.
   
3. del_fp: false positive를 없애주는 알고리즘으로, 실제 적용을 위해서 사용됨 
   -> 모델의 성능평가와는 무관하지만 실제 적용을 위해 사용됨.

4. confu_mat: tp, fp, fn 을 출력해주며, 여러 사진의 결과를 모아서 precision, recall 값 계산

2024. 08. 25. 최종
@author: Kwihoon
"""

import numpy as np
import math
from torchvision import transforms

def iou(box1, box2):
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou

def nms (bbox, conf, iou_thres = 0):
    iou_thres = 0
    bboxes = np.array(bbox)

    answer = [True for x in range(bboxes.shape[0])]

    for i in range(bboxes.shape[0]):
        if answer[i] is False:
            continue
        for j in range(bboxes.shape[0]):
            iou_val = iou(bboxes[i], bboxes[j])
            if iou_val > iou_thres and int(iou_val) != 1:
                answer[j] = False

    new_bbox = []
    new_conf = []

    for i, data in enumerate(zip(bbox, conf)):
        box, conf = data
        if answer[i] == True:
            new_bbox.append(box)
            new_conf.append(conf)
            
    return answer, new_bbox, new_conf

def nms_seg (bbox, conf, seg_box, iou_thres = 0):
    iou_thres = 0
    bboxes = np.array(bbox)
    transform_img = transforms.ToPILImage()

    answer = [True for x in range(bboxes.shape[0])]

    for i in range(bboxes.shape[0]):
        if answer[i] is False:
            continue
        for j in range(bboxes.shape[0]):
            iou_val = iou(bboxes[i], bboxes[j])
            if iou_val > iou_thres and int(iou_val) != 1:
                answer[j] = False

    new_bbox = []
    new_conf = []
    new_seg_box = []

    for i, data in enumerate(zip(bbox, conf, seg_box)):
        box, conf, seg = data
        if answer[i] == True:
            new_bbox.append(box)
            new_conf.append(conf)
            seg = transform_img(seg)
            seg = np.array(seg)
            new_seg_box.append(seg)
            
    new_seg_box = new_seg_box[0]    
    
    return answer, new_bbox, new_conf, new_seg_box

def del_fp (bbox, conf, num = 2):
    'num 개수에 해당하는 객체만 남기고 나머지 제거'
    num = 2 # detect 정답 개수
    coef = 1 / sum(conf) / 2
    
    product_x = []
    product_y = []
    
    for i, box in enumerate(bbox):
        product_x.append(conf[i] * (box[0] + box[2]))
        product_y.append(conf[i] * (box[1] + box[3]))

    centroid_x = coef * sum(product_x)
    centroid_y = coef * sum(product_y)
    
    dists = []
    
    for i, box in enumerate(bbox):
        dist = math.sqrt(math.pow(centroid_x - (box[0] + box[2]) / 2, 2)\
                          + math.pow(centroid_y - (box[1] + box[3]) / 2, 2))
        dists.append(dist)
        
    ranks = [sorted(dists, reverse=False).index(ele) for ele in dists]
    
    new_bbox = []
    new_conf = []
    
    for i, data in enumerate(zip(bbox, conf)):
        box_temp, conf_temp = data
        if ranks[i] < num:
            new_bbox.append(box_temp)
            new_conf.append(conf_temp)
    
    return new_bbox, new_conf 


def confu_mat (predicted, ground_truth, num):
    '''
    <<__confusion matrix__>>
    
    box1 = [5186, 3376, 5301, 3594]
    box2 = [5324, 3322, 5460, 5026]
    ground_truth = [box1,box2]
    
    bbox1 = [5324.75048828125, 3322.55224609375, 5460.12841796875, 5025.94091796875]
    bbox2 = [5186.54541015625, 3376.2900390625, 5301.01025390625, 3593.505615234375]
    bbox3 = [3215.101806640625, 3416.517578125, 3429.76416015625, 4097.3017578125]

    predicted = [bbox1, bbox2, bbox3]
    '''
    tp = 0
    fp = 0
    fn = 0
    if len(predicted) == 0:
        tp = 0
        fp = 0
        fn = num
    
    else:
        for j in predicted:
            iou_score = []
    
            for i in ground_truth:
                iou_score.append(iou(j,i))
            
            if sum(iou_score) > 0.01:
                tp += 1
            
            else:
                fp += 1
            
            fn = num - tp
    
    return tp, fp, fn


def pr_score(array):
    '''
    <<__precision recall score__>>
    input: [[1,1,1], [2,0,0], [0,0,2], ...]
    output: precision, recall
    '''
    tp, fp, fn = np.array(array).T
    tp = sum(tp)
    fp = sum(fp)
    fn = sum(fn)
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    return precision, recall
    
    
# =============================================================================
# bbox1 = [5324.75048828125, 3322.55224609375, 5460.12841796875, 5025.94091796875]
# bbox2 = [5186.54541015625, 3376.2900390625, 5301.01025390625, 3593.505615234375]
# bbox3 = [3215.101806640625, 3416.517578125, 3429.76416015625, 4097.3017578125]
# 
# bbox = [bbox1, bbox2, bbox3]
# 
# conf1 = 0.883294403553009
# conf2 = 0.8150808811187744
# conf3 = 0.34066125750541687
# 
# conf = [conf1, conf2, conf3]
# 
# box1 = [5186, 3376, 5301, 3594]
# box2 = [5324, 3322, 5460, 5026]
# ground_truth = [box1,box2]
# 
# results = []
# for i, box in enumerate(bbox_list):
#     result = confu_mat(box, ground_truth)
#     results.append(result)
# =============================================================================






























