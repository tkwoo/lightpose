import cv2

import numpy as np
from utils import usrcoco
import torch

from pycocotools.coco import COCO

from human_detector import HumanDetector
# from human_keypoint import HumanKeypoint
from onnx_keypoint import HumanKeypoint


def main():
    coco = COCO('./utils/tkwoo_annotation.json')

    total_start = cv2.getTickCount()
    detector = HumanDetector()
    keypoint = HumanKeypoint(device='cuda')
    time = (cv2.getTickCount() - total_start) / cv2.getTickFrequency() * 1000
    print ('[INFO] model loading time: %.3fms'%time)

    vc = cv2.VideoCapture('./test_video/pose_test.mp4')

    cnt = 0
    while True:
        cnt += 1
        if cnt > 1000:
            break
        _, img_orig = vc.read()
        if img_orig is None:
            break

        start = cv2.getTickCount()

        objs = detector.get_humanboxes(img_orig)
        box_time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
        # print ('[INFO] box time:%.3fms'%(time))

        start = cv2.getTickCount()
        annotations = keypoint.get_humankeypoints(img_orig, objs)
        time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
        print ('[INFO:%d] processing time: %.3fms, %.3fms'%(cnt, box_time, time))
        
        ### coco style prediction & draw the result
        cats = coco.loadCats(coco.getCatIds())[0]
        show = usrcoco.drawAnns(img_orig.copy(), annotations, cats)

        cv2.imshow('show', show)
        key = cv2.waitKey(1)
        if key == 27:
            break
    
    total_time = (cv2.getTickCount() - total_start) / cv2.getTickFrequency()
    print ('[INFO] total time: %.3fs'%total_time)

if __name__ == '__main__':
    main()