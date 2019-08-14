import cv2

import numpy as np
from utils import usrcoco
import torch
from os.path import join

from pycocotools.coco import COCO

from human_detector import HumanDetector
from human_keypoint import HumanKeypoint
# from onnx_keypoint import HumanKeypoint


def main():
    coco = COCO('./utils/tkwoo_annotation.json')

    total_start = cv2.getTickCount()
    detector = HumanDetector()
    keypoint = HumanKeypoint(device='cuda')
    time = (cv2.getTickCount() - total_start) / cv2.getTickFrequency() * 1000
    print ('[INFO] model loading time: %.3fms'%time)

    vc = cv2.VideoCapture('./human_cnt1.mp4')

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # ret, img_orig = vc.read()
    # h,w = img_orig.shape[:2]
    # out = cv2.VideoWriter('HPE.mp4',fourcc, 30.0, (w,h))

    cnt = 0
    aver_human = 0
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
        if len(objs) > 0:
            annotations = keypoint.get_humankeypoints(img_orig, objs)
        time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000

        cnt_human = 0
        for ann in annotations:
            if np.count_nonzero(ann['keypoints'][2::3]) > 4:
                cnt_human += 1
        aver_human = (cnt-1)*aver_human/cnt + cnt_human/cnt

        print ('[INFO:%d] times: (%.1fms,%.1fms), Human:%d, Aver:%.1f'%(cnt, box_time, time, cnt_human, aver_human))

        ### coco style prediction & draw the result
        cats = coco.loadCats(coco.getCatIds())[0]
        show = usrcoco.drawAnns(img_orig.copy(), annotations, cats)

        text = 'Human:%d, Aver:%.1f'%(cnt_human, aver_human)
        cv2.putText(show,text,(10,20),1,cv2.FONT_HERSHEY_PLAIN, (128,255,255), 1, cv2.LINE_8)
        cv2.putText(show,'%.1ffps'%(1000/(time+box_time)),(w-70,20),1,cv2.FONT_HERSHEY_PLAIN, (0,30,0), 1, cv2.LINE_8)
        
        # out.write(show)

        cv2.imshow('show', show)
        key = cv2.waitKey(1)
        if key == 27:
            break
        
    
    total_time = (cv2.getTickCount() - total_start) / cv2.getTickFrequency()
    print ('[INFO] total time: %.3fs'%total_time)

if __name__ == '__main__':
    main()