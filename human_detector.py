
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pprint

import cv2
import numpy as np
from os.path import join

prototxt = './model/MobileNetSSD_deploy.prototxt'
caffemodel = './model/MobileNetSSD_deploy.caffemodel'

class HumanDetector:
    ''' Human bbox detector by MobileNetSSD '''
    def __init__(self, prototxt=prototxt, caffemodel=caffemodel):
        
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]

        # load our serialized model from disk
        # print("[INFO] loading model...")
        self.detector = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        self.CONSIDER = set(["dog", "person", "car"])
        self.objCount = {obj: 0 for obj in self.CONSIDER}


    def get_humanboxes(self, img_orig, threshold=0.2):
        
        H,W = img_orig.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img_orig, (300, 300)),
        0.007843, (300, 300), 127.5)

        self.detector.setInput(blob)
        detections = self.detector.forward()
        objCount = {obj: 0 for obj in self.CONSIDER}

        ### bbox selection
        objs = []
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0,0,i,2]
            if confidence > threshold:
                idx = int(detections[0,0,i,1])
                if self.CLASSES[idx] in self.CONSIDER:
                    objCount[self.CLASSES[idx]] += 1
                    box = detections[0,0,i,3:7] * np.array([W,H,W,H])
                    x1,y1,x2,y2 = box.astype(np.float)
                    x,y,w,h = [x1, y1, x2-x1, y2-y1]
                    c, s = self.xywh2cs(x,y,w,h)
                    objs.append({"idx":0, "center":c, "scale": s, "bbox": [x,y,w,h], "bbox_conf": confidence, "class":self.CLASSES[idx]})
        return objs

    def xywh2cs(self, x, y, w, h, aspect_ratio=3/4, pixel_std=200):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = np.array(
            [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

if __name__ == '__main__':
    main()
