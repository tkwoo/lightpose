from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pprint

import cv2
import numpy as np
from os.path import join
import torch

from utils.pose_resnet import get_pose_net
from utils.transforms import get_affine_transform, xywh2cs
from utils.inference import get_final_preds
from utils import usrcoco

pth = './model/res50_256.pth'

class HumanKeypoint:
    ''' Human keypoint detector by simplepose '''
    def __init__(self, modelfile=pth, device='cuda'):
        
        self.model = get_pose_net()
        if device == 'cuda': self.model.cuda()
        checkpoint = torch.load(modelfile) # hw 256x192
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()
        self.device = device

    def get_humankeypoints(self, img_orig, objs, threshold=0.6):
        
        n_objs = len(objs)
        np_inputs = np.zeros(shape=(n_objs, 3, 256, 192), dtype=np.float32)
        
        for idx in range(n_objs):
            c = objs[idx]['center']
            s = objs[idx]['scale']
            trans = get_affine_transform(c, s, 0, (192,256), inv=0)
            warp_img = cv2.warpAffine(img_orig,trans,(192,256),flags=cv2.INTER_LINEAR)
            np_input = cv2.cvtColor(warp_img, cv2.COLOR_BGR2RGB)
            np_input = np.expand_dims(np_input, 0).astype(np.float32)
            np_inputs_nchw = np_input.transpose(0,3,1,2) / 255
            np_inputs[idx] = self.standardization(np_inputs_nchw[0])
        
        tc_input = torch.from_numpy(np_inputs)
        if self.device == 'cuda':
            tc_input = tc_input.cuda()
        with torch.no_grad():
            output_data = self.model(tc_input).cpu().numpy()

        list_c = [obj['center'] for obj in objs]
        list_s = [obj['scale'] for obj in objs]
        preds, maxvals = get_final_preds(output_data, list_c, list_s)

        annotations = []
        cnt_num_point = 0
        for obj_idx in range(len(objs)):
            keypoints = []
            for idx, ptval in enumerate(zip(preds[obj_idx], maxvals[obj_idx])):
                point, maxval = ptval
                x,y = np.array(point, dtype=np.float)
                if maxval > threshold:
                    keypoints.extend([x,y,2])
                    cnt_num_point += 1
                else:
                    keypoints.extend([0,0,0])

            x,y,w,h = objs[obj_idx]['bbox']

            annotation = usrcoco.create_annotation_info(annotation_id=obj_idx+1, image_id=1, category_info=1, keypoints=keypoints, num_keypoints=cnt_num_point, bounding_box=objs[obj_idx]['bbox'])
            annotations.append(annotation)

        return annotations
        
    def standardization(self, np_input, 
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        np_mean = np.array(mean, dtype=np.float32).reshape(3,1,1)
        np_std = np.array(std, dtype=np.float32).reshape(3,1,1)
        return (np_input - np_mean) / np_std
