import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
import random
from collections import Counter

from model.rpn.bbox_transform import clip_boxes
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import bbox_transform_inv
from model.roi_layers import nms
from model.utils.net_utils import save_net, load_net, vis_detections

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

def calculate_informativeness(fasterRCNN, im_data, im_info, gt_boxes, num_boxes, cfg, args, imdb_classes,imdb_num_classes, info_scale):
    fasterRCNN.eval()
    empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
    all_boxes = [[[] for _ in xrange(1)]
               for _ in xrange(imdb_num_classes)]
    score_f = 0.
    thresh = 0.05

    im_vec, ins_vec,_,_,rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
    
    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]
    
    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
          if args.class_agnostic:
              box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
              box_deltas = box_deltas.view(1, -1, 4)
          else:
              box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
              box_deltas = box_deltas.view(1, -1, 4 * len(imdb_classes))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        #print("clip bbx", im_info.data)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))
    
    #pred_boxes /= info_scale.item()#if nit show, donot need to scale back to original size

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()
    #print(pred_boxes)
    
    for j in xrange(1, imdb_num_classes):
        inds = torch.nonzero(scores[:,j]>thresh).view(-1)
        # if there is det
        if inds.numel() > 0:
          cls_scores = scores[:,j][inds]
          _, order = torch.sort(cls_scores, 0, True)
          if args.class_agnostic:
            cls_boxes = pred_boxes[inds, :]
          else:
            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
          cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
          # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
          cls_dets = cls_dets[order]
          keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
          cls_dets = cls_dets[keep.view(-1).long()]
            
          all_boxes[j][0] = cls_dets.cpu().numpy()
        else:
          all_boxes[j][0] = empty_array
    
    mean_score=[]
    max_score=0
    ins_pred_bbox=None
    show_boxes = {j:[] for j in xrange(1, imdb_num_classes)}
    conf_score_list=[]
    for j in xrange(1, imdb_num_classes):
        show_boxes[j]=[all_boxes[j][0][d,:4] for d in range(np.minimum(10, all_boxes[j][0].shape[0])) if all_boxes[j][0][d,-1]>0.]
        real_score=[all_boxes[j][0][d,-1] for d in range(np.minimum(10, all_boxes[j][0].shape[0])) if all_boxes[j][0][d,-1]>0.]
        mean_score.extend(real_score)
        if len(real_score)!=0:
            conf_score_list.append(np.mean(real_score))
            temp_max_score=max(real_score)
            if temp_max_score>max_score:
                max_score=temp_max_score
                ins_pred_bbox=show_boxes[j][real_score.index(temp_max_score)].astype(int)
        else:
            conf_score_list.append(0.)

    #if len(mean_score)!=0:
    #    conf_score_list.sort()
    #    #score_f=np.mean(mean_score)
    #    score_f=conf_score_list[-1]-conf_score_list[-2]
    score_f=max_score
    return score_f,max_score,ins_pred_bbox,im_vec, ins_vec

