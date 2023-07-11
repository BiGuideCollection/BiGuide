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

from roi_data_layer.roidb import combined_roidb
from model.rpn.bbox_transform import clip_boxes
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import bbox_transform_inv
from model.roi_layers import nms
from model.utils.net_utils import save_net, load_net, vis_detections

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
from scipy.spatial import distance

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.benchmark = False
     torch.backends.cudnn.deterministic = True

def judgeimcluster(im_vec, in_vec,im,pca,normlized_centers,x_min,x_max,im_rep,im_rep_num_list,thresh=0.1):

    judge=True
    '''
    img_gray=cv2.cvtColor(im[0].detach().cpu().numpy().transpose(1,2,0),cv2.COLOR_RGB2GRAY)
    img=cv2.resize(img_gray,(600,600))
    img=img.reshape(1,360000)  
    im_vec=img
    '''
    img=im_vec
    pca_im=pca.transform(img)
    #print(pca_im.shape)
    normlized_pca_im=(pca_im - x_min) / (x_max - x_min)
    pca_rep=pca.transform(im_rep)
    normlized_pca_rep=(pca_rep - x_min) / (x_max - x_min)
   
    im_dist_list=[]
    for center in normlized_centers:
      dist=distance.euclidean(center, normlized_pca_im)
      im_dist_list.append(dist)
      #print(dist)
      if dist<thresh:
        judge=False
    back_div1=min(im_dist_list)
    im_clus=im_dist_list.index(back_div1)
    
    # for the closest dist
    if im_clus==0:
      rep_clus=normlized_pca_rep[:im_rep_num_list[0]]
    else:
      rep_clus=normlized_pca_rep[sum(im_rep_num_list[:im_clus]):sum(im_rep_num_list[:im_clus+1])]
    im_dist_list=[]
    for j in rep_clus:
      dist=distance.euclidean(j, normlized_pca_im)
      im_dist_list.append(dist)
      #print(dist)
      if dist<thresh:
        judge=False
    back_div2=min(im_dist_list)
    back_div=(back_div1+back_div2)/2.
    return back_div, im_clus, im_vec
