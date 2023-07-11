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

def judgeallcluster(im_vec, in_vec,pca,normlized_centers,x_min,x_max, ins_pca,ins_normlized_centers,ins_x_min,ins_x_max,ins_score,ins_pred_bbox,im_rep,in_rep, im_rep_num_list,in_rep_num_list,thresh=0.1):

    judge=True
    #print(im_vec.shape, in_vec.shape)
    img=im_vec
    pca_im=pca.transform(img)
    #print(pca_im.shape)
    normlized_pca_im=(pca_im - x_min) / (x_max - x_min)
    pca_rep=pca.transform(im_rep)
    normlized_pca_rep=(pca_rep - x_min) / (x_max - x_min)
   
    im_dist_list=[]
    for center in normlized_centers:
      #print(center.shape,normlized_pca_im.shape)
      dist=distance.euclidean(center, normlized_pca_im.reshape(-1))
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
      dist=distance.euclidean(j, normlized_pca_im.reshape(-1))
      im_dist_list.append(dist)
      #print(dist)
      if dist<thresh:
        judge=False
    back_div2=min(im_dist_list)
    closest_im=rep_clus[im_dist_list.index(back_div2)]
    back_div=(back_div1+back_div2)/2.

    if ins_score==0:
      in_vec=[[0,0]]
      
      return back_div, 100, im_clus, -1, im_vec, in_vec#, normlized_pca_im, in_vec, normlized_centers[im_clus], in_vec[0], closest_im, in_vec[0]

    resized_ins=in_vec
    pca_ins=ins_pca.transform(resized_ins)
    normlized_pca_ins=(pca_ins - ins_x_min) / (ins_x_max - ins_x_min)
    in_pca_rep=ins_pca.transform(in_rep)
    in_normlized_pca_rep=(in_pca_rep - ins_x_min) / (ins_x_max - ins_x_min)
    
    ins_dist_list=[]
    for i,center in enumerate(ins_normlized_centers):
      dist=distance.euclidean(center, normlized_pca_ins.reshape(-1))
      ins_dist_list.append(dist)
      #print(dist)
      if dist<thresh:
        judge=False
    ins_div1=min(ins_dist_list)
    in_clus=ins_dist_list.index(ins_div1)
    
    # for the closest dist
    if in_clus==0:
      in_rep_clus=in_normlized_pca_rep[:in_rep_num_list[0]]
    else:
      in_rep_clus=in_normlized_pca_rep[sum(in_rep_num_list[:in_clus]):sum(in_rep_num_list[:in_clus+1])]
    ins_dist_list=[]
    for j in in_rep_clus:
      dist=distance.euclidean(j, normlized_pca_ins.reshape(-1))
      ins_dist_list.append(dist)
      #print(dist)
      if dist<thresh:
        judge=False
    ins_div2=min(ins_dist_list)
    closest_ins=in_rep_clus[ins_dist_list.index(ins_div2)]
    ins_div=(ins_div1+ins_div2)/2.
    return back_div,ins_div, im_clus, in_clus, im_vec, in_vec#, normlized_pca_im, normlized_pca_ins, normlized_centers[im_clus], ins_normlized_centers[in_clus], closest_im, closest_ins
