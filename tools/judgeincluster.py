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
import matplotlib.pyplot as plt

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

def judgeincluster(im_vec, in_vec,im, ins_pca, ins_normlized_centers, ins_x_min, ins_x_max, ins_score, ins_pred_bbox, in_rep, in_rep_num_list, thresh=0.1):

    judge=True
    '''
    img_gray=cv2.cvtColor(im[0].detach().cpu().numpy().transpose(1,2,0),cv2.COLOR_RGB2GRAY)
    img=cv2.resize(img_gray,(600,600))
    img=img.reshape(1,360000)  
    im_vec=img
    
    
    if ins_score==0:
      return 100, -1, im_vec
    #print(ins_score, ins_pred_bbox)
    #print(img_gray.shape)
    ins=img_gray[ins_pred_bbox[1]:ins_pred_bbox[3],ins_pred_bbox[0]:ins_pred_bbox[2]]
    resized_ins=cv2.resize(ins,(600,600))
    resized_ins=resized_ins.reshape(1,360000) 
    in_vec=resized_ins
    '''
    resized_ins=in_vec
    pca_ins=ins_pca.transform(resized_ins)
    pca_rep=ins_pca.transform(in_rep)
    normlized_pca_ins=(pca_ins - ins_x_min) / (ins_x_max - ins_x_min)
    normlized_pca_rep=(pca_rep - ins_x_min) / (ins_x_max - ins_x_min)

    ins_dist_list=[]
    for i,center in enumerate(ins_normlized_centers):
      dist=distance.euclidean(center, normlized_pca_ins)
      ins_dist_list.append(dist)
      #print(dist)
      if dist<thresh:
        judge=False
    ins_div1=min(ins_dist_list)
    in_clus=ins_dist_list.index(ins_div1)
    
    # for the closest dist
    if in_clus==0:
      rep_clus=normlized_pca_rep[:in_rep_num_list[0]]
    else:
      rep_clus=normlized_pca_rep[sum(in_rep_num_list[:in_clus]):sum(in_rep_num_list[:in_clus+1])]
    ins_dist_list=[]
    for j in rep_clus:
      dist=distance.euclidean(j, normlized_pca_ins)
      ins_dist_list.append(dist)
      #print(dist)
      if dist<thresh:
        judge=False
    ins_div2=min(ins_dist_list)      
    ins_div=(ins_div1+ins_div2)/2.
    return ins_div, in_clus, in_vec

a_info=0.1
b_info=10

a_div=-0.1#0.5#0.2
b_div=10

def info_sigmoid(x,cent):
    global a_info
    global b_info
    #print(a_info,b_info,cent)
    a_info=a_info+cent
    #b=10
    #print(a_info,b_info)
    s=1/(1+np.exp(b_info*(a_info-x)))
    #the_s=1/(1+np.exp(b*(a-np.linspace(0,1,100))))
    #min_s=min(the_s)
    #max_s=max(the_s)
    return s#1*(s-min_s)/(max_s-min_s)

def div_sigmoid(x,cent):
    global a_div
    global b_div
    print(a_div,b_div)
    a_div=a_div+cent
    print(a_div,b_div)
    #b=10
    s=1/(1+np.exp(b_div*(a_div-x)))
    return s#1*(s-min(s))/(max(s)-min(s))

im_a_div=-0.4#0.1#0.2#-0.1
im_b_div=10

def im_div_sigmoid(x,cent,reset):
    global im_a_div
    global im_b_div
    if reset==True:
        im_a_div=-0.4#0. for indoor#0.1
    print("im_div a, b: ",im_a_div,im_b_div)
    im_a_div=im_a_div+cent
    print(im_a_div,im_b_div)
    s=1/(1+np.exp(im_b_div*(im_a_div-x)))
    return s


in_a_div=-0.4#0.2#-0.1
in_b_div=10

def in_div_sigmoid(x,cent,reset):
    global in_a_div
    global in_b_div
    if reset==True:
        in_a_div=-0.4#0.1 for indoor
    print("in_div a, b: ",in_a_div,in_b_div)
    in_a_div=in_a_div+cent
    print(in_a_div,in_b_div)
    s=1/(1+np.exp(in_b_div*(in_a_div-x)))
    return s

def accept_prob(uncertainty, im_div, in_div, all_im_div, all_in_div,im_info_cent,in_info_cent,all_im_info_cent,all_in_info_cent, im_div_cent,in_div_cent,all_im_div_cent,all_in_div_cent,reset):
    informativeness=1.-uncertainty
    #im_prob = info_sigmoid(informativeness,im_info_cent)*div_sigmoid(im_div,im_div_cent)
    im_accept = True#random.random() < im_prob
    
    #in_prob = info_sigmoid(informativeness, in_info_cent)*div_sigmoid(in_div,in_div_cent)
    in_accept = True#random.random() < in_prob
    '''
    fig=plt.figure()
    plt.subplot(1,3,1)
    plt.plot(np.linspace(0,1,100), info_sigmoid(np.linspace(0,1,100)))
    plt.subplot(1,3,2)
    plt.plot(np.linspace(0,1,100), div_sigmoid(np.linspace(0,1,100)))
    plt.subplot(1,3,3)
    plt.plot(np.linspace(0,1,100), info_sigmoid(np.linspace(0,1,100))*div_sigmoid(np.linspace(0,1,100)))
    plt.show()
    exit()
    '''
    
    all_im_prob = info_sigmoid(informativeness, all_im_info_cent)*im_div_sigmoid(all_im_div, all_im_div_cent,reset)
    all_im_accept = random.random() < all_im_prob
    
    all_in_prob = info_sigmoid(informativeness, all_in_info_cent)*in_div_sigmoid(all_in_div, all_in_div_cent,reset)
    all_in_accept = random.random() < all_in_prob
    
    return im_accept, in_accept, all_im_accept, all_in_accept
