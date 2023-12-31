# 1. Library imports
import uvicorn
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
import pandas as pd
import copy

import sys
import pdb
from typing import Sequence, Dict, List
from mmengine.registry import init_default_scope
import mmcv
from mmdet.apis import inference_detector, init_detector
from mmengine import Config, DictAction
from mmengine.logging import print_log
from mmengine.utils import ProgressBar, path
from mmyolo.utils import switch_to_deploy
from mmyolo.utils.labelme_utils import LabelmeFormat
from mmyolo.utils.misc import auto_arrange_images,get_file_list, show_data_classes

import torch
import io, json
import os.path
from pathlib import Path
import cv2 as cv
import numpy as np
import base64
import matplotlib.pyplot as plt
from time import sleep, time
import datetime
from PIL import Image as im
import threading
from xml.etree.ElementTree import parse, Element, SubElement, ElementTree
import xml.etree.ElementTree as ET
from collections import defaultdict

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from judgeincluster import accept_prob
from judgeallcluster import judgeallcluster
import pickle as pk
import pandas as pd
from sklearn.metrics import silhouette_score
import random

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


class ActivationsWrapper:

    def __init__(self, model, target_layers):
        self.model = model
        self.activations = []
        self.handles = []
        self.image = None
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))

    def save_activation(self, module, input, output):
        self.activations.append(output)

    def __call__(self, img_path):
        self.activations = []
        results = inference_detector(self.model, img_path)
        return results, self.activations

    def release(self):
        for handle in self.handles:
            handle.remove()

# 2. Create app and model objects
app = FastAPI()


cfg = Config.fromfile("/path to/mmyolo/configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb1-30e_indoor.py")
init_default_scope(cfg.get('default_scope', 'mmyolo'))
model = init_detector("/path to/mmyolo/configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb1-30e_indoor.py", "/path to/mmyolo/work_dirs/your pretrained model.pth", device='cuda:0')

target_layers = []
for target_layer in ['backbone.stage3', 'neck.bottom_up_layers[0]']:
    #print(target_layer)
    try:
        target_layers.append(eval(f'model.{target_layer}'))
    except Exception as e:
        print(model)
        raise RuntimeError('layer does not exist', e)

activations_wrapper = ActivationsWrapper(model, target_layers)

print("Load model successful")

max_per_image=1
class_idx_name={0:'mobile_phone',1:'scissors',2:'light_bulb',3:'tin_can',4:'ball',5:'mug',6:'remote_control'}
idx_class_name={'mobile_phone':0,'scissors':1,'light_bulb':2,'tin_can':3,'ball':4,'mug':5,'remote_control':6}

round_num=20
round_cnt=0
im_adj_cnt=0
in_adj_cnt=0
incom_name_list=[]
score_set=[]
width=720#1440
height=1480#2960
dataset_name='user_study_demo'
training_t_set_name="train_biguide_demo"
testing_t_set_name="test_biguide_demo"
save_train_name_list=[]
save_test_name_list=[]

both_guidance=["tilt_your_phone", "raise_your_phone", "lower_your_phone", "change_your_position", "change_the_object_pose"]
im_guidance=["tilt_your_phone", "raise_your_phone", "lower_your_phone", "change_your_position"]
in_guidance=["change_the_object_pose"]

all_im_rep_num_list=[5,5,5,5]#[10,10,10,10]
all_in_rep_num_list=[5,5,5,5]#[10,10,10,10]
all_old_im_rep_num_list=[5,5,5,5]#[10,10,10,10]
all_old_in_rep_num_list=[5,5,5,5]#[10,10,10,10]
all_im_final_clus_num=4
all_in_final_clus_num=4

all_im_cnt=0
all_in_cnt=0
all_im_div_list=[]
all_in_div_list=[]
all_im_last=0
all_in_last=0

repeat_im_reject=0
repeat_in_reject=0
repeat_both_reject=0

all_im_pca = pk.load(open("/path to/mmyolo/im_pca_indoor_lessrep.pkl",'rb'))
all_im_normlized_centers = np.load('/path to/mmyolo/im_normlized_centers_indoor_lessrep.npy')
all_im_x_min = np.load('/path to/mmyolo/im_min_max_indoor_lessrep.npy')[0]
all_im_x_max = np.load('/path to/mmyolo/im_min_max_indoor_lessrep.npy')[1]
all_im_rep = np.load('/path to/mmyolo/im_rep_indoor_lessrep.npy')
all_in_pca = pk.load(open("/path to/mmyolo/in_pca_indoor_lessrep.pkl",'rb'))
all_in_normlized_centers = np.load('/path to/mmyolo/in_normlized_centers_indoor_lessrep.npy')
all_in_x_min = np.load('/path to/mmyolo/in_min_max_indoor_lessrep.npy')[0]
all_in_x_max = np.load('/path to/mmyolo/in_min_max_indoor_lessrep.npy')[1]
all_in_rep = np.load('/path to/mmyolo/in_rep_indoor_lessrep.npy')

def updateimcluster(all_im_clus,all_im_vec):
    t1 = time()

    global all_im_rep_num_list
    global all_im_rep
    global all_im_pca
    global all_im_normlized_centers
    global all_im_x_min
    global all_im_x_max
    global all_im_final_clus_num

    for rep in range(all_im_rep_num_list[all_im_clus]-1):#7-1#4-1#10-1
        all_im_rep[(all_im_rep_num_list[all_im_clus]-rep-1)+sum(all_im_rep_num_list[:all_im_clus])]=all_im_rep[(all_im_rep_num_list[all_im_clus]-2-rep)+sum(all_im_rep_num_list[:all_im_clus])]
    all_im_rep[0+sum(all_im_rep_num_list[:all_im_clus])]=all_im_vec
    
    all_im_pca=PCA(n_components=15)
    all_im_pca_scale=all_im_pca.fit_transform(all_im_rep)
    all_im_pca_df_scale=pd.DataFrame(all_im_pca_scale,columns=['pc'+str(val+1) for val in range(15)])
    all_im_silhouette_score_list=[]
    all_im_kmeans_list=[]
    all_im_labels_pca_scale_list=[]
    all_im_clus_num_list=[4,5]
    for clus_num in all_im_clus_num_list:
        all_im_kmeans = KMeans(n_clusters=clus_num, n_init=10, max_iter=300, init='k-means++', random_state=42)#100,400
        all_im_labels_pca_scale = all_im_kmeans.fit_predict(all_im_pca_df_scale)
        all_im_labels_pca_scale_list.append(all_im_labels_pca_scale)
        all_im_kmeans_list.append(all_im_kmeans)
        all_im_silhouette_score_list.append(silhouette_score(all_im_pca_df_scale, all_im_kmeans.labels_))
    all_im_sih_idx=all_im_silhouette_score_list.index(max(all_im_silhouette_score_list))
    all_im_final_clus_num=all_im_clus_num_list[all_im_sih_idx]
    all_im_kmeans=all_im_kmeans_list[all_im_sih_idx]
    all_im_labels_pca_scale=all_im_labels_pca_scale_list[all_im_sih_idx]
    all_im_rep_num_list=[]
    all_new_im_rep=all_im_rep.copy()
    for all_im_rep_clu in range(all_im_final_clus_num):
        all_im_rep_num_list.append(len(all_im_labels_pca_scale[all_im_labels_pca_scale == all_im_rep_clu]))
        if all_im_rep_clu==0:
            all_new_im_rep[0:sum(all_im_rep_num_list)] = all_im_rep[all_im_labels_pca_scale == all_im_rep_clu]
        else:
            all_new_im_rep[sum(all_im_rep_num_list[:all_im_rep_clu]):sum(all_im_rep_num_list)] = all_im_rep[all_im_labels_pca_scale == all_im_rep_clu]
    all_im_rep=all_new_im_rep.copy()

    all_im_centers=all_im_kmeans.cluster_centers_
    all_im_x_min, all_im_x_max = np.min(all_im_pca_scale, 0), np.max(all_im_pca_scale, 0)
    all_im_normlized_centers = (all_im_centers - all_im_x_min) / (all_im_x_max - all_im_x_min)


def updateincluster(all_in_clus,all_in_vec):

    global all_in_rep_num_list
    global all_in_rep
    global all_in_pca
    global all_in_normlized_centers
    global all_in_x_min
    global all_in_x_max
    global all_in_final_clus_num
    
    if all_in_clus!=-1:
        for rep in range(all_in_rep_num_list[all_in_clus]-1):
            all_in_rep[(all_in_rep_num_list[all_in_clus]-rep-1)+sum(all_in_rep_num_list[:all_in_clus])]=all_in_rep[(all_in_rep_num_list[all_in_clus]-2-rep)+sum(all_in_rep_num_list[:all_in_clus])]
        all_in_rep[0+sum(all_in_rep_num_list[:all_in_clus])]=all_in_vec
    
    all_in_pca=PCA(n_components=15)
    all_in_pca_scale=all_in_pca.fit_transform(all_in_rep)
    all_in_pca_df_scale=pd.DataFrame(all_in_pca_scale,columns=['pc'+str(val+1) for val in range(15)])
    all_in_silhouette_score_list=[]
    all_in_kmeans_list=[]
    all_in_labels_pca_scale_list=[]
    all_in_clus_num_list=[4,5]#[4]#[2,3,4,5,6,7,8]
    for clus_num in all_in_clus_num_list:
        all_in_kmeans = KMeans(n_clusters=clus_num, n_init=10, max_iter=300, init='k-means++', random_state=42)
        all_in_labels_pca_scale = all_in_kmeans.fit_predict(all_in_pca_df_scale)
        all_in_labels_pca_scale_list.append(all_in_labels_pca_scale)
        all_in_kmeans_list.append(all_in_kmeans)
        all_in_silhouette_score_list.append(silhouette_score(all_in_pca_df_scale, all_in_kmeans.labels_))
    all_in_sih_idx=all_in_silhouette_score_list.index(max(all_in_silhouette_score_list))
    all_in_final_clus_num=all_in_clus_num_list[all_in_sih_idx]
    all_in_kmeans=all_in_kmeans_list[all_in_sih_idx]
    all_in_labels_pca_scale=all_in_labels_pca_scale_list[all_in_sih_idx]
    all_in_rep_num_list=[]
    all_new_in_rep=all_in_rep.copy()
    for all_in_rep_clu in range(all_in_final_clus_num):
        all_in_rep_num_list.append(len(all_in_labels_pca_scale[all_in_labels_pca_scale == all_in_rep_clu]))
        if all_in_rep_clu==0:
            all_new_in_rep[0:sum(all_in_rep_num_list)] = all_in_rep[all_in_labels_pca_scale == all_in_rep_clu]
        else:
            all_new_in_rep[sum(all_in_rep_num_list[:all_in_rep_clu]):sum(all_in_rep_num_list)] = all_in_rep[all_in_labels_pca_scale == all_in_rep_clu]
    all_in_rep=all_new_in_rep.copy()


    all_in_centers=all_in_kmeans.cluster_centers_
    all_in_x_min, all_in_x_max = np.min(all_in_pca_scale, 0), np.max(all_in_pca_scale, 0)
    all_in_normlized_centers = (all_in_centers - all_in_x_min) / (all_in_x_max - all_in_x_min)


class ImgPathData(BaseModel):
    rgb_base64: str
    currentTime: str

class LabelData(BaseModel):
    sent_im_name_list: str
    sent_bbnum_list: str
    sent_class_list: str
    sent_bbox_list: str

def base64str_to_OpenCVImage(rgb_base64):
    rgb = rgb_base64  # raw data with base64 encoding
    decoded_data = base64.b64decode(rgb)
    np_data = np.frombuffer(decoded_data, np.uint8)
    img_rgb = cv.imdecode(np_data,cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB)
    return img_rgb

def calculate_informativeness(img_name):
    #print("in calculate_informativeness:",img_name)
    result, featmaps= activations_wrapper("/path to/mmyolo/data/"+dataset_name+"/JPEGImages/"+img_name+".jpg")
    pred_instances = result.cpu().pred_instances
    
    if len(pred_instances.scores)==0:
      print("!!!!!!! NOthing is predicted!!!!!!!!!!!")
      row=0
      col=0
      max_score=0
      ins_pred_bbox=None
    else:
      row=result.pred_instances.instance_row_idx[0].item()
      col=result.pred_instances.instance_col_idx[0].item()
      max_score = pred_instances.scores[0]
      ins_pred_bbox = pred_instances.bboxes[0]
      
    img_feature,_ = torch.max(featmaps[0][0], 0)  
    instance_feature=featmaps[1][0,:,row, col].detach().cpu()
    im_vec=img_feature.detach().cpu().reshape(1,-1)
    ins_vec= instance_feature.reshape(1,-1)
    
    score_f=max_score
    return score_f,max_score,ins_pred_bbox,im_vec,ins_vec

@app.put("/guidance")
def guidance(d:ImgPathData):
    tg=time()
    now = datetime.datetime.now()
    print("receive image!"+str(now))
    img_rgb = base64str_to_OpenCVImage(d.rgb_base64)
    img_name = d.currentTime
    cur_im = im.fromarray(img_rgb, 'RGB')
    cur_im_savename="/path to/mmyolo/data/"+dataset_name+"/JPEGImages/"+img_name+".jpg"
    #print("in when save in guidance:",img_name)
    cur_im.save(cur_im_savename)
    
    global round_cnt
    global im_adj_cnt
    global in_adj_cnt
    global incom_name_list

    global all_im_div_list
    global all_in_div_list
    global all_in_last
    global all_in_cnt
    global all_im_last
    global all_im_cnt
    global all_old_in_rep_num_list
    global all_old_im_rep_num_list
    
    global repeat_im_reject
    global repeat_in_reject
    global repeat_both_reject
    
    global save_train_name_list
    global save_test_name_list
        
    all_in_last=all_in_last+1
    all_im_last=all_im_last+1

    informativeness,ins_score,ins_pred_bbox,im_vec, ins_vec=calculate_informativeness(img_name)

    all_im_div, all_in_div, all_im_clus, all_in_clus, all_im_vec, all_in_vec = judgeallcluster(im_vec, ins_vec, all_im_pca, all_im_normlized_centers, all_im_x_min, all_im_x_max, all_in_pca, all_in_normlized_centers, all_in_x_min, all_in_x_max, ins_score, ins_pred_bbox,all_im_rep,all_in_rep,all_im_rep_num_list,all_in_rep_num_list)
    
    all_im_div_cent=0
    all_in_div_cent=0
    _, _, all_im_accept, all_in_accept = accept_prob(1-informativeness, 0, 0, all_im_div, all_in_div,0,0,0,0,0,0,all_im_div_cent,all_in_div_cent,False)
    print("first judge the image is", all_im_accept, all_in_accept,1-informativeness)
    #  dynamic adjust thresh
    if all_in_accept==False:
        if len(all_in_div_list)==0 or all_in_last==in_adj_cnt+1:
            in_adj_cnt=all_in_last
            all_in_cnt=all_in_cnt+1
            all_in_div_list.append(all_in_div)  
        else:
            all_in_div_list=[]
            all_in_cnt=0
        if all_in_cnt==2:   
            all_in_div_cent=-0.1
            all_in_div_list=[]
            all_in_cnt=0      

    if all_im_accept==False:
        if len(all_im_div_list)==0 or all_im_last==im_adj_cnt+1:
            im_adj_cnt=all_im_last
            all_im_cnt=all_im_cnt+1
            all_im_div_list.append(all_im_div)
        else:
            all_im_div_list=[]
            all_im_cnt=0
        if all_im_cnt==2:
            all_im_div_cent=-0.1
            all_im_div_list=[]
            all_im_cnt=0
            

    if all_in_div_cent!=0 or all_im_div_cent!=0:
        _, _, all_im_accept, all_in_accept = accept_prob(1-informativeness, 0, 0, all_im_div, all_in_div,0,0,0,0,0,0,all_im_div_cent,all_in_div_cent,False)      
    print("after adjustment, the current image is",all_im_accept,all_in_accept,1-informativeness)

    if all_im_accept and all_in_accept:
        #update cluster
        tc=time()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!before update cluster time:"+str(datetime.datetime.now()))
        updateincluster_thread = threading.Thread(target=updateincluster, name="updateincluster", args=(all_in_clus, all_in_vec))
        updateincluster_thread.start()
        updateimcluster_thread = threading.Thread(target=updateimcluster, name="updateimcluster", args=(all_im_clus, all_im_vec))
        updateimcluster_thread.start()

        updateincluster_thread.join()
        updateimcluster_thread.join()
        
        
        s1=np.argsort(np.array(all_old_in_rep_num_list))
        s2=np.argsort(np.array(all_in_rep_num_list))
        print("in,",all_old_in_rep_num_list,all_in_rep_num_list)
        if np.array_equal(np.array(all_old_in_rep_num_list)[s1],np.array(all_in_rep_num_list)[s2]):
            _, _, _, _ = accept_prob(1-informativeness, 0, 0, all_im_div, all_in_div,0,0,0,0,0,0,0,0.05,False)
        all_old_in_rep_num_list=all_in_rep_num_list.copy()
        
        s11=np.argsort(np.array(all_old_im_rep_num_list))
        s22=np.argsort(np.array(all_im_rep_num_list))
        print("im,",all_old_im_rep_num_list,all_im_rep_num_list)
        if np.array_equal(np.array(all_old_im_rep_num_list)[s11],np.array(all_im_rep_num_list)[s22]):
            _, _, _, _ = accept_prob(1-informativeness, 0, 0, all_im_div, all_in_div,0,0,0,0,0,0,0.05,0,False)
        all_old_im_rep_num_list=all_im_rep_num_list.copy()

        incom_name_list.append(img_name)
        round_cnt=round_cnt+1
        if round_cnt==round_num:       
            print(incom_name_list)
            picked_image_list=incom_name_list
            print(picked_image_list)
            picked_name_prediction_dict={}
            for picked_name in picked_image_list:
                picked_name_prediction_dict[picked_name]=[0,picked_name,[[],],[]]
            user_guidance=picked_name_prediction_dict
            
            save_train_name_list.extend(incom_name_list)
            training_t_set=open("/home/i3t/lin/mmyolo/data/"+dataset_name+"/ImageSets/Main/"+training_t_set_name+".txt","w")
            for line in save_train_name_list:
                training_t_set.write(line+"\n")
            training_t_set.close()
            
            round_cnt=0
            incom_name_list=[]
            _, _, _, _ = accept_prob(0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,True)
            
        else:
            user_guidance="0+good job" 
    else:
        os.remove(cur_im_savename)
        if all_im_accept==False and all_in_accept==False:
            if repeat_both_reject==all_im_last-1:
                user_guidance="1+" + random.choice(both_guidance)
            else:
                user_guidance="2+" + random.choice(both_guidance)
            repeat_both_reject=all_im_last
        elif all_im_accept==False and all_in_accept==True:
            if repeat_im_reject==all_im_last-1:
                user_guidance="3+" + random.choice(im_guidance)
            else:
                user_guidance="4+" + random.choice(im_guidance) 
            repeat_im_reject=all_im_last
        else:
            if repeat_in_reject==all_im_last-1:
                user_guidance="5+" + random.choice(in_guidance)
            else:
                user_guidance="6+" + random.choice(in_guidance)
            repeat_in_reject=all_im_last
    
    
    jsonData = json.dumps(user_guidance)
    return jsonData

# Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app="UserStudy_BIGUIDE_indoor:app", host='192.168.1.19', port=5958)
