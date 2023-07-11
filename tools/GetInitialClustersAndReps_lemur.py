import random
import os
import sys
import numpy as np
import pdb
import time

import torch

import argparse
from pathlib import Path

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

import xml.etree.ElementTree as ET

import copy
import pandas as pd
import cv2 as cv
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle as pk
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize feature map')
    parser.add_argument(
        'img', help='Image path, include image file, dir and URL.')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--target-layers',
        default=['backbone'],
        nargs='+',
        type=str,
        help='The target layers to get feature map, if not set, the tool will '
        'specify the backbone')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument(
        '--show', action='store_true', help='Show the featmap results')
    parser.add_argument(
        '--topk',
        type=int,
        default=4,
        help='Select topk channel to show by the sum of each channel')
    args = parser.parse_args()
    return args

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

def get_label2id(labels_path: str) -> Dict[str, int]:
    """id is 1 start"""
    with open(labels_path, 'r') as f:
        labels_str = f.read().split()
    labels_ids = list(range(1, len(labels_str)+1))
    return dict(zip(labels_str, labels_ids))

def get_coco_annotation_from_obj(obj, label2id):
    label = obj.findtext('name')
    assert label in label2id, f"Error: {label} is not in label2id !"
    category_id = label2id[label]
    bndbox = obj.find('bndbox')
    xmin = int(float(bndbox.findtext('xmin'))) - 1
    ymin = int(float(bndbox.findtext('ymin'))) - 1
    xmax = int(float(bndbox.findtext('xmax')))
    ymax = int(float(bndbox.findtext('ymax')))
    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
        'segmentation': []  # This script is not for segmentation
    }
    return ann

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.benchmark = False
     torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
  setup_seed(3)#3,30,300
  args = parse_args()

  print('Called with args:')
  print(args)

  cfg = Config.fromfile(args.config)

  init_default_scope(cfg.get('default_scope', 'mmyolo'))

  model = init_detector(args.config, args.checkpoint, device=args.device)

  target_layers = []
  for target_layer in args.target_layers:
      #print(target_layer)
      try:
          target_layers.append(eval(f'model.{target_layer}'))
      except Exception as e:
          print(model)
          raise RuntimeError('layer does not exist', e)

  activations_wrapper = ActivationsWrapper(model, target_layers)
  
  if torch.cuda.is_available():
    print("cuda available")
  
  ################## center of gray scale of old target data
  #lines=open('/root/code/faster-rcnn.pytorch/data/indoor/VOC2007/ImageSets/Main/train_mixup_s.txt').readlines()
  #random.shuffle(lines)
  #open('/root/code/faster-rcnn.pytorch/data/indoor/VOC2007/ImageSets/Main/train_s_shuffle.txt','w').writelines(lines)
  
  # get file list
  lines=open('/path to/mmyolo/data/cleanlemur/VOC2007/ImageSets/Main/train_new4class_s.txt').readlines()
  #image_list, _ = get_file_list(args.img)

  label2id = get_label2id(labels_path="data/cleanlemur/VOC2007/labels.txt")
  # start detector inference
  im_vector_list=None#np.zeros((140,1600))#middle base feature 40*40
  instance_vector_list=None
  insNoneFlag=True
  imNoneFlag=True
  class_num_cnt_dict={0:0,1.:0.,2.:0,3.:0,4.:0}
  progress_bar = ProgressBar(len(lines))
  for line in lines:
    image_path='data/source_cleanlemur_coco/images/'+line.strip()+'.jpg'
    t1=time.time()
    result, featmaps = activations_wrapper(image_path)
    #print(result.pred_instances)
    #exit()
    img_feature,_ = torch.max(featmaps[0][0], 0)
    instance_feature=featmaps[1][0,:,result.pred_instances.instance_row_idx[0].item(), result.pred_instances.instance_col_idx[0].item()].detach().cpu()
     
    im_vector=img_feature.detach().cpu().reshape(1,-1)
    if imNoneFlag==True:
        im_vector_list=im_vector
        imNoneFlag=False
    else:
        im_vector_list=np.concatenate((im_vector_list,im_vector),axis=0)
    
    ins_vector= instance_feature.reshape(1,-1)
    
    img_name=image_path.split('/')[-1][:-4]
    ann_tree = ET.parse("data/cleanlemur/VOC2007/Annotations/"+img_name+".xml")
    ann_root = ann_tree.getroot()
    old_t_num_boxes=len(ann_root.findall('object'))
    ann = get_coco_annotation_from_obj(obj=ann_root.findall('object')[0], label2id=label2id)
    label=ann['category_id']
    if class_num_cnt_dict[label]<20:
          class_num_cnt_dict[label]=class_num_cnt_dict[label]+1
          im_vector_list[class_num_cnt_dict[0]]=im_vector
          class_num_cnt_dict[0]=class_num_cnt_dict[0]+1
          ins_num=old_t_num_boxes
          #print(str(class_num_cnt_dict[0])+':ins_vect shape',instance_vector_list.shape if insNoneFlag==False else 0)
          if insNoneFlag==True:
              instance_vector_list=ins_vector
              insNoneFlag=False
          else:
              instance_vector_list=np.concatenate((instance_vector_list,ins_vector),axis=0)
          
  print(im_vector_list.shape, instance_vector_list.shape)
  
  pca=PCA(n_components=15)#15#25%
  print("start to fit pca")
  pca_scale=pca.fit_transform(im_vector_list)
  pk.dump(pca,open("im_pca_lemur_lessrep.pkl","wb"))
  print("start to transform pca")
  #pca_scale=pca.transform(im_vector_list)
  pca_df_scale=pd.DataFrame(pca_scale,columns=['pc'+str(val+1) for val in range(15)])
  print("pca of img fitted!")
  #print(pca_scale.shape)
  
  silhouette_score_list=[]
  kmeans_list=[]
  labels_pca_scale_list=[]
  clus_num_list=[4]#[2,3,4,5,6,7,8]
  for clus_num in clus_num_list:
    kmeans = KMeans(n_clusters=clus_num, n_init=10, max_iter=300, init='k-means++', random_state=42)#100,400
    labels_pca_scale = kmeans.fit_predict(pca_df_scale)
    labels_pca_scale_list.append(labels_pca_scale)
    kmeans_list.append(kmeans)
    silhouette_score_list.append(silhouette_score(pca_df_scale, kmeans.labels_))
  sih_idx=silhouette_score_list.index(max(silhouette_score_list))
  final_clus_num=clus_num_list[sih_idx]
  kmeans=kmeans_list[sih_idx]
  labels_pca_scale=labels_pca_scale_list[sih_idx]
  print("kmeans of img fitted! best one is: ",final_clus_num )
  x_min, x_max = np.min(pca_scale, 0), np.max(pca_scale, 0)
  np.save('im_min_max_lemur_lessrep.npy', np.array([x_min, x_max]))
  #print(x_min, x_max, np.load('im_min_max.npy'))
  center_list= kmeans.cluster_centers_
  normlized_centers = (center_list - x_min) / (x_max - x_min)
  np.save('im_normlized_centers_lemur_lessrep.npy',normlized_centers)
  #print(normlized_centers, np.load('im_normlized_centers.npy'))
  im_repres=[]
  rep_num=round(20/final_clus_num)#40
  print("im rep num,", rep_num)
  rep_num_list=[]
  for clus in set(labels_pca_scale):
    clus_pca_scale=im_vector_list[labels_pca_scale==clus]
    #print(clus_pca_scale.shape)
    #print(clus_pca_scale)
    np.random.shuffle(clus_pca_scale)
    #print(clus_pca_scale)
    if len(im_repres)==0:
      im_repres=clus_pca_scale[:rep_num]#7
    else:
      print(im_repres.shape,clus_pca_scale[:rep_num].shape)
      im_repres=np.concatenate((im_repres,clus_pca_scale[:rep_num]), axis=0)
    rep_num_list.append(clus_pca_scale[:rep_num].shape[0])
  np.save('im_rep_lemur_lessrep.npy',im_repres)
  #print(im_repres)
  
  print("start to fit and transform pca")
  ins_pca=PCA(n_components=15)
  ins_pca_scale=ins_pca.fit_transform(instance_vector_list)
  pk.dump(ins_pca,open("in_pca_lemur_lessrep.pkl","wb"))
  
  ins_pca_df_scale=pd.DataFrame(ins_pca_scale,columns=['pc'+str(val+1) for val in range(15)])
  print("pca of ins fitted!")
  
  ins_silhouette_score_list=[]
  ins_kmeans_list=[]
  ins_labels_pca_scale_list=[]
  ins_clus_num_list=[4]#[2,3,4,5,6]
  
  for clus_num in ins_clus_num_list:
    ins_kmeans = KMeans(n_clusters=clus_num, n_init=10, max_iter=300, init='k-means++', random_state=42)
    ins_labels_pca_scale = ins_kmeans.fit_predict(ins_pca_df_scale)
    ins_labels_pca_scale_list.append(ins_labels_pca_scale)
    ins_kmeans_list.append(ins_kmeans)
    ins_silhouette_score_list.append(silhouette_score(ins_pca_df_scale, ins_kmeans.labels_))
  ins_sih_idx=ins_silhouette_score_list.index(max(ins_silhouette_score_list))
  ins_final_clus_num=ins_clus_num_list[ins_sih_idx]
  ins_kmeans=ins_kmeans_list[ins_sih_idx]
  ins_labels_pca_scale=ins_labels_pca_scale_list[ins_sih_idx]
  print("kmeans of in fitted! best one is: ",ins_final_clus_num )
  ins_x_min, ins_x_max = np.min(ins_pca_scale, 0), np.max(ins_pca_scale, 0)
  np.save('in_min_max_lemur_lessrep.npy', np.array([ins_x_min, ins_x_max]))
  #print(ins_x_min, ins_x_max, np.load('in_min_max.npy'))
  ins_center_list= ins_kmeans.cluster_centers_
  ins_normlized_centers = (ins_center_list - ins_x_min) / (ins_x_max - ins_x_min)
  np.save('in_normlized_centers_lemur_lessrep.npy',ins_normlized_centers)
  #print(ins_normlized_centers, np.load('in_normlized_centers.npy'))
  #ins_normlized_centers = (ins_center_list - x_min) / (x_max - x_min)
  in_repres=[]
  in_rep_num=round(20/ins_final_clus_num)
  in_rep_num_list=[]
  print("in rep num,", in_rep_num)
  for clus in set(ins_labels_pca_scale):
    clus_pca_scale=instance_vector_list[ins_labels_pca_scale==clus]
    print(clus_pca_scale.shape)
    #print(clus_pca_scale)
    np.random.shuffle(clus_pca_scale)
    #print(clus_pca_scale[:3])
   
    if len(in_repres)==0:
      in_repres=clus_pca_scale[:in_rep_num]
    else:
      print(in_repres.shape,clus_pca_scale[:in_rep_num].shape)
      in_repres=np.concatenate((in_repres,clus_pca_scale[:in_rep_num]), axis=0)
    in_rep_num_list.append(clus_pca_scale[:in_rep_num].shape[0])
  np.save('in_rep_lemur_lessrep.npy',in_repres)
  print(in_repres)
  
  
  
#python tools/GetInitialClustersAndReps_lemur.py data/cleanlemur_coco/images/                                 configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb1-30e_lemur.py                                 work_dirs/yolov5_s-v61_syncbn_fast_1xb1-30e_cleanlemur/best_coco_bbox_mAP_epoch_41.pth                                 --target-layers backbone.stage3 neck.bottom_up_layers[0]

