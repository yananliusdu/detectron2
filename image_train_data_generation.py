# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 22:59:07 2022

@author: 97091

data generation for GNN training
"""

from detectron2.data.datasets import register_coco_instances

import os

from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog

import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.checkpoint import DetectionCheckpointer
import torch
import time
if __name__ == '__main__': 
 CLASS_NAMES =["part2","part3","part4"]
# 数据集路径
 DATASET_ROOT = '/media/yanan/One Touch/detectron2_data_Ma/detectron2_data/coco'
#标注文件夹路径
 ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')
#训练图片路径
 IMG_ROOT=os.path.join(DATASET_ROOT, 'images')
 TRAIN_PATH = os.path.join(IMG_ROOT, 'train2017')
#测试图片路径
 VAL_PATH = os.path.join(IMG_ROOT, 'val2017')
#训练集的标注文件
 TRAIN_JSON = os.path.join(ANN_ROOT, 'instances_train2017.json')
#验证集的标注文件
# VAL_JSON = os.path.join(ANN_ROOT, 'val.json')
#测试集的标注文件
 VAL_JSON = os.path.join(ANN_ROOT, 'instances_val2017.json')
 
 register_coco_instances("my_train", {}, TRAIN_JSON, TRAIN_PATH)
 train=MetadataCatalog.get("my_train").set(thing_classes=CLASS_NAMES,  
                                                    evaluator_type='coco', 
                                                    json_file=TRAIN_JSON,
                                                    image_root=TRAIN_PATH)
 register_coco_instances("my_val", {}, VAL_JSON, VAL_PATH)
 val=MetadataCatalog.get("my_val").set(thing_classes=CLASS_NAMES,  
                                                    evaluator_type='coco', 
                                                    json_file=VAL_JSON,
                                                    image_root=VAL_PATH)
 config_file =  "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
 cfg = get_cfg()

 cfg.merge_from_file(model_zoo.get_config_file(config_file))
 cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  
 cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 

 cfg.DATASETS.TRAIN = ("my_train",)   
 dataset_dicts = DatasetCatalog.get("my_train")
 
 data_f ='D:/detectron2_data/color/54.jpg'

 cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
 cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
 dataset_dicts = DatasetCatalog.get("my_val")
 predictor = DefaultPredictor(cfg)
 path = '/media/yanan/One Touch/detectron2_data_Ma/detectron2_data/color/'
 path_list=os.listdir(path)
 data_set1=[]
 f=[]
 save_root='/media/yanan/One Touch/detectron2_data_Ma/detectron2_data/crops_temp/'
 count=0
 for filename in path_list:
    if os.path.splitext(filename)[1] == '.png':
        f.append(filename)
 for f_ in f:
  data_f=os.path.join(path,f_)
  save_path=os.path.join(save_root, str(count))
  os.makedirs(save_path)

  image1= cv2.imread(data_f)
 
  raw_height, raw_width = image1.shape[:2]
 
  image = predictor.aug.get_transform(image1).apply_image(image1)
 

  image = torch.as_tensor(image1.astype("float32").transpose(2, 0, 1))
  inputs = [{"image": image, "height": raw_height, "width": raw_width}]
  with torch.no_grad():
     images = predictor.model.preprocess_image(inputs)  # don't forget to preprocess
    
     features = predictor.model.backbone(images.tensor)  # set of cnn features
     proposals, _ = predictor.model.proposal_generator(images, features, None)  # RPN

     features_ = [features[f] for f in predictor.model.roi_heads.box_in_features]
     box_features =predictor.model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
     box_features = predictor.model.roi_heads.box_head(box_features)  # features of all 1k candidates
     predictions = predictor.model.roi_heads.box_predictor(box_features)
     pred_instances, pred_inds = predictor.model.roi_heads.box_predictor.inference(predictions, proposals)
     pred_instances = predictor.model.roi_heads.forward_with_given_boxes(features, pred_instances)

      # output boxes, masks, scores, etc
     pred_instances = predictor.model._postprocess(pred_instances, inputs, images.image_sizes)  # scale box to orig size
     
     feats = box_features[pred_inds]
  label=pred_instances[0]['instances'].pred_classes
  label=label.cpu().detach().numpy()
 
  point_size = 1
  point_color = (0, 0, 255) # BGR
  thickness = 4 # 可以为 0 、4、8

  data_set=[]
 
  ce=[]
  for i in range(len(feats)):   
      predictions1 = predictor.model.roi_heads.box_predictor(feats[i])

      box=pred_instances[0]['instances'].pred_boxes.tensor.cpu().detach().numpy()[i]
      
      center=[int((box[0]+box[2])/2),int((box[1]+box[3])/2)]
      ll=pred_instances[0]['instances'].pred_classes.cpu().detach().numpy()[i]
      # cv2.circle(image1,center, 10, point_color, 0)
      crop=image1[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
     
      cen=[center[0]/640,center[1]/480]
      d={'feature': feats[i],'center':cen,'label':ll,'box':box}
      p11=os.path.join(save_path,str(i)+'.png')
      cv2.imwrite(p11,crop)
      p22=os.path.join(save_path,str(i))
      np.save(p22, d)
      # ce.append(cen)
      # image = cv2.putText(image1, str(i), center,cv2.FONT_HERSHEY_SIMPLEX, 
      #             1, (255, 0, 0), thickness, cv2.LINE_AA)
  p11=os.path.join(save_path,'ori.png')
  cv2.imwrite(p11,image1)    
  count=count+1     
  # v = Visualizer(image1[:,:,::-1],
  #           scale=1,
  #           metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            
  #           )
  # v = v.draw_instance_predictions(pred_instances[0]['instances'].to("cpu"))
  
  # box=pred_instances[0]['instances'].pred_boxes.tensor.cpu().detach().numpy()   
  # cv2.imshow('image', v.get_image()[:, :, ::-1])

 
  # cv2.waitKey(1)
 
  # caption=input()
  # fe=feats.cpu().detach().numpy()
  # ll=pred_instances[0]['instances'].pred_classes.cpu().detach().numpy()
  # d={'feature': fe,'center':ce,'caption':caption,'label':ll,'box':box}
  # data_set.append(d)
  # cv2.destroyAllWindows()
  # data_set1.append(data_set)
  

