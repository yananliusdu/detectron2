# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 22:59:07 2022

@author: 97091
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
import matplotlib as plt

if __name__ == '__main__': 
 CLASS_NAMES =["1","2","3","4"]
# 数据集路径
 DATASET_ROOT = '/media/yanan/One Touch/Cropped/coco'
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

 MODEL_PATH = '/media/yanan/One Touch/Cropped/output'
 
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
 cfg.MODEL.WEIGHTS = os.path.join(MODEL_PATH, "model_final.pth")
 cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4 
 cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
 cfg.DATASETS.TRAIN = ("my_train",)   
 dataset_dicts = DatasetCatalog.get("my_train")
 
 data_f = '/media/yanan/One Touch/Cropped/coco/images/color/0.png'

 cfg.MODEL.WEIGHTS = os.path.join(MODEL_PATH, "model_final.pth")  # path to the model we just trained
 cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
 dataset_dicts = DatasetCatalog.get("my_val")
 predictor = DefaultPredictor(cfg)


 start=time.time() 
 image= cv2.imread(data_f)
 
 raw_height, raw_width = image.shape[:2]
 
 image = predictor.aug.get_transform(image).apply_image(image)
 image1=image

 image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
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
     # pred_instances = predictor.model._postprocess(pred_instances, inputs, images.image_sizes)  # scale box to orig size
     # features of the proposed boxes
     feats = box_features[pred_inds]
 label=pred_instances[0].pred_classes
 print(label)
 label=label.cpu().detach().numpy()
 index=np.where(label==0)
 print(index[0])
 box=pred_instances[0].pred_boxes.tensor.cpu().detach().numpy()[index[0][0]]
 base=[int((box[0]+box[2])/2),int((box[1]+box[3])/2)]
 point_size = 1
 point_color = (0, 0, 255) # BGR
 thickness = 4 # 可以为 0 、4、8
 # cv2.circle(image1,base, 10, point_color, 0)
 data_set=[]
 for i in range(len(feats)):   
  predictions = predictor.model.roi_heads.box_predictor(feats[i])


  box=pred_instances[0].pred_boxes.tensor.cpu().detach().numpy()[i]
  
  center=[int((box[0]+box[2])/2),int((box[1]+box[3])/2)]
  cv2.circle(image1, tuple(center), 10, point_color, 0)

  
  v = Visualizer(image1[:,:,::-1],
                 scale=1,
                 metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                 )
  v = v.draw_instance_predictions(pred_instances[0][i].to("cpu"))
  
  cv2.imshow('images', v.get_image()[:, :, ::-1])

  cv2.waitKey(0)
  
  caption=input()
  d={'feature': feats[i],'center':center,'base':base,'caption':caption}
  data_set.append(d)
  cv2.destroyAllWindows()
