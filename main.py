# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 22:59:07 2022

@author: 97091
"""

from detectron2.data.datasets import register_coco_instances
import os
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import torch
from utli import *
from model import GNN,DecoderRNN
from detectron2.utils.visualizer import Visualizer

if __name__ == '__main__': 
 CLASS_NAMES =["part2","part3","part4"]
# 数据集路径
 DATASET_ROOT = r'/home/yanan/project-2/project/coco'
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
 
 Encoder = GNN(hidden_channels=256)
 Encoder.load_state_dict(torch.load('encoder1.pt'))
 Encoder.eval()
 
 Decoder=DecoderRNN(128,100,7)
 Decoder.load_state_dict(torch.load('decoder.pt'))
 Decoder.eval()
 

 cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
 cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
 dataset_dicts = DatasetCatalog.get("my_val")
 predictor = DefaultPredictor(cfg)
 
 #读图片的代码 
 path = r'/home/yanan/project-2/project/crops1'
 path_list=os.listdir(path)
 data_set1=[]

 save_root=r'/home/yanan/project-2/project/test_tmp'
 f_='176/ori.png'
 data_f=os.path.join(path,f_)
 #把探测到的数据保存
 temp_f=5
 save_path=os.path.join(save_root, str(temp_f))

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
 
 ce=[]
 for i in range(len(feats)):   
      predictions1 = predictor.model.roi_heads.box_predictor(feats[i])

      box=pred_instances[0]['instances'].pred_boxes.tensor.cpu().detach().numpy()[i]
      
      center=[int((box[0]+box[2])/2),int((box[1]+box[3])/2)]
      ll=pred_instances[0]['instances'].pred_classes.cpu().detach().numpy()[i]
      
      crop=image1[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
      p11=os.path.join(save_path,str(i)+'.png')
      cv2.imwrite(p11,crop)
      cen=[center[0]/640,center[1]/480]
      d={'feature': feats[i],'center':cen,'label':ll,'box':box}
      p22=os.path.join(save_path,str(i))
      np.save(p22, d)
      
 v = Visualizer(image1[:,:,::-1],
              scale=1,
              metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
              )
 v = v.draw_instance_predictions(pred_instances[0]['instances'].to("cpu"))
  
 box=pred_instances[0]['instances'].pred_boxes.tensor.cpu().detach().numpy()   
 cv2.imshow('image', v.get_image()[:, :, ::-1])
 cv2.waitKey(0)

 data,boxes=create_graph(temp_f)
 result=output(data,Encoder,Decoder,image1,boxes)
 cv2.imshow('result',result)
 cv2.waitKey(0)
      
 
  

 

