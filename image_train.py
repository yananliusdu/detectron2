# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 21:22:58 2022

@author: 97091
"""

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
import os
import random
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog

import cv2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
if __name__ == '__main__': 
#声明类别，尽量保持
 CLASS_NAMES =["part2","part3","part4"]
# 数据集路径
 DATASET_ROOT = '/media/yanan/One Touch/detectron2_data/coco'
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
 train=MetadataCatalog.get("my_train").set(thing_classes=CLASS_NAMES,  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
                                                    evaluator_type='coco', # 指定评估方式
                                                    json_file=TRAIN_JSON,
                                                    image_root=TRAIN_PATH)
 register_coco_instances("my_val", {}, VAL_JSON, VAL_PATH)
 val=MetadataCatalog.get("my_val").set(thing_classes=CLASS_NAMES,  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
                                                    evaluator_type='coco', # 指定评估方式
                                                    json_file=VAL_JSON,
                                                    image_root=VAL_PATH)

 
 setup_logger()
 config_file =  "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
 cfg = get_cfg()

 cfg.merge_from_file(model_zoo.get_config_file(config_file))
 cfg.DATASETS.TRAIN = ("my_train",)
 cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
 cfg.DATALOADER.NUM_WORKERS = 4
    # cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
 cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
 cfg.SOLVER.IMS_PER_BATCH = 4
 cfg.SOLVER.BASE_LR = 0.0025
 cfg.SOLVER.STEPS = []
 cfg.SOLVER.MAX_ITER = (500)  # 300 iterations seems good enough, but you can certainly train longer
 cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (128)  # faster, and good enough for this toy dataset
 cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
 os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
 trainer = DefaultTrainer(cfg)
 trainer.resume_or_load(resume=False)
 trainer.train()


 cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
 cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
 dataset_dicts = DatasetCatalog.get("my_val")
 predictor = DefaultPredictor(cfg)
 for d in random.sample(dataset_dicts, 3):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                    metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), 
                    scale=0.5, 
                    instance_mode=ColorMode.IMAGE_BW 
                    # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('rr',out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
