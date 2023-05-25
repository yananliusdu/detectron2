# Author: Yanan Liu
# Date: 25/05/2023 14:00
# Location:
# Version:
# Description: Enter a brief description here.


#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Other necessary imports
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import cv2
import os
from detectron2.structures import Boxes

cfg = get_cfg()

CLASS_NAMES =["part2","part3","part4"]


# Define the callback for the camera subscriber
def image_callback(img_msg):
    # Convert ROS Image message to OpenCV image
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

    # Perform object detection on the image
    outputs = predictor(cv_image)

    # TODO: Add code to handle the outputs of the object detection model, e.g.,
    # visualize the results on the image, publish the results to a ROS topic, etc.


    MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes = CLASS_NAMES
    class_names = MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes

    # Get bounding boxes, classes, and calculate centers
    if outputs["instances"].has("pred_boxes") and outputs["instances"].has("pred_classes"):
        boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        classes = outputs["instances"].pred_classes.cpu().numpy()
        for box, class_index in zip(boxes, classes):
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            class_name = class_names[class_index]
            print(f"Object: {class_name}, Bounding box: ({x1}, {y1}), ({x2}, {y2}), Center: ({center_x}, {center_y})")

    # For example, to visualize the results on the image:
    v = Visualizer(cv_image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    visualized_image = out.get_image()[:, :, ::-1]
    cv2.imshow('Visualized image', visualized_image)
    cv2.waitKey(1)


def main():
    # Initialize the ROS node
    rospy.init_node('object_detection_node')

    # Initialize the object detection model
    config_file =  "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.DATASETS.TEST = ("my_val",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    global predictor
    predictor = DefaultPredictor(cfg)

    # Subscribe to the RealSense camera image topic
    rospy.Subscriber('/camera/color/image_raw', Image, image_callback)

    # Keep the node running until it's shut down
    rospy.spin()

if __name__ == '__main__':
    main()
