import numpy as np
import tensorflow as tf
# from tensorflow.python.platform import flags
# from tensorflow.python.training import monitored_session
from datetime import datetime
import imutils
from object_detection.utils import label_map_util
#import label_map_util
from object_detection.user_utils import return_coordinates, return_coordinates5
#from object_detection.utils import visualization_utils as vis_util
import os
import csv
import pandas as pd
import time
import cv2
import csv
import asyncio
#import shape
import math
import queue
import threading, time
#from obj_tracking.sort import Sort
import global_var
#from pymongo import MongoClient
import collections

object_attribute_list = np.array([2,3,4,5])

class ojectDetector(object):

    def __init__(self,obj_saved_model_path):
        self.sess = tf.compat.v1.Session()

        load_graph = tf.saved_model.loader.load(
        self.sess,
        tags=["serve"],
        export_dir=obj_saved_model_path
        )
        detection_graph = tf.get_default_graph()
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self.boxes_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.scores_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
        self.classes_tensor = detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections_tensor = detection_graph.get_tensor_by_name('num_detections:0')

    def detect(self, image):
        image_np_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num_detections) = self.sess.run(
                [self.boxes_tensor, self.scores_tensor, self.classes_tensor, self.num_detections_tensor],
                feed_dict={self.image_tensor: image_np_expanded})
        return boxes, scores, classes, num_detections

class ocrReader(object):

    def __init__(self,ocr_model_model_path):
        self.sess_ocr = tf.compat.v1.Session()
        tf.saved_model.loader.load(
            self.sess_ocr,
            tags=["serve"],
            export_dir=ocr_model_model_path
        )
        self.ocr_image_tensor = self.sess_ocr.graph.get_tensor_by_name('Placeholder:0')
        self.ocr_output_tensor = self.sess_ocr.graph.get_tensor_by_name('AttentionOcr_v1/ReduceJoin:0')

    def extractText(self, ocr_images_data):
        predictions = self.sess_ocr.run(
                                        self.ocr_output_tensor,
                                        feed_dict={self.ocr_image_tensor: ocr_images_data})
        return predictions


def adjust_gamma(image, gamma=2.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def detection_boxes( boxes, classes, scores,category_index, img_shape,use_normalized_coordinates=False, 
                        display_str_percentaage=False ,max_boxes_to_draw=20, min_score_thresh=.5):
        object_id_lst = np.empty(shape=[0, 6])
        object_attr_lst = np.empty(shape=[0, 6])
        width, height = img_shape
        if not max_boxes_to_draw:
            max_boxes_to_draw = boxes.shape[0]
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            if scores is None or scores[i] > min_score_thresh:
                ymin, xmin, ymax, xmax = boxes[i].tolist()
                ymin, xmin, ymax, xmax = int(ymin*height), int(xmin*width), int(ymax*height), int(xmax*width)
                display_str = ''
                if classes[i] in category_index.keys():
                    class_name = category_index[classes[i]]['name']
                    class_id = int(classes[i])
                else:
                    class_name = 'N/A'
                display_str = str(class_name) 
                if display_str_percentaage:
                    if not display_str:
                        display_str = '{}%'.format(int(100*scores[i]))
                    else:
                        display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
                # lst = np.array([int(xmin),int(ymin),int(xmax),int(ymax),class_id,display_str])
                lst = np.array([xmin,ymin,xmax,ymax,class_id,display_str],dtype=object)
                if class_id in object_attribute_list:
                    object_attr_lst = np.vstack((object_attr_lst,lst))
                    #print(ymin, xmin, ymax, xmax,display_str)
                else:
                    object_id_lst = np.vstack((object_id_lst,lst))
        return object_id_lst, object_attr_lst


class obj_Inference(object):

    def __init__(self,obj_saved_model_path,obj_label_path,ocr_model_model_path,des_height, des_width,NUM_CLASSES):
        self.objclient = ojectDetector(obj_saved_model_path)
        self.ocrclient = ocrReader(ocr_model_model_path)
        label_map = label_map_util.load_labelmap(obj_label_path)
        categories = label_map_util.convert_label_map_to_categories(
                            label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        self.des_height = des_height
        self.des_width = des_width

    def leddar_detect_speed(self,image_np_rgb,speed):
        ocrDBJson = []
        ocr_images_data = np.ndarray(shape=(32, 150, 600, 3),dtype='uint8')
        boxes, scores, classes, num_detections = self.objclient.detect(image_np_rgb)
        object_id_list, object_attr_list = detection_boxes(
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    self.category_index,min_score_thresh=.75,img_shape=(self.des_width,self.des_height)
                )
        object_attr_list = object_attr_list[np.lexsort((object_attr_list[:,1],object_attr_list[:,0]))]
        for j in object_id_list:
            cv2.rectangle(image_np_rgb,(int(j[0]),int(j[1])),(int(j[2]),int(j[3])),(0,255,255))
        for i in object_attr_list:
            cv2.rectangle(image_np_rgb,(int(i[0]),int(i[1])),(int(i[2]),int(i[3])),(0,255,255))
            if i[4] == 2:
                e_xmin = int(i[0])
                e_ymin = int(i[1])
                e_xmax = int(i[2])
                e_ymax = int(i[3])
                img = image_np_rgb[e_ymin:e_ymax, e_xmin:e_xmax]
                rs,cs = img.shape[:2]
                imb = np.zeros((cs+30, cs+30, 3), np.uint8)
                rb,cb = imb.shape[:2]
                top = (rb-rs)/2
                bottom = rb - rs - top
                left = (cb-cs)/2
                right = cb - cs - left
                img1 = cv2.copyMakeBorder(img,int(top),int(bottom),int(left),int(right),cv2.BORDER_REPLICATE)
                imgGama = adjust_gamma(img1)
                img1 = cv2.resize(img1,(150,150))
                img2 = cv2.resize(imgGama,(150,150))
                img3 = imutils.rotate(img1, -15)
                imgBlur = cv2.GaussianBlur(imgGama,(3,3),0)
                img4 = cv2.resize(imgBlur,(150,150))
                img5 = np.concatenate((img1,img2,img4,img3),axis=1)
                ocr_images_data[0, ...] = np.asarray(img5)
                predictions = self.ocrclient.extractText(ocr_images_data)
                print(predictions[0].decode("ascii", errors="ignore"))
                z_str=predictions[0].decode("ascii", errors="ignore")
                est_np=str(z_str).replace(" ","")
                ocrWriteJson = {
                                    'entityID': i[4],
                                    'entity_imageURL': 'IMAGE URL',
                                    'entity_detectionArray': [e_xmin, e_ymin, e_xmax, e_ymax],
                                    'np_string': est_np,
                                    'videoID': 'videoID:78',
                                    'frameID': 'frameID:605',
                                    'timeStmap': '2020-05-05:020502',
                                    'speedMark': speed[count_image]
                                }
                ocrDBJson.append(ocrWriteJson)
        cv2.imshow('image_np_rgb',image_np_rgb)
        cv2.waitKey(-1)
        
        print(object_id_list, object_attr_list)
        print("================")



# client = MongoClient("mongodb://127.0.0.1:27017")
# db = client['leddar_db']
# ocr_collection = db['ocr_reader']
# ocr_error_collection = db['leddar_error']
