from detect_class import obj_Inference
import os
import cv2
from glob import glob

inf = obj_Inference(os.getcwd() + '/obj_model_graph/v1' + '/saved_model',
                    os.getcwd() + '/obj_model_graph/v1' + '/object_detection.pbtxt',
                    os.getcwd() + '/ocr_model_graph/v1' + '/saved_model',
                    720,1280,5)
speed = [100]
xImage = glob(os.getcwd()+'/test_img/*')
for img in xImage:
    image_np_bgr = cv2.imread(img)
    image_np_rgb = cv2.cvtColor(image_np_bgr,cv2.COLOR_BGR2RGB)
    inf.leddar_detect_speed(image_np_rgb,speed)