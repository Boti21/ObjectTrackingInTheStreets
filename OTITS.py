from ultralytics import YOLO
import numpy as np
import cv2

import os
import time


from OTITS.DATA import *
from OTITS.OD import *

N = 200 # Frames to run

if __name__ == "__main__":
    
    # Init models & feature extractors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO("OTITS/models/best.pt").to(device)
    
    des_classes = [0,1,2]
    label_path = "data\\34759_final_project_rect\\seq_01\\labels.txt" if SEQ == 1 else "data\\34759_final_project_rect\\seq_02\\labels.txt"
    annotations = load_ground_truth(label_path) 

    
    # Initiate ByteTrack
    pedestrian_tracker = BYTETrack(0)
    cyclist_tracker = BYTETrack(100)
    car_tracker = BYTETrack(200)
    
    
    for k in range(N): # for frame in video
        # LEFT IMAGE
        left_path = get_left_image_path(k)
        img_left = cv2.imread(left_path)

        ### Perform Inference ###
        results = model(left_path, classes=des_classes, conf=YOLOCONF)
        res = results[0]

        # Now you get full YOLO Results objects per class
        D_k_pedestrians = filter_results_by_class(res, 2)  # person
        D_k_cyclists    = filter_results_by_class(res, 1)  # bicycle
        D_k_cars        = filter_results_by_class(res, 0)  # car
        
        T = []
        T_pedestrian = pedestrian_tracker.step(D_k_pedestrians,img_left)
        T.extend(T_pedestrian)
        
        T_cyclist = cyclist_tracker.step(D_k_cyclists,img_left)
        T.extend(T_cyclist)

        T_car = car_tracker.step(D_k_cars,img_left)
        T.extend(T_car)
       
       
        # Visualization
        #img_left = draw_ground_truth(img_left, annotations, k)
        #img_left = draw_yolo_predictions(img_left, results)
        img_left = draw_tracklets(img_left,T)
        cv2.imshow("Left Image with Ground Truth + YOLO", img_left)

        cv2.waitKey(25)

    cv2.destroyAllWindows()



    """
        # RIGHT IMAGE
        right_path = get_right_image_path(i)
        results_right = model(source=right_path, classes=des_classes)
        img_right = cv2.imread(right_path)
        img_right = draw_ground_truth(img_right, annotations, i)
        img_right = draw_yolo_predictions(img_right, results_left, draw_color='left')
        img_right = draw_yolo_predictions(img_right, results_right, draw_color='right')
        
        cv2.imshow("Right Image with Ground Truth + YOLO", img_right)
    """

