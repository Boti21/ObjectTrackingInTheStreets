from ultralytics import YOLO
import numpy as np
import cv2

import os
import time


from OTITS.DATA import *
from OTITS.OD import *
from OTITS.EVAL import *

from OTITS import disparity_module
from OTITS.disparity_module import DisparityModule_SGM
from OTITS.three_d_pos_calc import calculate_3d_position

N = 200 # Frames to run

if __name__ == "__main__":
    
    # Init models & feature extractors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO("OTITS/models/best.pt").to(device)
    
    des_classes = [0,1,2]
    label_path = os.path.join("data","34759_final_project_rect",f"seq_{SEQ:02d}","labels.txt")
    annotations = load_ground_truth(label_path) 

    # Init disparity module
    focal_length = 7.070493e+02
    baseline = 0.54
    disparity_module = DisparityModule_SGM(focal_length, baseline)

    # Initiate ByteTrack
    pedestrian_tracker = BYTETrack(0)
    cyclist_tracker = BYTETrack(100)
    car_tracker = BYTETrack(200)
    
    
    for k in range(N): # for frame in video
        #################################
        #           Tracking            #
        #################################
        # Get left image
        left_path = get_left_image_path(k)
        img_left = cv2.imread(left_path)
        right_path = get_right_image_path(k)
        img_right = cv2.imread(right_path)

        depth_map = disparity_module.compute_depth_map(img_left, img_right)


        ### Perform Inference ###
        results = model(left_path, classes=des_classes, conf=YOLOCONF)
        res = results[0]

        # Now you get full YOLO Results objects per class
        D_k_pedestrians = filter_results_by_class(res, 2)  # person
        D_k_cyclists    = filter_results_by_class(res, 1)  # bicycle
        D_k_cars        = filter_results_by_class(res, 0)  # car
        
        T = []
        T_pedestrian = pedestrian_tracker.step(D_k_pedestrians,img_left,depth_map)
        T.extend(T_pedestrian)
        
        T_cyclist = cyclist_tracker.step(D_k_cyclists,img_left,depth_map)
        T.extend(T_cyclist)

        T_car = car_tracker.step(D_k_cars,img_left,depth_map)
        T.extend(T_car)
       
        #################################
        #           Evaluation          #
        #################################
        frame_annotations = get_frame_annotations(annotations, k)
        #rmse_ped = rmse_xyz(T_pedestrian, frame_annotations)
        #rmse_cyc = rmse_xyz(T_cyclist, frame_annotations)
        #rmse_car = rmse_xyz(T_car, frame_annotations)
        

        preds = []
        for tr in T:  
            preds.append({
                "bbox": xysr_to_xyxy(tr.kf.x[:4]),   
                "class": tr.cls,
            })

        metrics = precision_recall(preds, frame_annotations, ["Car","Cyclist","Pedestrian"])
        print(f"Frame {k}: {metrics}")

        #################################
        #          Visualization        #
        #################################
        #img_left = draw_ground_truth(img_left, annotations, k)
        #img_left = draw_yolo_predictions(img_left, results)
        img_left = draw_tracklets(img_left,T)
        img_left = add_cls_count(img_left,T)
        
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

