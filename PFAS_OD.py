from ultralytics import YOLO
import numpy as np
import cv2

import os
import time


from OTITS.DATA import *
from OTITS.OD import *

N = 200 # Frames to run
LOCAL_AREA_SCALE = 0.2
if __name__ == "__main__":
    
    # Init models & feature extractors
    sift = cv2.SIFT_create()

    model = YOLO("yolo11x.pt")
    des_classes = [0, 1, 2]
    label_path = "34759_final_project_rect\seq_01\labels.txt" if SEQ == 1 else "34759_final_project_rect\\seq_02\\labels.txt"
    annotations = load_ground_truth(label_path) 

    
    # Implementing ByteTrack
    tau = 0.6
    YOLOCONF = 0.1
    DEADALIVERATIO = 0.25 # Not used
    DEAD_TIME = 90
    T = [] # tracklets

    base_id = 0 # ID to start counting from

    for k in range(N): # for frame in video
        # LEFT IMAGE
        left_path = get_left_image_path(k)
        img_left = cv2.imread(left_path)

        ### Perform Inference ###
        D_k = model(source=left_path, classes=des_classes,conf=YOLOCONF)
        
        D_high = []
        D_low = []
        ### Apply tau-threshold ###
        for d in D_k[0].boxes:
            box = d.xyxy[0].cpu().numpy()
            conf = float(d.conf)
            cls  = int(d.cls)
            det = (box, conf, cls)
            if conf > tau:
                D_high.append(det)
            else:
                D_low.append(det)


        
        ### Predict new locations of tracks ###
        for t in T:
            t.kf.predict()
        
        ### First Association ###
        matches = find_matches_with_hungarian_algorithm(T, D_high,
                                                        similarity="IoU",
                                                        img=img_left,
                                                        sift=sift)

        # Keep track of which detections and tracks were used
        matched_track_ids = set()
        matched_det_ids = set()

        for i, j in matches:
            # Update track i with detection j
            bbox = D_high[j][0]  # detection bbox [x1,y1,x2,y2]
            z = xyxy_to_xysr(bbox)
            T[i].kf.update(z)  # Kalman filter correction step
            
            local_area = get_local_area(img_left,bbox,scale=LOCAL_AREA_SCALE)
            T[i].local_area = local_area
            gray = cv2.cvtColor(local_area,cv2.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray, None)
            T[i].local_features = des
            

            matched_track_ids.add(i)
            matched_det_ids.add(j)

        # Unmatched detections
        D_remain = [d for idx, d in enumerate(D_high) if idx not in matched_det_ids]

        # Unmatched tracks
        T_remain = [t for idx, t in enumerate(T) if idx not in matched_track_ids]

        # Reset time not seen count
        for idx, t in enumerate(T):
            if idx in matched_track_ids:
                t.time_not_seen = 0
                

        ### Second Association ###
        matches = find_matches_with_hungarian_algorithm(T_remain, D_low)

        # Keep track of which detections and tracks were used
        matched_track_ids = set()
        matched_det_ids = set()

        for i, j in matches:
            # Update track i with detection j
            bbox = D_low[j][0] # detection bbox [x1,y1,x2,y2]
            z = xyxy_to_xysr(bbox)
            T_remain[i].kf.update(z)  # Kalman filter correction step

            local_area = get_local_area(img_left,bbox,scale=LOCAL_AREA_SCALE)
            T_remain[i].local_area = local_area
            gray = cv2.cvtColor(local_area,cv2.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray, None)
            T_remain[i].local_features = des
            

            matched_track_ids.add(i)
            matched_det_ids.add(j)

        # Unmatched tracks in D_low, further unmatched BBoxes will be deleted
        T_reremain = [t for idx, t in enumerate(T) if idx not in matched_track_ids]

        for t in T_reremain:
            t.time_not_seen += 1

        alive_tracks = []
        for t in T:
            t.time_alive += 1
            if t.time_not_seen > DEAD_TIME:
            # if (t.time_not_seen / t.time_alive) > DEADALIVERATIO:
                print(f"Killing Tracklet wit ID {t.id}, and class: {t.cls}!")
                continue  
            else:
                alive_tracks.append(t)

        # Replace T with only surviving tracks
        T = alive_tracks

        ### Initialize New Tracklets ###
        for d in D_remain:
            base_id += 1
            bbox, conf, cls = d

            local_area = get_local_area(img_left,bbox,scale=LOCAL_AREA_SCALE)
            cv2.imshow("Local Area of Tracklet", local_area)
            gray = cv2.cvtColor(local_area,cv2.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray, None)
            local_features = des
            T.append(Tracklet(id=base_id, 
                              z=xyxy_to_xysr(bbox), 
                              cls=cls,
                              local_area=local_area,
                              local_features=local_features)
                              )


        # Visualization
        #img_left = draw_ground_truth(img_left, annotations, k)
        #img_left = draw_yolo_predictions(img_left, D_k)
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

