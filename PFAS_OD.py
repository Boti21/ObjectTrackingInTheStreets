from ultralytics import YOLO
import numpy as np
import cv2

import os
import time


from OTITS.DATA import *
from OTITS.OD import *

N = 45 # Frames to run

if __name__ == "__main__":
    model = YOLO("yolo11x.pt")
    des_classes = [0, 1, 2]
    label_path = "34759_final_project_rect\seq_01\labels.txt" if SEQ == 1 else "34759_final_project_rect\\seq_02\\labels.txt"
    annotations = load_ground_truth(label_path) 

    prev_detections = None
    # Implementing ByteTrack
    tau = 0.6
    YOLOCONF = 0.4
    DEADALIVERATIO = 0.25 # Not used
    DEAD_TIME = 10
    T = [] # tracklets

    base_id = 0 # ID to start counting from

    for k in range(N): # for frame in video
        # LEFT IMAGE
        
        ### Perform Inference ###
        left_path = get_left_image_path(k)
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
        matches = find_matches_with_hungarian_algorithm(T, D_high)

        # Keep track of which detections and tracks were used
        matched_track_ids = set()
        matched_det_ids = set()

        for i, j in matches:
            # Update track i with detection j
            z = D_high[j][0]  # detection bbox [x1,y1,x2,y2]
            T[i].kf.update(z)  # Kalman filter correction step
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
            z = D_low[j][0] # detection bbox [x1,y1,x2,y2]
            T[i].kf.update(z)  # Kalman filter correction step
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
            T.append(Tracklet(id=base_id, z=bbox, cls=cls))


        # Visualization
        img_left = cv2.imread(left_path)
        img_left = draw_ground_truth(img_left, annotations, k)
        #img_left = draw_yolo_predictions(img_left, D_k)
        img_left = draw_tracklets(img_left,T)
        cv2.imshow("Left Image with Ground Truth + YOLO", img_left)

        cv2.waitKey(100)

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

