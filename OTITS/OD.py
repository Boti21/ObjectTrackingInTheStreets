import numbers as np
import cv2
from ultralytics import YOLO

import numpy as np
from scipy.optimize import linear_sum_assignment

from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag

dt = 1
point_model = np.array([[1, dt],
                  [0, 1]])
F = block_diag(point_model,point_model,point_model,point_model) # State-transition Matrix
H = np.array([[1,0,0,0,0,0,0,0],                                # Measurment Model
              [0,0,1,0,0,0,0,0],
              [0,0,0,0,1,0,0,0],
              [0,0,0,0,0,0,1,0],])

Q = np.eye(8) * 0.1 
R = np.eye(4) * 1

class Tracklet:
    def __init__(self,id,z,cls):
        
        # Init internal Kalman Filter
        kf = KalmanFilter(dim_x=8, dim_z=4)  
        kf.F = F
        kf.H = H
        kf.x = np.array([z[0],0,z[1],0,z[2],0,z[3],0])

        kf.P *= 1000.0 # Large initial uncertainty
        kf.Q = Q
        kf.R = R

        self.kf = kf
        
        # Init ID
        self.id = id

        self.cls = cls

        self.time_not_seen = 0
        self.time_alive = 0

def iou(track, detection):
    boxA = track.kf.H @ track.kf.x # Access current bbox estimate
    boxB = detection[0]
    # box format: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def find_matches_with_hungarian_algorithm(tracks, detections, verbose=False):

    # Build cost matrix
    cost_matrix = np.zeros((len(tracks), len(detections)))
    for i, t in enumerate(tracks):
        for j, d in enumerate(detections):
            cost_matrix[i, j] = 1 - iou(t, d)

    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Apply threshold
    matches = []
    for i, j in zip(row_ind, col_ind):
        if iou(tracks[i], detections[j]) > 0.1:  # IoU threshold
            matches.append((i, j))  # track i matched with detection j

    if verbose:
        print("Matches:", matches)
    
    return matches


def draw_yolo_predictions(image, results, draw_color='left'):
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{results[0].names[cls_id]} {conf:.2f}"
        color = (0, 255, 255) if draw_color == 'left' else (0,0,255) # Yellow for left YOLO predictions and red for right pred
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def draw_tracklets(img, tracklets, color=(255, 255, 0)):
    for t in tracklets:
        # Get current bbox estimate from Kalman filter
        bbox = t.kf.H @ t.kf.x   # [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox.astype(int)

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
        # Draw ID above the box
        cv2.putText(img, f"ID:{t.id} cls:{t.cls}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img
