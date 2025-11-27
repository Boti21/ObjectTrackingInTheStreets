import cv2
from ultralytics import YOLO

import numpy as np
from scipy.optimize import linear_sum_assignment

from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag

import random
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity


import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from OTITS.disparity_module import *
from OTITS.three_d_pos_calc import *
from globals import *

# ByteTrack Params
tau_pedestrians = 0.5
tau_cyclists = 0.85#0.79
tau_cars = 0.5 #seq2 0.8
YOLOCONF = 0.1
YOLOCONF_pedestrians = 0.4
YOLOCONF_cyclists = 0.5
YOLOCONF_cars = 0.2 #seg2:0.5
MATCHTHRESHOLD_IoU = 0.05#0.05
MATCHTHRESHOLD_ReID = 0.4
DEADALIVERATIO = 0.25 # Not used
DEAD_TIME_cars = 20#se2:40
DEAD_TIME_pedestrians = 40#se2:40
DEAD_TIME_cyclists = 25#se2:40
DEAD_TIME_NOT_OCCLUDED = 10
LOCAL_AREA_SCALE = 0.8
FIRST_ASSOCIATION_METRIC = "IoU"# #"IoU", "ReID"

# Rebirth params
REBIRTH_DIST_THRESHOLD = 400#seq2100 #300  # pixels
REBIRTH_LOOK_AHEAD = 30 #seq2:30
REBIRTH_TIME_NOT_SEEN = 4
# Kalman Params
USE_ACCEL = False
Q =np.diag([1,1,1,1,0.01,0.01,1,0.0001]) 
R = np.diag([10,10,100,100,100])
Q_accel =np.diag([1,1,1,1,0.000001,0.0000001,1,0.000001,0.0000001,0.0000001]) 
R_accel = np.diag([10,10,1,1,1])
base_id = 0 # ID to start counting from



CLASS_MAP = {0: "Car", 1: "Cyclist", 2: "Pedestrian"}

# Camera Calibration Constants (Placeholder values until calibration module is done
#focal_length = 707.0493
#baseline = 0.54
#cx_pp = 604.0814
#cy_pp = 180.5066




if FIRST_ASSOCIATION_METRIC == "ReID":
    # Load pretrained ResNet50
    resnet50 = models.resnet50(pretrained=True)

    # Remove the final classification layer (fc) as feature extractor
    resnet50 = nn.Sequential(*list(resnet50.children())[:-1])  # outputs 2048‑D feature vector

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet50 = resnet50.to(device)
    resnet50.eval()

    # Preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),   # ResNet expects 224×224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])


def get_resnet50_embedding(img_crop):
    # img_crop: numpy array (BGR from OpenCV)
    img_rgb = Image.fromarray(img_crop[..., ::-1])  # convert BGR→RGB
    tensor = preprocess(img_rgb).unsqueeze(0).to(device)  # add batch dimension, move to GPU
    with torch.no_grad():
        emb = resnet50(tensor)                      # shape [1, 2048, 1, 1]
    return emb.squeeze().cpu().numpy()              # move back to CPU, shape [2048]



dt = 1
point_model = np.array([[1, dt],
                  [0, 1]])
# State-transition Matrix for u,v,s,r,u_dot,v_dot,s_dot
"""
F = np.array([[1, 0, 0, 0, dt, 0, 0.5*dt**2, 0], # u
              [0, 1, 0, 0, 0, dt, 0, 0.5*dt**2], # v
              [0, 0, 1, 0, 0, 0, 0, 0], # s 
              [0, 0, 0, 1, 0, 0, 0, 0], # r
              [0, 0, 0, 0, 1, 0, dt, 0], # u_dot
              [0, 0, 0, 0, 0, 1, 0, dt], # v_dot
              [0, 0, 0, 0, 0, 0, 1, 0], # u_dotdot
              [0, 0, 0, 0, 0, 0, 0, 1], # v_dotdot
            ])
H = np.array([[1,0,0,0,0,0,0,0],  # x
              [0,1,0,0,0,0,0,0],  # y
              [0,0,1,0,0,0,0,0],  # s
              [0,0,0,1,0,0,0,0],  # r
            ])
"""
F = np.array([[1, 0, 0, 0, dt, 0,0,0], # u
              [0, 1, 0, 0, 0, dt,0,0], # v
              [0, 0, 1, 0, 0, 0,0,0], # s Area
              [0, 0, 0, 1, 0, 0,0,0], # r Aspect ratio
              [0, 0, 0, 0, 1, 0,0,0], # u_dot
              [0, 0, 0, 0, 0, 1,0,0], # v_dot
              [0, 0, 0, 0, 0, 0,1,0], # d
              [0, 0, 0, 0, 0, 0,0,1], # d_dot
            ])
H = np.array([[1,0,0,0,0,0,0,0],  # x
              [0,1,0,0,0,0,0,0],  # y
              [0,0,1,0,0,0,0,0],  # s
              [0,0,0,1,0,0,0,0],  # r
              [0,0,0,0,0,0,1,0],  # d
            ])
F_accel = np.array([[1, 0, 0, 0, dt, 0,0,0,0.5*dt**2,0], # u
                    [0, 1, 0, 0, 0, dt,0,0,0,0.5*dt**2], # v
                    [0, 0, 1, 0, 0, 0,0,0,0,0], # s Area
                    [0, 0, 0, 1, 0, 0,0,0,0,0], # r Aspect ratio
                    [0, 0, 0, 0, 1, 0,0,0,dt,0], # u_dot
                    [0, 0, 0, 0, 0, 1,0,0,0,dt], # v_dot
                    [0, 0, 0, 0, 0, 0,1,dt,0,0,], # d
                    [0, 0, 0, 0, 0, 0,0,1,0,0], # d_dot
                    [0, 0, 0, 0, 0, 0,0,0,1,0], # u_ddot
                    [0, 0, 0, 0, 0, 0,0,0,0,1], # v_ddot
                    ])
H_accel = np.array([[1,0,0,0,0,0,0,0,0,0],  # x
                    [0,1,0,0,0,0,0,0,0,0],  # y
                    [0,0,1,0,0,0,0,0,0,0],  # s
                    [0,0,0,1,0,0,0,0,0,0],  # r
                    [0,0,0,0,0,0,1,0,0,0],  # d
                    ])
"""

F = np.array([[1, 0, 0, 0, dt, 0], # u
              [0, 1, 0, 0, 0, dt], # v
              [0, 0, 1, 0, 0, 0], # s Area
              [0, 0, 0, 1, 0, 0], # r Aspect ratio
              [0, 0, 0, 0, 1, 0], # u_dot
              [0, 0, 0, 0, 0, 1], # v_dot
            ])
H = np.array([[1,0,0,0,0,0],  # x
              [0,1,0,0,0,0],  # y
              [0,0,1,0,0,0],  # s
              [0,0,0,1,0,0],  # r
            ])
"""
"""
F = block_diag(point_model,point_model,point_model,point_model) # State-transition Matrix for xyxy
H = np.array([[1,0,0,0,0,0,0,0],                                # Measurment Model
              [0,0,1,0,0,0,0,0],
              [0,0,0,0,1,0,0,0],
              [0,0,0,0,0,0,1,0],])
"""

#Q =np.diag([1,1,1,1,0.001,0.001]) 
#R = np.diag([10,10,1,1])
#Q = 0.01*np.diag([1,1,1,1,0.1,0.1,0.001,0.001]) 
#R = np.diag([100,100,1,1])

class Tracklet:
    def __init__(self,id,z,cls,xyz=np.array([0,0,0]),local_area=None,local_features=None):
        
        # Init internal Kalman Filter
        if USE_ACCEL:
            kf = KalmanFilter(dim_x=10, dim_z=5) 
            kf.F = F_accel
            kf.H = H_accel
            kf.x = np.array([z[0],z[1],z[2],z[3],0,0,z[4],0,0,0])
            kf.P *= 100.0 # Large initial uncertainty
            kf.Q = Q_accel
            kf.R = R_accel
            
        else: 
            kf = KalmanFilter(dim_x=8, dim_z=5) 
            kf.F = F
            kf.H = H
            kf.x = np.array([z[0],z[1],z[2],z[3],0,0,z[4],0])

            kf.P *= 100.0 # Large initial uncertainty
            kf.Q = Q
            kf.R = R

        self.kf = kf
        
        # Location
        self.xyz = xyz

        # Init ID & other attributes
        self.id = id
        self.occluded = False
        self.cls = cls

        self.time_not_seen = 0
        self.time_alive = 0

        # Features etc.
        self.local_area = local_area
        self.local_features = local_features
        self.embedding = get_resnet50_embedding(self.local_area) if FIRST_ASSOCIATION_METRIC == "ReID" else None

def iou(track, detection):
    boxA =  xysr_to_xyxy(track.kf.x[:4]) # Access current bbox estimate
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


def xyxy_to_xysr(bbox):
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    x = x1 + w / 2.0   # center x
    y = y1 + h / 2.0   # center y
    s = w * h          # scale (area)
    r = w / h          # aspect ratio
    return np.array([x, y, s, r], dtype=float)


def xysr_to_xyxy(xysr):
    x, y, s, r = xysr

    # Recover width and height from scale and aspect ratio
    if (s < 0) or (r < 0):
        print("s or r smaller than zero!")
    s = max(1e-6, s)   # ensure positive area
    r = max(1e-6, r)   # ensure positive aspect ratio
    w = np.sqrt(s * r)
    h = s / w

    # Convert to corners
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0

    return np.array([x1, y1, x2, y2], dtype=float)


def find_matches_with_hungarian_algorithm(tracks, detections, 
                                          similarity="IoU", # Could be SSD, SIFT etc.
                                          img=None, # Has to be given for everything but IoU
                                          sift=None, # Has to be given for SIFT
                                          verbose=False):

    # Build cost matrix
    cost_matrix = np.zeros((len(tracks), len(detections)))
    for i, t in enumerate(tracks):
        for j, d in enumerate(detections):
            match similarity:
                case "IoU":
                    cost_matrix[i, j] = 1 - iou(t, d)
                case "SIFT":
                    # Extract local patch for detection
                    x1, y1, x2, y2 = d[0].astype(int)
                    det_patch = img[y1:y2, x1:x2]
                    gray_patch = cv2.cvtColor(det_patch, cv2.COLOR_BGR2GRAY)

                    # Compute SIFT features
                    kp2, des2 = sift.detectAndCompute(gray_patch, None)

                    # Match against tracklet’s stored descriptors
                    bf = cv2.BFMatcher()
                    if t.local_features is not None and des2 is not None:
                        matches = bf.knnMatch(t.local_features, des2, k=2)
                        # Lowe’s ratio test
                        good = []
                        for m_n in matches:
                            if len(m_n) == 2:  # only if we have both m and n
                                m, n = m_n
                                if m.distance < 0.75 * n.distance:
                                    good.append(m)

                        # Cost = inverse of match quality
                        cost_matrix[i, j] = 1.0 - (len(good) / (len(matches)+1e-6))
                    else:
                        cost_matrix[i, j] = 1.0  # high cost if no features

                case "SAD":
                    x1, y1, x2, y2 = d[0].astype(int)
                    det_patch = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                    track_patch = cv2.cvtColor(t.local_area, cv2.COLOR_BGR2GRAY)

                    # Resize to same size
                    det_patch = cv2.resize(det_patch, (track_patch.shape[1], track_patch.shape[0]))
                    sad = np.sum(np.abs(track_patch.astype(np.float32) - det_patch.astype(np.float32)))
                    cost_matrix[i, j] = sad
                
                case "SSD":
                    x1, y1, x2, y2 = d[0].astype(int)
                    det_patch = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                    track_patch = cv2.cvtColor(t.local_area, cv2.COLOR_BGR2GRAY)

                    det_patch = cv2.resize(det_patch, (track_patch.shape[1], track_patch.shape[0]))
                    ssd = np.sum((track_patch.astype(np.float32) - det_patch.astype(np.float32))**2)
                    cost_matrix[i, j] = ssd

                case "ReID":
                    # d[0] is the bbox [x1,y1,x2,y2]
                    x1, y1, x2, y2 = d[0].astype(int)
                    det_patch = img[y1:y2, x1:x2]  # crop detection from full frame

                    # Compute embedding for detection crop
                    det_embedding = get_resnet50_embedding(det_patch)

                    # Compare with tracklet’s stored embedding
                    if hasattr(t, "embedding") and t.embedding is not None:
                        cost_matrix[i, j] = 1.0 - cosine_similarity(
                            t.embedding.reshape(1, -1),
                            det_embedding.reshape(1, -1)
                        )[0, 0]
                    else:
                        cost_matrix[i, j] = 1.0



    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Apply threshold
    matches = []
    if similarity == "IoU":
        for i, j in zip(row_ind, col_ind):
            if (iou(tracks[i], detections[j]) > MATCHTHRESHOLD_IoU):  # IoU threshold
                matches.append((i, j))  # track i matched with detection j
    
    if similarity == "ReID":
        for i, j in zip(row_ind, col_ind):
            x1, y1, x2, y2 = detections[j][0].astype(int)
            det_patch = img[y1:y2, x1:x2]  # crop detection from full frame

            # Compute embedding for detection crop
            det_embedding = get_resnet50_embedding(det_patch)

            # Cosine similarity with track’s stored embedding
            if tracks[i].embedding is not None:
                sim = cosine_similarity(
                    tracks[i].embedding.reshape(1, -1),
                    det_embedding.reshape(1, -1)
                )[0, 0]

                if sim > MATCHTHRESHOLD_ReID:
                    matches.append((i, j))
                    # refresh track embedding
                    tracks[i].embedding = det_embedding
    #else:
    #    matches = list(zip(row_ind, col_ind))

    if verbose:
        print("Matches:", matches)

    
    return matches


class BYTETrack:
    
    def __init__(self,base_id,tau,deadtime=35):
        self.T = []
        self.base_id = base_id
        self.tau = tau
        self.deadtime = deadtime

    
    def step(self, D_k,img_left,depth_map):    
        

        if D_k.boxes is None or len(D_k.boxes) == 0:
            # No detections, just predict next state for all tracks
            for t in self.T:
                t.kf.predict()
                t.time_not_seen += 1
                if t.time_not_seen > self.deadtime or (not t.occluded and t.time_not_seen > DEAD_TIME_NOT_OCCLUDED):
                    print(f"Killing Tracklet wit ID {t.id}, and class: {t.cls}!")
                    self.T.remove(t)

            return self.T.copy()

        D_high = []
        D_low = []
        ### Apply tau-threshold ###
        for d in D_k.boxes:#D_k[0].boxes:
            box = d.xyxy[0].cpu().numpy()
            conf = float(d.conf)
            cls  = int(d.cls)
            det = (box, conf, cls)
            if conf > self.tau:
                D_high.append(det)
            else:
                D_low.append(det)

 
        ### Predict new locations of tracks ###
        for t in self.T:
            t.kf.predict()
            
        
        ### First Association ###
        matches = find_matches_with_hungarian_algorithm(self.T, D_high,
                                                        similarity=FIRST_ASSOCIATION_METRIC,
                                                        img=img_left)

        # Keep track of which detections and tracks were used
        matched_track_ids = set()
        matched_det_ids = set()

        for i, j in matches:
            # Update track i with detection j
            bbox = D_high[j][0]  # detection bbox [x1,y1,x2,y2]
            z = xyxy_to_xysr(bbox)
            x1,y1,x2,y2 = bbox
            w = x2 - x1
            h = y2 - y1
            # Get depth Measurement
            _, _, Z_avg = calculate_3d_position(cx=z[0],cy=z[1],f=focal_length,cx_pp=cx_pp,cy_pp=cy_pp,width=w,height=h,depth_map=depth_map)
            z = np.append(z,Z_avg)
            self.T[i].kf.update(z)  # Kalman filter correction step
            X, Y, Z_avg = calculate_3d_position(cx=self.T[i].kf.x[0],cy=self.T[i].kf.x[1],f=focal_length,cx_pp=cx_pp,cy_pp=cy_pp,width=w,height=h,depth_map=depth_map)
            self.T[i].xyz = np.array([X,Y,Z_avg])
            
            self.T[i].occluded = False
            
            if FIRST_ASSOCIATION_METRIC == "ReID":
                local_area = get_local_area(img_left,bbox,scale=LOCAL_AREA_SCALE)
                self.T[i].local_area = local_area
                self.T[i].embedding = get_resnet50_embedding(local_area)
            #gray = cv2.cvtColor(local_area,cv2.COLOR_BGR2GRAY)
            #kp, des = sift.detectAndCompute(gray, None)
            #self.T[i].local_features = des
            

            matched_track_ids.add(i)
            matched_det_ids.add(j)

        # Unmatched detections
        D_remain = [d for idx, d in enumerate(D_high) if idx not in matched_det_ids]

        # Unmatched tracks
        T_remain = [t for idx, t in enumerate(self.T) if idx not in matched_track_ids]

        # Reset time not seen count
        for idx, t in enumerate(self.T):
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
            z=xyxy_to_xysr(bbox)
            x1,y1,x2,y2 = bbox
            w = x2 - x1
            h = y2 - y1
            # Get depth Measurement
            _, _, Z_avg = calculate_3d_position(cx=z[0],cy=z[1],f=focal_length,cx_pp=cx_pp,cy_pp=cy_pp,width=w,height=h,depth_map=depth_map)
            z = np.append(z,Z_avg)
            T_remain[i].kf.update(z)  # Kalman filter correction step
            # Get Kalman filtered xyz
            X, Y, Z_avg = calculate_3d_position(cx=T_remain[i].kf.x[0],cy=T_remain[i].kf.x[1],f=focal_length,cx_pp=cx_pp,cy_pp=cy_pp,width=w,height=h,depth_map=depth_map)
            T_remain[i].xyz = np.array([X,Y,Z_avg])

            T_remain[i].occluded = False


            if FIRST_ASSOCIATION_METRIC == "ReID":
                local_area = get_local_area(img_left,bbox,scale=LOCAL_AREA_SCALE)
                T_remain[i].local_area = local_area
                T_remain[i].embedding = get_resnet50_embedding(local_area)
            #gray = cv2.cvtColor(local_area,cv2.COLOR_BGR2GRAY)
            #kp, des = sift.detectAndCompute(gray, None)
            #T_remain[i].local_features = des
            

            matched_track_ids.add(i)
            matched_det_ids.add(j)

        # Unmatched tracks in D_low, further unmatched BBoxes will be deleted
        T_reremain = [t for idx, t in enumerate(T_remain) if idx not in matched_track_ids] # CHANGED SELF.T to T_remain

        for t in T_reremain:
            t.time_not_seen += 1
            x1, y1, x2, y2 = xysr_to_xyxy(t.kf.x[:4])
            w = x2 - x1
            h = y2 - y1
            # Check whether tracklet is occluded
            _, _, Z_avg = calculate_3d_position(cx=t.kf.x[0],cy=t.kf.x[1],f=focal_length,cx_pp=cx_pp,cy_pp=cy_pp,width=w,height=h,depth_map=depth_map)
            if t.kf.x[6] > Z_avg:
                t.occluded = True

        alive_tracks = []
        for t in self.T:
            t.time_alive += 1

            if t.time_not_seen > self.deadtime or (not t.occluded and t.time_not_seen > DEAD_TIME_NOT_OCCLUDED):
                print(f"Killing Tracklet wit ID {t.id}, and class: {t.cls}!")
                continue  
            
            else:
                alive_tracks.append(t)

        # Replace T with only surviving tracks
        self.T = alive_tracks

        ### See whether missed/occluded tracklets can be matched with new detections ###
        # --- Collect occluded tracks ---
        occluded_tracks = [t for t in self.T if t.occluded and t.time_not_seen > REBIRTH_TIME_NOT_SEEN]

        if occluded_tracks and D_remain:
            cost_matrix = np.full((len(occluded_tracks), len(D_remain)), np.inf)

            for i, t in enumerate(occluded_tracks):
                preds = predict_future_positions(t, N=REBIRTH_LOOK_AHEAD)
                for j, d in enumerate(D_remain):
                    bbox, conf, cls = d
                    cx_det = (bbox[0]+bbox[2]) / 2
                    cy_det = (bbox[1]+bbox[3]) / 2

                    distances = [np.linalg.norm([px - cx_det, py - cy_det]) for (px, py) in preds]
                    min_dist = min(distances)

                    if min_dist < REBIRTH_DIST_THRESHOLD and cls == t.cls:
                        cost_matrix[i, j] = min_dist
            
            try:
                track_indices, det_indices = linear_sum_assignment(cost_matrix)
            except:
                track_indices, det_indices = [], []

            used_dets = set()
            for i, j in zip(track_indices, det_indices):
                if cost_matrix[i, j] != np.inf:
                    t = occluded_tracks[i]
                    bbox, conf, cls = D_remain[j]

                    z = xyxy_to_xysr(bbox)
                    z = np.append(z, t.kf.x[6])  # keep depth
                    t.kf.update(z)
                    t.time_not_seen = 0
                    t.occluded = False

                    used_dets.add(j)

            D_remain = [d for idx, d in enumerate(D_remain) if idx not in used_dets]

        ### Initialize New Tracklets ###
        for d in D_remain:
            self.base_id += 1
            bbox, conf, cls = d

            local_area = get_local_area(img_left,bbox,scale=LOCAL_AREA_SCALE)
            cv2.imshow("Local Area of Tracklet", local_area)
            #gray = cv2.cvtColor(local_area,cv2.COLOR_BGR2GRAY)
            #kp, des = sift.detectAndCompute(gray, None)
            #local_features = des
            z=xyxy_to_xysr(bbox)
            x1,y1,x2,y2 = bbox
            w = x2 - x1
            h = y2 - y1
            X, Y, Z_avg = calculate_3d_position(cx=z[0],cy=z[1],f=focal_length,cx_pp=cx_pp,cy_pp=cy_pp,width=w,height=h,depth_map=depth_map)
            z = np.append(z,Z_avg)
            self.T.append(Tracklet(id=self.base_id, 
                                cls=cls,
                                z=z,
                                xyz=np.array([X,Y,Z_avg]),
                                local_area=local_area,
                                #local_features=local_features
                                ))
        return self.T.copy()

def predict_future_positions(t, N=REBIRTH_LOOK_AHEAD):
    cx, cy = t.kf.x[0], t.kf.x[1]
    vx, vy = t.kf.x[4], t.kf.x[5]
    preds = [(cx + k*vx, cy + k*vy) for k in range(1, N+1)]
    return preds


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

def id_to_color(tracklet_id, base_color=None):
    """
    Generate a reproducible distinct color per tracklet ID.
    If base_color is given, blend with it; otherwise pick a random hue.
    """
    rng = np.random.default_rng(seed=tracklet_id)  # deterministic per ID
    rand_color = rng.integers(0, 256, size=3)
    if base_color is not None:
        # Blend 50/50 with base class color
        blended = (0.5 * np.array(base_color) + 0.5 * rand_color).astype(int)
        return tuple(int(c) for c in blended)
    else:
        return tuple(int(c) for c in rand_color)
"""
def draw_tracklets(img, tracklets, color=(255, 255, 0)):
    for t in tracklets:
        # Get current bbox estimate from Kalman filter
        bbox = xysr_to_xyxy(t.kf.x[:4])   # [x1, y1, x2, y2]
        u_dot,v_dot = t.kf.x[4:6]
        time_not_seen = t.time_not_seen
        x1, y1, x2, y2 = bbox.astype(int)

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
        # Draw ID above the box
        cv2.putText(img, f"ID:{t.id} cls:{t.cls}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img
"""
def draw_tracklets(img, tracklets, velocity_scale=10, accel_scale=20):
    # Base class colors (BGR)
    class_colors = {
        2: (255, 255, 0),   # cyan
        1: (0, 255, 0),     # green
        0: (0, 0, 255)      # red
    }

    for idx, t in enumerate(tracklets):
        # Get current bbox estimate from Kalman filter
        bbox = xysr_to_xyxy(t.kf.x[:4])   # [x1, y1, x2, y2]
        u_dot, v_dot = t.kf.x[4:6]        # velocity components
        depth = t.kf.x[6]
        if USE_ACCEL:
            ax, ay = t.kf.x[7:9]              # acceleration components
        time_not_seen = t.time_not_seen
        x1, y1, x2, y2 = bbox.astype(int)

        # --- Instance-specific shade ---
        # Pick color based on class
        color = class_colors.get(t.cls, (255, 255, 255))
        #color = id_to_color(t.id, base_color)


        
        # Draw rectangle or if t.occluded than make it a dashed line
        if t.occluded:
            draw_dashed_rectangle(img, (x1, y1), (x2, y2), color, 2)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


        # Draw ID above the box
        cv2.putText(img, f"ID:{t.id} cls:{t.cls}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- Draw velocity vector (green arrow) ---
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2   # centroid
        vel_end = (int(cx + velocity_scale * u_dot),
                   int(cy + velocity_scale * v_dot))
        cv2.arrowedLine(img, (cx, cy), vel_end, (0, 255, 0), 2, tipLength=0.3)

        if USE_ACCEL:
            # --- Draw acceleration vector (purple arrow) ---
            acc_end = (int(cx + accel_scale * ax),
                    int(cy + accel_scale * ay))
            cv2.arrowedLine(img, (cx, cy), acc_end, (255, 0, 255), 2, tipLength=0.3)

        # --- Display time_not_seen if > 1 ---
        if time_not_seen > 1:
            cv2.putText(img, f"missed:{time_not_seen}", (x2+10, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # --- Display depth
        cv2.putText(img, f"Depth:{depth:.2f}m", (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

    return img
"""
def draw_tracklets(img, tracklets, velocity_scale=10):
    # Define class-dependent colors
    class_colors = {
        2: (255, 255, 0),   # cyan (BGR: blue+green)
        1: (0, 255, 0),     # green
        0: (0, 0, 255)      # red
    }

    for t in tracklets:
        # Get current bbox estimate from Kalman filter
        bbox = xysr_to_xyxy(t.kf.x[:4])   # [x1, y1, x2, y2]
        u_dot, v_dot = t.kf.x[4:6]         # velocity components
        depth = t.kf.x[6]
        time_not_seen = t.time_not_seen
        x1, y1, x2, y2 = bbox.astype(int)

        # Pick color based on class
        color = class_colors.get(t.cls, (255, 255, 255))  # default white if unknown

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw ID above the box
        cv2.putText(img, f"ID:{t.id} cls:{t.cls}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- Draw velocity vector ---
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2   # centroid
        end_point = (int(cx + velocity_scale * u_dot),
                     int(cy + velocity_scale * v_dot))
        cv2.arrowedLine(img, (cx, cy), end_point, (0, 255, 0), 2, tipLength=0.3)

        # --- Display time_not_seen if > 1 ---
        if time_not_seen > 1:
            cv2.putText(img, f"missed:{time_not_seen}", (x2, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # --- Display depth
        cv2.putText(img, f"Depth:{depth:.2f}m", (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

    return img
"""

def get_local_area(img, bbox, scale=0.5):
    
    h_img, w_img = img.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    # Center of bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # Original width/height
    w = x2 - x1
    h = y2 - y1

    # Scaled width/height
    w_scaled = int(w * scale)
    h_scaled = int(h * scale)

    # New coordinates
    x1_new = max(0, cx - w_scaled // 2)
    x2_new = min(w_img, cx + w_scaled // 2)
    y1_new = max(0, cy - h_scaled // 2)
    y2_new = min(h_img, cy + h_scaled // 2)

    # Crop patch
    if x2_new > x1_new and y2_new > y1_new:
        return img[y1_new:y2_new, x1_new:x2_new]
    else:
        return None  # invalid bbox


# Helper: filter Results by class with confidence thresholds
def filter_results_by_class(res, target_cls):
    # Map class → confidence threshold
    thresholds = {
        0: YOLOCONF_cars,
        1: YOLOCONF_cyclists,
        2: YOLOCONF_pedestrians
    }

    # Boolean mask for class and confidence
    cls_mask = res.boxes.cls.int().cpu().numpy() == target_cls
    conf_mask = res.boxes.conf.cpu().numpy() >= thresholds.get(target_cls, 0.0)

    mask = cls_mask & conf_mask

    # Create a new Results object with same metadata
    new_res = res.new()
    new_res.boxes = res.boxes[mask]
    return new_res


def add_cls_count(img_left,T):
    num_cars = sum(1 for t in T if t.cls == 0)
    num_cyclists    = sum(1 for t in T if t.cls == 1)
    num_pedestrians = sum(1 for t in T if t.cls == 2)

    # Prepare text lines
    lines = [
        f"Cars: {num_cars}",
        f"Cyclists: {num_cyclists}",
        f"Pedestrians: {num_pedestrians}"
    ]

    # Draw each line in white in the top-left corner
    y0, dy = 30, 30  # starting y and line spacing
    for i, line in enumerate(lines):
        y = y0 + i*dy
        cv2.putText(img_left, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        
    return img_left


def draw_dashed_rectangle(img, pt1, pt2, color, thickness=2, dash_length=10):
    x1, y1 = pt1
    x2, y2 = pt2

    # Top edge
    for x in range(x1, x2, dash_length*2):
        cv2.line(img, (x, y1), (min(x+dash_length, x2), y1), color, thickness)
    # Bottom edge
    for x in range(x1, x2, dash_length*2):
        cv2.line(img, (x, y2), (min(x+dash_length, x2), y2), color, thickness)
    # Left edge
    for y in range(y1, y2, dash_length*2):
        cv2.line(img, (x1, y), (x1, min(y+dash_length, y2)), color, thickness)
    # Right edge
    for y in range(y1, y2, dash_length*2):
        cv2.line(img, (x2, y), (x2, min(y+dash_length, y2)), color, thickness)


