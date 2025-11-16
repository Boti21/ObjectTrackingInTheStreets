import cv2
from ultralytics import YOLO

import numpy as np
from scipy.optimize import linear_sum_assignment

from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag

MATCHTHRESHOLD = 0.2

dt = 1
point_model = np.array([[1, dt],
                  [0, 1]])
# State-transition Matrix for u,v,s,r,u_dot,v_dot,s_dot
F = np.array([[1, 0, 0, 0, dt, 0, 0], # u
              [0, 1, 0, 0, 0, dt, 0], # v
              [0, 0, 1, 0, 0, 0, dt], # s
              [0, 0, 0, 1, 0, 0, 0], # r
              [0, 0, 0, 0, 1, 0, 0], # u_dot
              [0, 0, 0, 0, 0, 1, 0], # v_dot
              [0, 0, 0, 0, 0, 0, 1], # s_dot
            ])
H = np.array([[1,0,0,0,0,0,0],  # x
              [0,1,0,0,0,0,0],  # y
              [0,0,1,0,0,0,0],  # s
              [0,0,0,1,0,0,0],  # r
            ])
"""
F = block_diag(point_model,point_model,point_model,point_model) # State-transition Matrix for xyxy
H = np.array([[1,0,0,0,0,0,0,0],                                # Measurment Model
              [0,0,1,0,0,0,0,0],
              [0,0,0,0,1,0,0,0],
              [0,0,0,0,0,0,1,0],])
"""

Q = np.diag([1,1,10,10,10,10,1]) 
R = np.diag([5,5,1,10])

class Tracklet:
    def __init__(self,id,z,cls,local_area=None,local_features=None):
        
        # Init internal Kalman Filter
        kf = KalmanFilter(dim_x=7, dim_z=4)  
        kf.F = F
        kf.H = H
        kf.x = np.array([z[0],z[1],z[2],z[3],0,0,0])

        kf.P *= 1000.0 # Large initial uncertainty
        kf.Q = Q
        kf.R = R

        self.kf = kf
        
        # Init ID
        self.id = id

        self.cls = cls

        self.time_not_seen = 0
        self.time_alive = 0

        # Features etc.
        self.local_area = local_area
        self.local_features = local_features

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


    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Apply threshold
    matches = []
    if similarity == "IoU":
        for i, j in zip(row_ind, col_ind):
            if iou(tracks[i], detections[j]) > MATCHTHRESHOLD:  # IoU threshold
                matches.append((i, j))  # track i matched with detection j
    else:
        matches = list(zip(row_ind, col_ind))
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
        bbox = xysr_to_xyxy(t.kf.x[:4])   # [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox.astype(int)

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
        # Draw ID above the box
        cv2.putText(img, f"ID:{t.id} cls:{t.cls}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


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


