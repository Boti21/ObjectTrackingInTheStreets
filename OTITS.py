from ultralytics import YOLO
import numpy as np
import cv2
import glob

import os
import time


from OTITS.DATA import *
from OTITS.OD import *
from OTITS.CAM_CALIB import *
from OTITS.EVAL import *

from OTITS import disparity_module
from OTITS.disparity_module import DisparityModule_SGM
from OTITS.three_d_pos_calc import calculate_3d_position
from globals import *

N = 200 # Frames to run

calibrate = False
use_our_calib = False

if __name__ == "__main__":
    
    # Init models & feature extractors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO("OTITS/models/best.pt").to(device)
    
    des_classes = [0,1,2]
    label_path = os.path.join("data","34759_final_project_rect",f"seq_{SEQ:02d}","labels.txt")
    annotations = load_ground_truth(label_path) 

    # Initiate ByteTrack
    pedestrian_tracker = BYTETrack(0)
    cyclist_tracker = BYTETrack(100)
    car_tracker = BYTETrack(200)

    #focal_length = 7.070493e+02
    #baseline = 0.54
    #cx_pp = 604.0814
    #cy_pp = 180.5066

    print(f'Focal length: {focal_length}')
    print(f'Baseline: {baseline}')
    print(f'Cx: {cx_pp}')
    print(f'Cy: {cy_pp}')
    
    # Raw paths
    left_raw_path = ''
    right_raw_path = ''
    raw_path = "../data/34759_final_project_raw/"
    if SEQ == 1:
        left_raw_path = raw_path + "seq_01/image_02/data/"
        right_raw_path = raw_path + "seq_01/image_03/data/"
    elif SEQ == 2:
        left_raw_path = raw_path + "seq_02/image_02/data/"
        right_raw_path = raw_path + "seq_02/image_03/data/"
    else:
        left_raw_path = raw_path + "seq_02/image_02/data/"
        right_raw_path = raw_path + "seq_02/image_03/data/"


    our_calib_left_path = os.path.join("data", "34759_final_project_raw",f"seq_{SEQ:02d}","image_02","data/")
    our_calib_left_images = sorted(glob.glob(our_calib_left_path + '*.png'))

    our_calib_right_path = os.path.join("data", "34759_final_project_raw",f"seq_{SEQ:02d}","image_03","data/")
    our_calib_right_images = sorted(glob.glob(our_calib_left_path + '*.png'))



    # Calibrating camera
    left_calib_path = 'data/34759_final_project_raw/calib/image_02/data/'
    right_calib_path = 'data/34759_final_project_raw/calib/image_03/data/'

    if calibrate:
        ret_stereo, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = calibrate_cameras(left_raw_path=left_calib_path,
                                                                             right_raw_path=right_calib_path,
                                                                             verbose_calib=False)
    else:
        mtx_l = np.array([[990.54279248, 0, 683.62700852],
                    [0, 990.04449073, 278.94829591],
                    [0, 0, 1]])

        dist_l = np.array([[-3.67497084e-01],
                     [4.13810215e-01],
                     [4.17168231e-04],
                     [-2.23502645e-03],
                     [-4.57329160e-01]])

        mtx_r = np.array([[939.09699311, 0, 692.92619815],
                    [0, 945.47266784, 268.42592101],
                    [0, 0, 1]])

        dist_r = np.array([[-3.42502670e-01],
                    [2.40074946e-01],
                    [-2.52471525e-04],
                    [1.55403226e-03],
                    [-1.36452084e-01]])

        R = np.array([[0.99950091, 0.01952748, -0.02483177],
              [-0.01933429,  0.9997811, 0.00799613],
              [0.02498248, -0.00751204, 0.99965966]])
  
        T = np.array([[-0.55931108],
              [0.01640622],
              [0.05854456]])



    row, col, _ = cv2.imread(our_calib_left_images[0]).shape
    img_shape = (col, row)
    map_l_x, map_l_y, map_r_x, map_r_y, P1, P2 = get_rect_map(mtx_l,
                                                      dist_l,
                                                      mtx_r, 
                                                      dist_r, 
                                                      R, T,
                                                      img_shape, 
                                                      0.0)

    if use_our_calib:
        focal_length = P1[0][0]
        cx_pp = P1[0, 2]
        cy_pp = P1[1, 2]
        if np.abs(P1[0][3]) > np.abs(P2[0][3]):
            baseline = np.abs(P1[0][3]) / focal_length
        else:
            baseline = np.abs(P2[0][3]) / focal_length
    
    print(f'Focal length: {focal_length}')
    print(f'Baseline: {baseline}')
    print(f'Cx: {cx_pp}')
    print(f'Cy: {cy_pp}')

    # Init disparity module
    disparity_module = DisparityModule_SGM(focal_length, baseline)
    
    for k in range(N): # for frame in video
        #################################
        #           Tracking            #
        #################################
        left_path = get_left_image_path(k)
        img_left = cv2.imread(left_path)
        right_path = get_right_image_path(k)
        img_right = cv2.imread(right_path)

        # not an efficient solution but this is python
        if use_our_calib:
            img_left = cv2.imread(our_calib_left_images[k])
            img_right = cv2.imread(our_calib_right_images[k])
        
            # Rectification
            img_left, img_right = rect_img_pair(img_left, img_right,
                                            map_l_x, map_l_y,
                                            map_r_x, map_r_y)

        cv2.imshow("rect images", img_left)

        # Disparity
        depth_map = disparity_module.compute_depth_map(img_left, img_right)
        debug_depth_map = cv2.normalize(depth_map, None, 255, 0, cv2.NORM_MINMAX)
        debug_depth_map = np.uint8(255 - depth_map)
        debug_depth_map = cv2.applyColorMap(debug_depth_map, cv2.COLORMAP_BONE)
        cv2.imshow("depth map", debug_depth_map)


        # LEFT IMAGE



        ### Perform Inference ###
        results = model(img_left, classes=des_classes, conf=YOLOCONF)
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

