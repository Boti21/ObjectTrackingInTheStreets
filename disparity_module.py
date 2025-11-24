import numpy as np
import cv2

class DisparityModule_SGM:
    def __init__(self, focal_length, baseline_meters):
        self.f = focal_length
        self.B = baseline_meters
        
        # SGM Parameters
        num_disparities = 128
        block_size = 5

        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 3 * block_size**2,
            P2=32 * 3 * block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
        )

    def compute_depth_map(self, img_left_rectified: np.ndarray, img_right_rectified: np.ndarray) -> np.ndarray:
    # Computes the dense depth map in meters (float).
        # 1. Convert to grayscale
        if len(img_left_rectified.shape) == 3:
            gray_l = cv2.cvtColor(img_left_rectified, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_right_rectified, cv2.COLOR_BGR2GRAY)
        else:
            gray_l = img_left_rectified.astype(np.uint8)
            gray_r = img_right_rectified.astype(np.uint8)
            
        # 2. Compute Raw Disparity Map (Output is disparity * 16)
        raw_disparity = self.sgbm.compute(gray_l, gray_r)

        # 3. Scaling and Filtering
        disparity_map = raw_disparity.astype(np.float32) / 16.0
        valid_mask = disparity_map > 0.0

        # 4. Calculate Depth Map (Z = (f * B) / d)
        depth_map = np.zeros_like(disparity_map)
        depth_map[valid_mask] = (self.f * self.B) / disparity_map[valid_mask]
        
        return depth_map