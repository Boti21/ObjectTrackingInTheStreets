import numpy as np
import math

def calculate_3d_position(
    cx: float, 
    cy: float, 
    f: float, 
    cx_pp: float, 
    cy_pp: float, 
    width: float, 
    height: float, 
    depth_map: np.ndarray
) -> tuple:
    """
    Calculates the 3D position (X, Y, Z_avg) of a single tracked object.
        cx, cy (float): Center pixel coordinates of the object.
        f (float): Rectified focal length (in pixels).
        cx_pp, cy_pp (float): Principal Point coordinates (c_x, c_y) (in pixels).
        width, height (float): Bounding box dimensions (in pixels).
        depth_map (np.ndarray): Dense depth map in meters.
    Returns:
        tuple: (X, Y, Z_avg) position in camera coordinates (meters).
    """
    
    # 1. Depth Sampling Logic (Average Z)
    
    # Calculate sampling window side length (using requested 0.05 * Area scaling)
    bndbox_area = width * height
    sampling_area_scaled = bndbox_area * 0.05
    
    # Calculate a side length for a square sampling patch
    window_side_len = int(math.sqrt(sampling_area_scaled)) 
    
    # Oddd window size for centering and minimum size
    window_side_len = max(3, window_side_len) 
    if window_side_len % 2 == 0:
        window_side_len += 1 

    half_window = window_side_len // 2
    
    # Define pixel coordinates for the sampling patch
    x_start = int(cx) - half_window
    x_end = int(cx) + half_window + 1
    y_start = int(cy) - half_window
    y_end = int(cy) + half_window + 1

    # Clamp coordinates to image boundaries
    H, W = depth_map.shape
    x_start = np.clip(x_start, 0, W)
    x_end = np.clip(x_end, 0, W)
    y_start = np.clip(y_start, 0, H)
    y_end = np.clip(y_end, 0, H)

    # Extract the depth patch and calculate average depth (Z_avg)
    depth_patch = depth_map[y_start:y_end, x_start:x_end]
    valid_depths = depth_patch[depth_patch > 0.0]

    if valid_depths.size == 0:
        Z_avg = 0.0
    else:
        # Use mean for averaging depth as specified by the user
        Z_avg = np.mean(valid_depths) 

    # 2. 3D Position Calculation (Pinhole Camera Model)
    # X = (u - c_x) * Z / f  and Y = (v - c_y) * Z / f

    if Z_avg > 0:
        X = (cx - cx_pp) * Z_avg / f
        Y = (cy - cy_pp) * Z_avg / f
    else:
        # If Z is invalid, the 3D position is unknown (output 0, 0, 0)
        X, Y = 0.0, 0.0

    return X, Y, Z_avg