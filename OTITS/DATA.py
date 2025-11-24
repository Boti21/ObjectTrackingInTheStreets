from ultralytics import YOLO
import numpy as np
import cv2

import os
import time

SEQ = 3

def load_ground_truth(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    return [parse_ground_truth_line(line) for line in lines]


def parse_ground_truth_line(line):
    parts = line.strip().split()
    return {
        "frame_id": int(parts[0]),
        "class": parts[2],
        "bbox": [float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9])]
    }

def draw_ground_truth(image, annotations, frame_id):
    for ann in annotations:
        if ann["frame_id"] == frame_id:
            x1, y1, x2, y2 = map(int, ann["bbox"])
            color = (0, 255, 0) if ann["class"] == "Cyclist" else (255, 0, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, ann["class"], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image






def get_left_image_path(index,seq=SEQ):
    filename = f"{index:06d}.png" if seq == 1 else f"{index:010d}.png"
    return os.path.join(f"data\\34759_final_project_rect\\seq_0{seq}\\image_02\\data", filename)

def get_right_image_path(index,seq=SEQ):
    filename = f"{index:06d}.png" if seq == 1 else f"{index:010d}.png"
    return os.path.join(f"data\\34759_final_project_rect\\seq_0{seq}\\image_03\\data", filename)

def get_left_image(index,seq=SEQ):
    return cv2.imread(get_left_image_path(index,SEQ))
def get_right_image(index,seq=SEQ):
    return cv2.imread(get_right_image_path(index,SEQ))
