import numpy as np
from scipy.optimize import linear_sum_assignment


from OTITS.OD import *

def rmse_xyz(tracklets, ground_truth):
    errors = []
    for tr in tracklets:
        # Find matching ground truth by frame_id and class
        gt_objs = [gt for gt in ground_truth if gt["frame_id"] == tr.frame_id and gt["class"] == tr.cls]
        
        # Simple matching: nearest bbox center or IoU > threshold
        for gt in gt_objs:
            err = np.linalg.norm(tr.xyz - np.array(gt["location"]))
            errors.append(err)
    if errors:
        return np.sqrt(np.mean(np.square(errors)))
    return None

def detection_rate(tracklets, ground_truth, cls_name, dist_thresh=30.0):
    gt_filtered = [gt for gt in ground_truth if gt["class"] == cls_name and gt["distance"] <= dist_thresh]
    detected = 0
    
    for gt in gt_filtered:
        for tr in tracklets:
            if tr.cls == cls_name:
                if iou(tr.kf.x[:4], gt["bbox"]) > 0.5:  # matched
                    detected += 1
                    break
    
    if gt_filtered:
        return detected / len(gt_filtered)
    return None



def build_cost_matrix(preds, gts):
    cost_matrix = np.ones((len(preds), len(gts)))
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            cost_matrix[i, j] = 1 - iou(p["bbox"], g["bbox"])
    return cost_matrix

def match_predictions(preds, gts, iou_thresh=0.5):
    if not preds or not gts:
        return 0, len(preds), len(gts)  # TP=0, FP=all preds, FN=all gts

    cost_matrix = build_cost_matrix(preds, gts)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    TP, FP, FN = 0, 0, 0
    matched_gts = set()

    for r, c in zip(row_ind, col_ind):
        iou_val = iou(preds[r]["bbox"], gts[c]["bbox"])
        if iou_val >= iou_thresh:
            TP += 1
            matched_gts.add(c)
        else:
            FP += 1
            FN += 1

    FP += len(preds) - len(row_ind)   # unmatched predictions
    FN += len(gts) - len(matched_gts) # unmatched ground truths

    return TP, FP, FN


def precision_recall(preds, gts, classes, iou_thresh=0.5):
    results = {}
    for cls in classes:
        preds_cls = [p for p in preds if p["class"] == cls]
        gts_cls   = [g for g in gts if g["class"] == cls]

        TP, FP, FN = match_predictions(preds_cls, gts_cls, iou_thresh)

        precision = TP / (TP + FP) if (TP+FP) > 0 else 0
        recall    = TP / (TP + FN) if (TP+FN) > 0 else 0

        results[str(cls)] = {"precision": precision, "recall": recall}
    return results

