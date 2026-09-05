import numpy as np
import torch


# ==========================================
# Metricas semanticas (pixel a pixel) - usadas nas Partes 0 e 1
# ==========================================
def calculate_metrics(logits, targets, threshold=0.5, smooth=1e-6):
    """
    Calcula o IoU (Intersection over Union) e o Dice Coefficient
    para segmentacao binaria (fundo vs objeto).
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum()
    total_union = preds_flat.sum() + targets_flat.sum() - intersection

    iou = (intersection + smooth) / (total_union + smooth)
    dice = (2.0 * intersection + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)

    return iou.item(), dice.item()


# ==========================================
# Metrica de instancia (mAP + matching guloso) - usada nas Partes 1, 2, 4, 6
#
# Regra de matching (documentada aqui, uma vez so, para toda comparacao entre
# partes ser valida): guloso por IoU decrescente. Para cada limiar de IoU,
# ordenamos todos os pares (predicao, GT) com IoU >= limiar em ordem
# decrescente e casamos greedily, sem reuso de predicao nem de GT.
# ==========================================
def compute_iou_matrix(pred_map, gt_map, num_preds, num_gts):
    if num_preds == 0 or num_gts == 0:
        return np.zeros((num_preds, num_gts), dtype=np.float32)

    iou_matrix = np.zeros((num_preds, num_gts), dtype=np.float32)

    for i in range(1, num_preds + 1):
        pred_mask = (pred_map == i)
        pred_area = pred_mask.sum()
        if pred_area == 0:
            continue

        overlapping_gt_ids = np.unique(gt_map[pred_mask])
        for j in overlapping_gt_ids:
            if j == 0:
                continue
            gt_mask = (gt_map == j)
            gt_area = gt_mask.sum()

            intersection = (pred_mask & gt_mask).sum()
            union = pred_area + gt_area - intersection

            if union > 0:
                iou_matrix[i - 1, j - 1] = intersection / union

    return iou_matrix


def evaluate_image_ap(pred_map, gt_map, iou_thresholds=np.arange(0.50, 1.00, 0.05)):
    num_preds = int(pred_map.max())
    num_gts = int(gt_map.max())

    counting_error = abs(num_preds - num_gts)

    if num_gts == 0 and num_preds == 0:
        return 1.0, counting_error
    if num_gts == 0 or num_preds == 0:
        return 0.0, counting_error

    iou_matrix = compute_iou_matrix(pred_map, gt_map, num_preds, num_gts)

    ap_per_threshold = []

    for t in iou_thresholds:
        pairs = []
        for i in range(num_preds):
            for j in range(num_gts):
                if iou_matrix[i, j] >= t:
                    pairs.append((iou_matrix[i, j], i, j))

        pairs.sort(key=lambda x: x[0], reverse=True)

        matched_preds = set()
        matched_gts = set()

        for iou, i, j in pairs:
            if i not in matched_preds and j not in matched_gts:
                matched_preds.add(i)
                matched_gts.add(j)

        tp = len(matched_preds)
        fp = num_preds - tp
        fn = num_gts - tp

        denominator = tp + fp + fn
        ap_t = tp / denominator if denominator > 0 else 0.0
        ap_per_threshold.append(ap_t)

    mAP = float(np.mean(ap_per_threshold))
    return mAP, counting_error
