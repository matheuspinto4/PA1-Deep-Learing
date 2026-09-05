import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from scipy.ndimage import label
import matplotlib.pyplot as plt

from model import ModularUNet
from dataset import DSB2018Dataset

# ==========================================
# 1. Algoritmo de Matching Guloso (mAP @ 0.50:0.95)
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


# ==========================================
# 2. Avaliacao Completa do Modelo Baseline
# ==========================================
def evaluate_baseline_model(checkpoint_path="best_baseline_model.pth", data_dir=os.path.join("data", "stage1_train")):
    print("=== Avaliando o Modelo Baseline no Nivel de Instancia (Parte 1) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ModularUNet(num_classes=1).to(device)
    if not os.path.exists(checkpoint_path):
        print(f"Erro: Checkpoint '{checkpoint_path}' nao encontrado. Treine o modelo primeiro com train.py.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    full_dataset = DSB2018Dataset(root_dir=data_dir, image_size=(128, 128), preload_in_memory=True)
    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    _, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    image_mAPs = []
    counting_errors = []
    gt_object_counts = []

    with torch.no_grad():
        for idx, (image, _, gt_instance_mask) in enumerate(val_loader):
            image = image.to(device)
            output = model(image)
            prob = torch.sigmoid(output).squeeze().cpu().numpy()

            binary_pred = (prob > 0.5).astype(np.uint8)
            pred_instance_mask, num_features = label(binary_pred)

            gt_mask_np = gt_instance_mask.squeeze().numpy()
            num_gts = int(gt_mask_np.max())

            mAP_img, cnt_err = evaluate_image_ap(pred_instance_mask, gt_mask_np)

            image_mAPs.append(mAP_img)
            counting_errors.append(cnt_err)
            gt_object_counts.append(num_gts)

            if (idx + 1) % 30 == 0 or (idx + 1) == len(val_dataset):
                print(f"Amostra [{idx+1:03d}/{len(val_dataset)}] | "
                      f"mAP: {mAP_img:.4f} | Erro de Contagem: {cnt_err} | Nucleos Reais: {num_gts}")

    mean_mAP = np.mean(image_mAPs)
    mean_cnt_err = np.mean(counting_errors)

    print("\n" + "="*50)
    print(f"RESULTADOS DO BASELINE (METODO INGENUO):")
    print(f"  -> mAP @ [0.50:0.95] Medio: {mean_mAP:.4f}")
    print(f"  -> Erro Absoluto Medio de Contagem por Imagem: {mean_cnt_err:.2f} objetos")
    print("="*50)

    plot_failure_analysis(gt_object_counts, image_mAPs, counting_errors)

def plot_failure_analysis(gt_counts, mAPs, counting_errors, save_path="baseline_failure_analysis.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.scatter(gt_counts, mAPs, alpha=0.6, color="crimson", edgecolors="k")
    
    if len(gt_counts) > 1:
        z = np.polyfit(gt_counts, mAPs, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(gt_counts), max(gt_counts), 100)
        ax1.plot(x_trend, p(x_trend), "r--", linewidth=2, label="Tendencia")

    ax1.set_title("Fracasso da Abordagem Ingenua: mAP vs Densidade de Objetos", fontsize=12)
    ax1.set_xlabel("Densidade de Objetos (Número de Núcleos na Imagem)")
    ax1.set_ylabel("mAP @ [0.50:0.95]")
    ax1.legend()
    ax1.grid(True)

    ax2.scatter(gt_counts, counting_errors, alpha=0.6, color="navy", edgecolors="k")
    
    if len(gt_counts) > 1:
        z2 = np.polyfit(gt_counts, counting_errors, 1)
        p2 = np.poly1d(z2)
        ax2.plot(x_trend, p2(x_trend), "b--", linewidth=2, label="Tendencia de Erro")

    ax2.set_title("Erro Absoluto de Contagem vs Densidade de Objetos", fontsize=12)
    ax2.set_xlabel("Densidade de Objetos (Número de Núcleos na Imagem)")
    ax2.set_ylabel("Erro Absoluto de Contagem (|Pred - GT|)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nGrafico de analise de fracasso salvo em: '{save_path}'")

if __name__ == "__main__":
    evaluate_baseline_model()
