import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from model import ModularUNet
from dataset import DSB2018Dataset
from metrics import evaluate_image_ap
from postprocessing import connected_components_to_instances, watershed_to_instances


def evaluate_baseline_model(checkpoint_path=r"resultados/modelos/best_baseline_model.pth", data_dir=os.path.join("data", "stage1_train")):
    print("=== Avaliando o Modelo Baseline no Nivel de Instancia (Parte 1) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ModularUNet(num_classes=1).to(device)
    if not os.path.exists(checkpoint_path):
        print(f"Erro: Checkpoint '{checkpoint_path}' nao encontrado. Treine o modelo primeiro com train.py.")
        return None

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
            pred_instance_mask = connected_components_to_instances(binary_pred)

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

    plot_failure_analysis(gt_object_counts, image_mAPs, counting_errors,
                           save_path=r"resultados/imagens/baseline_failure_analysis.png")

    return mean_mAP, mean_cnt_err


def evaluate_instance_head_model(checkpoint_path=r"resultados/modelos/best_instance_head_model.pth",
                                  data_dir=os.path.join("data", "stage1_train")):
    print("=== Avaliando a Trilha A (fronteiras + watershed) no Nivel de Instancia (Parte 2) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ModularUNet(num_classes=3).to(device)
    if not os.path.exists(checkpoint_path):
        print(f"Erro: Checkpoint '{checkpoint_path}' nao encontrado. Treine o modelo primeiro com train_instance_head.py.")
        return None

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
            logits = model(image)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            pred_instance_mask = watershed_to_instances(probs)

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
    print(f"RESULTADOS DA TRILHA A (FRONTEIRAS + WATERSHED):")
    print(f"  -> mAP @ [0.50:0.95] Medio: {mean_mAP:.4f}")
    print(f"  -> Erro Absoluto Medio de Contagem por Imagem: {mean_cnt_err:.2f} objetos")
    print("="*50)

    plot_failure_analysis(gt_object_counts, image_mAPs, counting_errors,
                           save_path=r"resultados/imagens/instance_head_failure_analysis.png")

    return mean_mAP, mean_cnt_err


def plot_failure_analysis(gt_counts, mAPs, counting_errors, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.scatter(gt_counts, mAPs, alpha=0.6, color="crimson", edgecolors="k")

    if len(gt_counts) > 1:
        z = np.polyfit(gt_counts, mAPs, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(gt_counts), max(gt_counts), 100)
        ax1.plot(x_trend, p(x_trend), "r--", linewidth=2, label="Tendencia")

    ax1.set_title("mAP vs Densidade de Objetos", fontsize=12)
    ax1.set_xlabel("Densidade de Objetos (Numero de Nucleos na Imagem)")
    ax1.set_ylabel("mAP @ [0.50:0.95]")
    ax1.legend()
    ax1.grid(True)

    ax2.scatter(gt_counts, counting_errors, alpha=0.6, color="navy", edgecolors="k")

    if len(gt_counts) > 1:
        z2 = np.polyfit(gt_counts, counting_errors, 1)
        p2 = np.poly1d(z2)
        ax2.plot(x_trend, p2(x_trend), "b--", linewidth=2, label="Tendencia de Erro")

    ax2.set_title("Erro Absoluto de Contagem vs Densidade de Objetos", fontsize=12)
    ax2.set_xlabel("Densidade de Objetos (Numero de Nucleos na Imagem)")
    ax2.set_ylabel("Erro Absoluto de Contagem (|Pred - GT|)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nGrafico de analise de fracasso salvo em: '{save_path}'")


if __name__ == "__main__":
    baseline_results = evaluate_baseline_model()
    instance_head_results = evaluate_instance_head_model()

    if baseline_results is not None and instance_head_results is not None:
        print("\n" + "="*50)
        print("COMPARACAO BASELINE (Parte 1) vs TRILHA A (Parte 2)")
        print(f"{'Metodo':<25} {'mAP':>10} {'Erro Contagem':>15}")
        print(f"{'Componentes Conexos':<25} {baseline_results[0]:>10.4f} {baseline_results[1]:>15.2f}")
        print(f"{'Fronteiras + Watershed':<25} {instance_head_results[0]:>10.4f} {instance_head_results[1]:>15.2f}")
        print("="*50)
