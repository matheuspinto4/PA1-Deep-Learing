import os
import csv
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import random_split
import matplotlib.pyplot as plt

from model import ModularUNet
from dataset import DSB2018Dataset
from metrics import evaluate_image_ap
from postprocessing import watershed_to_instances

RESULTS_PATH = r"resultados/stress_test_resultados.csv"

# Cada corrupcao tem 3 intensidades crescentes (leve/moderada/forte).
# Parametros escolhidos para produzir degradacao visivel em imagens 128x128,
# sem cobrir a imagem inteira de ruido na intensidade mais leve.
BLUR_SIGMAS = [1.0, 2.5, 5.0]
NOISE_STDS = [0.05, 0.15, 0.30]
# (fator de contraste, deslocamento de brilho) -- fator < 1 reduz contraste
# em torno do cinza medio (0.5); deslocamento negativo escurece a imagem.
BRIGHTNESS_CONTRAST_LEVELS = [(0.8, -0.05), (0.6, -0.12), (0.35, -0.20)]

LEVEL_LABELS = ["leve", "moderada", "forte"]


def apply_blur(image_np, sigma):
    # sigma=0 no eixo do canal: borra so espacialmente, sem misturar RGB
    return gaussian_filter(image_np, sigma=(0, sigma, sigma))


def apply_noise(image_np, std, rng):
    noise = rng.normal(0, std, image_np.shape).astype(np.float32)
    return np.clip(image_np + noise, 0.0, 1.0).astype(np.float32)


def apply_brightness_contrast(image_np, contrast_factor, brightness_offset):
    corrupted = (image_np - 0.5) * contrast_factor + 0.5 + brightness_offset
    return np.clip(corrupted, 0.0, 1.0).astype(np.float32)


def evaluate_with_corruption(model, val_dataset, device, corrupt_fn=None):
    """
    Roda a avaliacao de instancia (Trilha A: watershed + mAP) em todo o
    conjunto de validacao, opcionalmente aplicando corrupt_fn na imagem
    antes da inferencia. corrupt_fn=None reproduz a avaliacao limpa
    (baseline), para comparar contra as versoes corrompidas.
    """
    image_mAPs = []
    counting_errors = []

    with torch.no_grad():
        for idx in range(len(val_dataset)):
            image, _, gt_instance_mask = val_dataset[idx]
            image_np = image.numpy()

            if corrupt_fn is not None:
                image_np = corrupt_fn(image_np)

            image_tensor = torch.from_numpy(image_np).float().unsqueeze(0).to(device)
            logits = model(image_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_instance_mask = watershed_to_instances(probs)

            gt_np = gt_instance_mask.numpy()
            mAP_img, cnt_err = evaluate_image_ap(pred_instance_mask, gt_np)

            image_mAPs.append(mAP_img)
            counting_errors.append(cnt_err)

    return float(np.mean(image_mAPs)), float(np.mean(counting_errors))


def run_stress_test(checkpoint_path=r"resultados/modelos/best_instance_head_model.pth",
                     data_dir=os.path.join("data", "stage1_train"), image_size=(128, 128)):
    print("=== Parte 6 - Teste de estresse: curva de degradacao por corrupcao ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(42)

    model = ModularUNet(num_classes=3, decoder_type="skip").to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    full_dataset = DSB2018Dataset(root_dir=data_dir, image_size=image_size, preload_in_memory=True)
    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    _, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    results = []

    print("\nAvaliando baseline (sem corrupcao)...")
    mAP_base, err_base = evaluate_with_corruption(model, val_dataset, device, corrupt_fn=None)
    results.append({"corruption": "baseline", "level": 0, "label": "sem corrupcao",
                     "mAP": mAP_base, "counting_error": err_base})
    print(f"  -> mAP: {mAP_base:.4f} | erro de contagem: {err_base:.2f}")

    print("\nAvaliando corrupcao: blur (desfoque gaussiano)...")
    for level, sigma in enumerate(BLUR_SIGMAS, start=1):
        fn = lambda img, s=sigma: apply_blur(img, s)
        mAP, err = evaluate_with_corruption(model, val_dataset, device, corrupt_fn=fn)
        results.append({"corruption": "blur", "level": level, "label": f"sigma={sigma}",
                         "mAP": mAP, "counting_error": err})
        print(f"  [{LEVEL_LABELS[level-1]}] sigma={sigma} -> mAP: {mAP:.4f} | erro: {err:.2f}")

    print("\nAvaliando corrupcao: ruido gaussiano aditivo...")
    for level, std in enumerate(NOISE_STDS, start=1):
        fn = lambda img, s=std: apply_noise(img, s, rng)
        mAP, err = evaluate_with_corruption(model, val_dataset, device, corrupt_fn=fn)
        results.append({"corruption": "ruido", "level": level, "label": f"std={std}",
                         "mAP": mAP, "counting_error": err})
        print(f"  [{LEVEL_LABELS[level-1]}] std={std} -> mAP: {mAP:.4f} | erro: {err:.2f}")

    print("\nAvaliando corrupcao: brilho/contraste...")
    for level, (contrast, brightness) in enumerate(BRIGHTNESS_CONTRAST_LEVELS, start=1):
        fn = lambda img, c=contrast, b=brightness: apply_brightness_contrast(img, c, b)
        mAP, err = evaluate_with_corruption(model, val_dataset, device, corrupt_fn=fn)
        results.append({"corruption": "brilho_contraste", "level": level,
                         "label": f"contraste={contrast},brilho={brightness}",
                         "mAP": mAP, "counting_error": err})
        print(f"  [{LEVEL_LABELS[level-1]}] contraste={contrast} brilho={brightness} "
              f"-> mAP: {mAP:.4f} | erro: {err:.2f}")

    save_results(results)
    plot_degradation_curves(results, baseline_mAP=mAP_base)

    return results


def save_results(results):
    os.makedirs("resultados", exist_ok=True)
    fieldnames = ["corruption", "level", "label", "mAP", "counting_error"]
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResultados salvos em '{RESULTS_PATH}'")


def plot_degradation_curves(results, baseline_mAP, save_path=r"resultados/imagens/stress_test_corruptions.png"):
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"blur": "crimson", "ruido": "navy", "brilho_contraste": "darkorange"}
    names = {"blur": "Blur (desfoque)", "ruido": "Ruido gaussiano", "brilho_contraste": "Brilho/contraste"}

    for corruption_type in ["blur", "ruido", "brilho_contraste"]:
        subset = [r for r in results if r["corruption"] == corruption_type]
        levels = [0] + [r["level"] for r in subset]
        maps = [baseline_mAP] + [r["mAP"] for r in subset]
        ax.plot(levels, maps, marker="o", label=names[corruption_type], color=colors[corruption_type])

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["sem\ncorrupcao", "leve", "moderada", "forte"])
    ax.set_xlabel("Intensidade da corrupcao")
    ax.set_ylabel("mAP @ [0.50:0.95]")
    ax.set_title("Curva de degradacao do mAP sob corrupcoes (Trilha A)")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Grafico de degradacao salvo em: '{save_path}'")


if __name__ == "__main__":
    run_stress_test()
