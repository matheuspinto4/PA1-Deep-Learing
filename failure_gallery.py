import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from model import ModularUNet
from dataset import DSB2018Dataset
from metrics import evaluate_image_ap
from postprocessing import watershed_to_instances


def build_failure_gallery(checkpoint_path=r"resultados/modelos/best_instance_head_model.pth",
                           data_dir=os.path.join("data", "stage1_train"), image_size=(128, 128),
                           num_worst=5, save_dir="resultados/imagens"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ModularUNet(num_classes=3, decoder_type="skip").to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    full_dataset = DSB2018Dataset(root_dir=data_dir, image_size=image_size, preload_in_memory=True)
    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    _, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    records = []
    with torch.no_grad():
        for idx in range(len(val_dataset)):
            image, _, gt_instance_mask = val_dataset[idx]
            logits = model(image.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_instance_mask = watershed_to_instances(probs)

            gt_np = gt_instance_mask.numpy()
            mAP_img, cnt_err = evaluate_image_ap(pred_instance_mask, gt_np)

            sizes = []
            for instance_id in np.unique(gt_np):
                if instance_id == 0:
                    continue
                ys, xs = np.where(gt_np == instance_id)
                sizes.append(max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1))

            records.append({
                "idx": idx, "image": image, "gt_mask": gt_np, "pred_mask": pred_instance_mask,
                "probs": probs, "mAP": mAP_img, "counting_error": cnt_err,
                "num_gts": int(gt_np.max()), "num_preds": int(pred_instance_mask.max()),
                "min_size": min(sizes) if sizes else 0, "max_size": max(sizes) if sizes else 0,
            })

    records.sort(key=lambda r: r["mAP"])
    worst = records[:num_worst]

    print(f"=== {num_worst} piores imagens (menor mAP) ===")
    for rank, r in enumerate(worst, start=1):
        print(f"#{rank} idx={r['idx']:03d} | mAP={r['mAP']:.4f} | erro contagem={r['counting_error']} | "
              f"nucleos reais={r['num_gts']} | nucleos previstos={r['num_preds']} | "
              f"tamanho nucleos (min-max)={r['min_size']}-{r['max_size']}px")

        plot_failure_case(r, rank, save_dir)

    return worst


def plot_failure_case(record, rank, save_dir):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    img_np = record["image"].permute(1, 2, 0).numpy()
    axes[0].imshow(img_np)
    axes[0].set_title("Imagem de entrada")
    axes[0].axis("off")

    axes[1].imshow(record["gt_mask"], cmap="tab20")
    axes[1].set_title(f"Ground Truth ({record['num_gts']} nucleos)")
    axes[1].axis("off")

    axes[2].imshow(record["pred_mask"], cmap="tab20")
    axes[2].set_title(f"Predicao ({record['num_preds']} nucleos) - mAP={record['mAP']:.3f}")
    axes[2].axis("off")

    fronteira_prob = record["probs"][2]
    im = axes[3].imshow(fronteira_prob, cmap="hot", vmin=0, vmax=1)
    axes[3].set_title("Prob. de fronteira (mapa intermediario)")
    axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"failure_case_{rank:02d}_idx{record['idx']:03d}.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  -> figura salva em '{save_path}'")


if __name__ == "__main__":
    build_failure_gallery()
