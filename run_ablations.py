import os
import csv
import itertools
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from model import ModularUNet
from dataset import DSB2018Dataset
from losses import FocalLoss, compute_class_weights
from metrics import calculate_multiclass_iou, evaluate_image_ap
from postprocessing import watershed_to_instances
from instance_labels import instance_mask_to_three_class

NUM_CLASSES = 3
RESULTS_PATH = r"resultados/ablacoes_resultados.csv"


def build_three_class_targets(instance_masks, border_size=2):
    targets = torch.stack([
        torch.from_numpy(instance_mask_to_three_class(mask.numpy(), border_size=border_size))
        for mask in instance_masks
    ])
    return targets.long()


def compute_dataset_class_weights(dataset, border_size=2):
    class_pixel_counts = [0, 0, 0]
    for _, _, instance_mask in dataset:
        three_class = instance_mask_to_three_class(instance_mask.numpy(), border_size=border_size)
        for c in range(NUM_CLASSES):
            class_pixel_counts[c] += int((three_class == c).sum())
    return compute_class_weights(class_pixel_counts, num_classes=NUM_CLASSES)


def train_and_evaluate_config(decoder_type, gamma, balanced, seed, num_epochs=3, batch_size=16,
                               image_size=(128, 128), data_dir=os.path.join("data", "stage1_train"),
                               border_size=2):
    """
    Treina UMA configuracao (mecanismo de resolucao + loss) com uma seed
    fixa e devolve as metricas finais de validacao. E a unidade basica do
    sweep de ablacoes: recebe os parametros da configuracao, devolve numeros.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = DSB2018Dataset(root_dir=data_dir, image_size=image_size, preload_in_memory=True)
    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size

    # split fixo (seed=42) em todas as configuracoes: todas sao comparadas
    # no mesmo conjunto de validacao, so a seed de treino/inicializacao muda
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    class_weights = None
    if balanced:
        class_weights = compute_dataset_class_weights(train_dataset, border_size=border_size).to(device)

    model = ModularUNet(num_classes=NUM_CLASSES, decoder_type=decoder_type).to(device)
    criterion = FocalLoss(gamma=gamma, weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, num_epochs + 1):
        model.train()
        for images, _, instance_masks in train_loader:
            images = images.to(device)
            targets = build_three_class_targets(instance_masks, border_size=border_size).to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

    model.eval()
    val_iou_sum = [0.0, 0.0, 0.0]
    image_mAPs = []
    counting_errors = []

    with torch.no_grad():
        for images, _, instance_masks in val_loader:
            images_dev = images.to(device)
            targets = build_three_class_targets(instance_masks, border_size=border_size).to(device)

            logits = model(images_dev)
            batch_iou = calculate_multiclass_iou(logits, targets, NUM_CLASSES)
            for c in range(NUM_CLASSES):
                val_iou_sum[c] += batch_iou[c] * images_dev.size(0)

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            gt_np = instance_masks.numpy()

            for i in range(probs.shape[0]):
                pred_instance_mask = watershed_to_instances(probs[i])
                mAP_img, cnt_err = evaluate_image_ap(pred_instance_mask, gt_np[i])
                image_mAPs.append(mAP_img)
                counting_errors.append(cnt_err)

    val_iou = [s / val_size for s in val_iou_sum]

    return {
        "decoder_type": decoder_type,
        "gamma": gamma,
        "balanced": balanced,
        "seed": seed,
        "iou_fundo": val_iou[0],
        "iou_interior": val_iou[1],
        "iou_fronteira": val_iou[2],
        "mAP": float(np.mean(image_mAPs)),
        "counting_error": float(np.mean(counting_errors)),
    }


def build_config_list():
    gammas = [0, 1, 2, 5]
    balanced_options = [False, True]
    seeds = [42, 123]

    configs = []
    # Eixo 2 completo: decoder fixo em "skip"
    for gamma, balanced, seed in itertools.product(gammas, balanced_options, seeds):
        configs.append({"decoder_type": "skip", "gamma": float(gamma), "balanced": balanced, "seed": seed})

    # Eixo 1: so o atrous_aspp precisa rodar de novo -- "skip" com a mesma
    # loss (gamma=0, balanceada) ja esta coberto pelo Eixo 2 acima
    for seed in seeds:
        configs.append({"decoder_type": "atrous_aspp", "gamma": 0.0, "balanced": True, "seed": seed})

    return configs


def run_all_ablations():
    configs = build_config_list()
    fieldnames = ["decoder_type", "gamma", "balanced", "seed",
                  "iou_fundo", "iou_interior", "iou_fronteira", "mAP", "counting_error"]

    os.makedirs("resultados", exist_ok=True)
    with open(RESULTS_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    for idx, config in enumerate(configs, start=1):
        print(f"\n[{idx:02d}/{len(configs)}] decoder={config['decoder_type']} "
              f"gamma={config['gamma']} balanced={config['balanced']} seed={config['seed']} ...")

        result = train_and_evaluate_config(**config)

        print(f"  -> IoU fronteira: {result['iou_fronteira']:.4f} | "
              f"mAP: {result['mAP']:.4f} | Erro contagem: {result['counting_error']:.2f}")

        with open(RESULTS_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(result)

    print(f"\nTodos os resultados salvos em '{RESULTS_PATH}'")


if __name__ == "__main__":
    run_all_ablations()
