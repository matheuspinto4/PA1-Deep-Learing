import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from model import ModularUNet
from dataset import DSB2018Dataset
from losses import compute_class_weights
from metrics import calculate_multiclass_iou
from instance_labels import instance_mask_to_three_class

torch.set_num_threads(os.cpu_count() or 4)

NUM_CLASSES = 3  # 0 = fundo, 1 = interior, 2 = fronteira


def build_three_class_targets(instance_masks, border_size=2):
    """
    Converte um lote (batch) de instance_masks (B, H, W) em rotulos de
    3 classes (B, H, W). A erosao e feita amostra a amostra em numpy
    (instance_labels.py), entao aqui so fazemos o loop do batch e
    empilhamos de volta em um tensor.
    """
    targets = torch.stack([
        torch.from_numpy(instance_mask_to_three_class(mask.numpy(), border_size=border_size))
        for mask in instance_masks
    ])
    return targets.long()


def compute_dataset_class_weights(dataset, border_size=2):
    print("Calculando pesos de classe (fundo/interior/fronteira) no conjunto de treino...")
    class_pixel_counts = [0, 0, 0]

    for _, _, instance_mask in dataset:
        three_class = instance_mask_to_three_class(instance_mask.numpy(), border_size=border_size)
        for c in range(NUM_CLASSES):
            class_pixel_counts[c] += int((three_class == c).sum())

    print(f"Contagem de pixels -> fundo: {class_pixel_counts[0]} | "
          f"interior: {class_pixel_counts[1]} | fronteira: {class_pixel_counts[2]}")

    return compute_class_weights(class_pixel_counts, num_classes=NUM_CLASSES)


def train_instance_head(data_dir=os.path.join("data", "stage1_train"), num_epochs=5, batch_size=16,
                         image_size=(128, 128), border_size=2):
    print("=== Iniciando o Treinamento da Trilha A (fronteiras + watershed) - Parte 2 ===")
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo em uso: {device}")

    full_dataset = DSB2018Dataset(root_dir=data_dir, image_size=image_size, preload_in_memory=True)
    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    class_weights = compute_dataset_class_weights(train_dataset, border_size=border_size).to(device)
    print(f"Pesos de classe (fundo, interior, fronteira): {class_weights.tolist()}")

    model = ModularUNet(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_object_iou = 0.0
    checkpoint_path = r"resultados/modelos/best_instance_head_model.pth"
    history = {"train_loss": [], "val_loss": [], "val_iou_fundo": [], "val_iou_interior": [], "val_iou_fronteira": []}

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # --- TREINO ---
        model.train()
        running_loss = 0.0

        for images, _, instance_masks in train_loader:
            images = images.to(device)
            targets = build_three_class_targets(instance_masks, border_size=border_size).to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / train_size

        # --- VALIDACAO ---
        model.eval()
        val_loss = 0.0
        val_iou_sum = [0.0, 0.0, 0.0]

        with torch.no_grad():
            for images, _, instance_masks in val_loader:
                images = images.to(device)
                targets = build_three_class_targets(instance_masks, border_size=border_size).to(device)

                logits = model(images)
                loss = criterion(logits, targets)
                val_loss += loss.item() * images.size(0)

                batch_iou = calculate_multiclass_iou(logits, targets, NUM_CLASSES)
                for c in range(NUM_CLASSES):
                    val_iou_sum[c] += batch_iou[c] * images.size(0)

        epoch_val_loss = val_loss / val_size
        epoch_val_iou = [s / val_size for s in val_iou_sum]
        epoch_val_object_iou = (epoch_val_iou[1] + epoch_val_iou[2]) / 2.0

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_iou_fundo"].append(epoch_val_iou[0])
        history["val_iou_interior"].append(epoch_val_iou[1])
        history["val_iou_fronteira"].append(epoch_val_iou[2])

        epoch_time = time.time() - epoch_start
        print(f"Epoca [{epoch:02d}/{num_epochs:02d}] ({epoch_time:.1f}s) | "
              f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
              f"IoU fundo: {epoch_val_iou[0]:.4f} | IoU interior: {epoch_val_iou[1]:.4f} | "
              f"IoU fronteira: {epoch_val_iou[2]:.4f}")

        if epoch_val_object_iou > best_val_object_iou:
            best_val_object_iou = epoch_val_object_iou
            torch.save(model.state_dict(), checkpoint_path)

    elapsed_time = time.time() - start_time
    print(f"\nTreinamento concluido em {elapsed_time:.1f} segundos!")
    print(f"Melhor media de IoU (interior+fronteira) atingida: {best_val_object_iou:.4f}")

    plot_training_curves(history)


def plot_training_curves(history, save_path=r"resultados/imagens/instance_head_training_curves.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_title("Curva de Perda (CE Balanceada)")
    ax1.set_xlabel("Epoca")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["val_iou_fundo"], label="IoU Fundo")
    ax2.plot(epochs, history["val_iou_interior"], label="IoU Interior")
    ax2.plot(epochs, history["val_iou_fronteira"], label="IoU Fronteira")
    ax2.set_title("IoU por Classe (Validacao)")
    ax2.set_xlabel("Epoca")
    ax2.set_ylabel("IoU")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Grafico de treinamento salvo em: '{save_path}'")


if __name__ == "__main__":
    train_instance_head()
