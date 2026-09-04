import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Importar o modelo e o dataset real DSB2018
from model import ModularUNet
from dataset import DSB2018Dataset

# Otimização de threads de CPU
torch.set_num_threads(os.cpu_count() or 4)

# ==========================================
# 1. Função de Perda (Dice Loss) e Métricas
# ==========================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        
        return 1.0 - dice_score

def calculate_metrics(logits, targets, threshold=0.5, smooth=1e-6):
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
# 2. Pipeline de Treinamento Ultrarrápido
# ==========================================
def train_baseline(data_dir=os.path.join("data", "stage1_train"), num_epochs=5, batch_size=16, image_size=(128, 128)):
    print("=== Iniciando o Treinamento Otimizado da Parte 1 ===")
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo em uso: {device}")

    # A) Carregar Dataset com Pre-carregamento em RAM e Resolução 128x128
    full_dataset = DSB2018Dataset(root_dir=data_dir, image_size=image_size, preload_in_memory=True)
    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # B) Inicializar Modelo, Loss e Otimizador
    model = ModularUNet(num_classes=1).to(device)

    bce_criterion = nn.BCEWithLogitsLoss()
    dice_criterion = DiceLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_iou = 0.0
    checkpoint_path = "best_baseline_model.pth"
    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": []}

    # C) Loop de Treinamento
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # --- TREINO ---
        model.train()
        running_loss = 0.0

        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss_bce = bce_criterion(outputs, masks)
            loss_dice = dice_criterion(outputs, masks)
            total_loss = loss_bce + loss_dice

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * images.size(0)

        epoch_train_loss = running_loss / train_size

        # --- VALIDAÇÃO ---
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss_bce = bce_criterion(outputs, masks)
                loss_dice = dice_criterion(outputs, masks)
                total_loss = loss_bce + loss_dice

                val_loss += total_loss.item() * images.size(0)

                iou, dice = calculate_metrics(outputs, masks)
                val_iou += iou * images.size(0)
                val_dice += dice * images.size(0)

        epoch_val_loss = val_loss / val_size
        epoch_val_iou = val_iou / val_size
        epoch_val_dice = val_dice / val_size

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_iou"].append(epoch_val_iou)
        history["val_dice"].append(epoch_val_dice)

        epoch_time = time.time() - epoch_start
        print(f"Época [{epoch:02d}/{num_epochs:02d}] ({epoch_time:.1f}s) | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val IoU: {epoch_val_iou:.4f} | "
              f"Val Dice: {epoch_val_dice:.4f}")

        if epoch_val_iou > best_val_iou:
            best_val_iou = epoch_val_iou
            torch.save(model.state_dict(), checkpoint_path)

    elapsed_time = time.time() - start_time
    print(f"\nTreinamento concluído em {elapsed_time:.1f} segundos!")
    print(f"Melhor Val IoU atingido: {best_val_iou:.4f}")

    plot_training_curves(history)

def plot_training_curves(history, save_path="baseline_training_curves.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_title("Curva de Perda (Loss)")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["val_iou"], label="Val IoU", color="green")
    ax2.plot(epochs, history["val_dice"], label="Val Dice", color="orange")
    ax2.set_title("Métricas Semânticas de Validação")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Gráfico de treinamento salvo em: '{save_path}'")

if __name__ == "__main__":
    train_baseline()
