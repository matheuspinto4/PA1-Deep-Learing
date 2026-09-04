import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Importar o modelo e o dataset sintético
from model import ModularUNet
from synthetic_dataset import SyntheticEllipseDataset

# ==========================================
# 1. Funções de Métrica e Perda (Loss & Metrics)
# ==========================================
class DiceLoss(nn.Module):
    """
    Dice Loss para Segmentação Semântica.
    Mede a sobreposição entre a previsão da rede e a máscara verdadeira.
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Aplica sigmoide para converter logits em probabilidades [0, 1]
        probs = torch.sigmoid(logits)
        
        # Aplanar os tensores (flatten) para vetor 1D
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        
        return 1.0 - dice_score

def calculate_metrics(logits, targets, threshold=0.5, smooth=1e-6):
    """
    Calcula o IoU (Intersection over Union) e o Dice Coefficient.
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
# 2. Pipeline Principal de Treinamento
# ==========================================
def train_synthetic():
    print("=== Iniciando o Teste Unitário Sintético (Parte 0) ===")
    start_time = time.time()

    # Configuração de Dispositivo (GPU se disponível, senão CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo em uso: {device}")

    # A) Instanciar Datasets e DataLoaders Sintéticos
    train_dataset = SyntheticEllipseDataset(num_samples=200, seed=42)
    val_dataset = SyntheticEllipseDataset(num_samples=40, seed=100)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # B) Inicializar Modelo, Perdas e Otimizador
    model = ModularUNet(num_classes=1).to(device)

    bce_criterion = nn.BCEWithLogitsLoss()
    dice_criterion = DiceLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    num_epochs = 5

    # C) Loop de Treinamento e Validação por Época
    for epoch in range(1, num_epochs + 1):
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
            
            # Perda combinada BCE + Dice
            total_loss = loss_bce + loss_dice

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * images.size(0)

        epoch_train_loss = running_loss / len(train_dataset)

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

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_iou = val_iou / len(val_dataset)
        epoch_val_dice = val_dice / len(val_dataset)

        print(f"Época [{epoch}/{num_epochs}] | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val IoU: {epoch_val_iou:.4f} | "
              f"Val Dice: {epoch_val_dice:.4f}")

    elapsed_time = time.time() - start_time
    print(f"\nTreinamento concluído em {elapsed_time:.2f} segundos!")

    # D) Salvar um plot visual com previsões do modelo
    plot_results(model, val_loader, device)

# ==========================================
# 3. Função para Gerar o Gráfico de Resultados
# ==========================================
def plot_results(model, val_loader, device, save_path="synthetic_results.png"):
    model.eval()
    images, masks, _ = next(iter(val_loader))
    images_dev = images.to(device)

    with torch.no_grad():
        outputs = model(images_dev)
        preds = (torch.sigmoid(outputs) > 0.5).float()

    # Selecionar 3 amostras para visualizar
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for i in range(3):
        # Imagem de Entrada (converter de C,H,W para H,W,C)
        img_np = images[i].permute(1, 2, 0).numpy()
        mask_gt = masks[i, 0].numpy()
        pred_mask = preds[i, 0].cpu().numpy()

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"Amostra {i+1}: Entrada")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask_gt, cmap="gray")
        axes[i, 1].set_title(f"Amostra {i+1}: Gabarito (GT)")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_mask, cmap="gray")
        axes[i, 2].set_title(f"Amostra {i+1}: Previsão da Rede")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Imagem com resultados visuais salva em: '{save_path}'")

if __name__ == "__main__":
    train_synthetic()
