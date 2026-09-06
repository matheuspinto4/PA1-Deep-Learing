import torch
import torch.nn as nn


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


def compute_class_weights(class_pixel_counts, num_classes):
    """
    Calcula pesos por classe a partir da contagem total de pixels de cada
    classe no conjunto de treino, para compensar o desbalanceamento (a
    classe fronteira e sempre a minoritaria, por ser so um anel fino ao
    redor de cada nucleo).

    Formula de frequencia inversa normalizada:
        peso_c = total_pixels / (num_classes * contagem_c)
    Quanto mais rara a classe, maior o peso.
    """
    total_pixels = sum(class_pixel_counts)
    weights = [total_pixels / (num_classes * count) for count in class_pixel_counts]
    return torch.tensor(weights, dtype=torch.float32)
