import torch
import torch.nn as nn
import torch.nn.functional as F


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



class FocalLoss(nn.Module):
    """
    Focal Loss generalizada (slides 77-79). Com gamma=0, reduz-se
    exatamente a Cross-Entropy (com ou sem peso por classe) -- e o teste
    de sanidade no final deste arquivo confirma isso numericamente.
    """
    def __init__(self, gamma=0.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        targets_unsqueezed = targets.unsqueeze(1)
        log_pt = log_probs.gather(1, targets_unsqueezed).squeeze(1)
        pt = probs.gather(1, targets_unsqueezed).squeeze(1)

        focal_term = (1 - pt) ** self.gamma
        loss = -focal_term * log_pt

        if self.weight is not None:
            weight_map = self.weight.to(logits.device)[targets]
            loss = loss * weight_map
            return loss.sum() / weight_map.sum()

        return loss.mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    logits = torch.randn(4, 3, 8, 8)
    targets = torch.randint(0, 3, (4, 8, 8))
    weights = torch.tensor([0.4, 5.3, 4.5])

    ce_loss = nn.CrossEntropyLoss(weight=weights)(logits, targets)
    focal_loss_gamma0 = FocalLoss(gamma=0.0, weight=weights)(logits, targets)

    print(f"CrossEntropyLoss (balanceada):            {ce_loss.item():.6f}")
    print(f"FocalLoss com gamma=0 (deveria ser igual): {focal_loss_gamma0.item():.6f}")
    assert torch.allclose(ce_loss, focal_loss_gamma0, atol=1e-5), \
        "FocalLoss com gamma=0 deveria reproduzir a CE balanceada!"
    print("OK: FocalLoss(gamma=0) reproduz exatamente a CE balanceada.")