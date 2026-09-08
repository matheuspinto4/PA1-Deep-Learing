import os
import torch
from torch.utils.data import random_split

from model import ModularUNet
from dataset import DSB2018Dataset
from metrics import evaluate_image_ap
from postprocessing import watershed_to_instances

FAILURE_VAL_INDICES = [19, 42, 47, 49, 123]


def evaluate_specific_images(checkpoint_path, image_size, val_indices=FAILURE_VAL_INDICES,
                              data_dir=os.path.join("data", "stage1_train")):
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

    results = {}
    with torch.no_grad():
        for val_idx in val_indices:
            image, _, gt_instance_mask = val_dataset[val_idx]
            logits = model(image.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_instance_mask = watershed_to_instances(probs)

            gt_np = gt_instance_mask.numpy()
            mAP_img, cnt_err = evaluate_image_ap(pred_instance_mask, gt_np)

            results[val_idx] = {
                "mAP": mAP_img, "counting_error": cnt_err,
                "num_gts": int(gt_np.max()), "num_preds": int(pred_instance_mask.max()),
            }

    return results


def compare_resolutions():
    print("Avaliando checkpoint treinado em 128x128...")
    results_128 = evaluate_specific_images(
        checkpoint_path=r"resultados/modelos/best_instance_head_model.pth", image_size=(128, 128))

    print("Avaliando checkpoint treinado em 256x256...")
    results_256 = evaluate_specific_images(
        checkpoint_path=r"resultados/modelos/best_instance_head_model_256.pth", image_size=(256, 256))

    print(f"\n{'val_idx':>8} | {'nucleos':>8} | {'mAP 128px':>10} | {'mAP 256px':>10} | "
          f"{'erro 128px':>10} | {'erro 256px':>10}")
    for val_idx in FAILURE_VAL_INDICES:
        r128, r256 = results_128[val_idx], results_256[val_idx]
        print(f"{val_idx:>8} | {r128['num_gts']:>8} | {r128['mAP']:>10.4f} | {r256['mAP']:>10.4f} | "
              f"{r128['counting_error']:>10} | {r256['counting_error']:>10}")


if __name__ == "__main__":
    compare_resolutions()
