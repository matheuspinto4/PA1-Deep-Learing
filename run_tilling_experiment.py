import torch
import matplotlib.pyplot as plt

from model import ModularUNet
from mosaic import build_mosaic
from tiling import run_tiled_inference_2d, stitch_naive_2d, stitch_with_fusion_2d
from metrics import evaluate_image_ap


def run_tiling_experiment(checkpoint_path=r"resultados/modelos/best_instance_head_model.pth",
                           grid_shape=(2, 2), image_size=(128, 128),
                           tile_height=96, tile_width=96, stride_h=80, stride_w=80):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ModularUNet(num_classes=3, decoder_type="skip").to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    mosaic_image, gt_instance_mask = build_mosaic(grid_shape=grid_shape, image_size=image_size)
    canvas_shape = gt_instance_mask.shape

    tiles, row_starts, col_starts = run_tiled_inference_2d(model, mosaic_image, tile_height, tile_width,
                                                            stride_h, stride_w, device)

    naive_mask = stitch_naive_2d(tiles, row_starts, col_starts, tile_height, tile_width, canvas_shape)
    fused_mask, merges = stitch_with_fusion_2d(tiles, row_starts, col_starts, tile_height, tile_width, canvas_shape)

    mAP_naive, err_naive = evaluate_image_ap(naive_mask, gt_instance_mask)
    mAP_fused, err_fused = evaluate_image_ap(fused_mask, gt_instance_mask)

    print(f"Nucleos reais no mosaico: {int(gt_instance_mask.max())}")
    print(f"Costura ingenua    -> mAP: {mAP_naive:.4f} | erro de contagem: {err_naive}")
    print(f"Com fusao de tiles -> mAP: {mAP_fused:.4f} | erro de contagem: {err_fused}")
    print(f"\nPares fundidos nas costuras: {len(merges)}")
    for seam_label, left_id, right_id, iou in merges:
        print(f"  {seam_label}: instancia {left_id} + {right_id}, IoU={iou:.2f}")

    plot_seam_comparison(gt_instance_mask, naive_mask, fused_mask, row_starts, col_starts, tile_height, tile_width)


def plot_seam_comparison(gt_mask, naive_mask, fused_mask, row_starts, col_starts, tile_height, tile_width,
                          save_path=r"resultados/imagens/tiling_seam_comparison.png"):
    """
    Mostra TODAS as costuras (verticais e horizontais) encontradas na
    grade, nao so a primeira de cada eixo -- com 3 tiles por eixo, ha
    2 costuras verticais e 2 horizontais.
    """
    window = 40
    plots = []

    for i in range(len(col_starts) - 1):
        seam_col = (col_starts[i] + tile_width + col_starts[i + 1]) // 2
        col0, col1 = max(0, seam_col - window), seam_col + window
        plots.append((f"vertical @ col={seam_col}", slice(None), slice(col0, col1), seam_col - col0, "v"))

    for i in range(len(row_starts) - 1):
        seam_row = (row_starts[i] + tile_height + row_starts[i + 1]) // 2
        row0, row1 = max(0, seam_row - window), seam_row + window
        plots.append((f"horizontal @ row={seam_row}", slice(row0, row1), slice(None), seam_row - row0, "h"))

    fig, axes = plt.subplots(len(plots), 3, figsize=(15, 5 * len(plots)))
    if len(plots) == 1:
        axes = axes.reshape(1, 3)

    for row_idx, (seam_type, row_slice, col_slice, line_pos, orientation) in enumerate(plots):
        for ax, mask, title in zip(axes[row_idx], [gt_mask, naive_mask, fused_mask],
                                    [f"Ground Truth ({seam_type})", f"Costura ingenua ({seam_type})",
                                     f"Com fusao ({seam_type})"]):
            crop = mask[row_slice, col_slice]
            ax.imshow(crop, cmap="tab20")
            if orientation == "v":
                ax.axvline(x=line_pos, color="red", linestyle="--", linewidth=1)
            else:
                ax.axhline(y=line_pos, color="red", linestyle="--", linewidth=1)
            ax.set_title(title)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nComparacao visual das costuras salva em: '{save_path}'")


if __name__ == "__main__":
    run_tiling_experiment()
