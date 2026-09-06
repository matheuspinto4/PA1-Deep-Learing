import numpy as np
import torch
from torch.utils.data import random_split

from dataset import DSB2018Dataset


def build_mosaic(grid_shape=(2, 2), image_size=(128, 128), data_dir="data/stage1_train", seed=42):
    """
    Monta uma imagem grande concatenando imagens de validacao numa grade
    grid_shape = (linhas, colunas), com uma mascara de instancia combinada
    (ids renumerados para serem unicos entre as sub-imagens).
    """
    rows, cols = grid_shape
    num_images = rows * cols

    full_dataset = DSB2018Dataset(root_dir=data_dir, image_size=image_size, preload_in_memory=True)
    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    _, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    h, w = image_size
    mosaic_image = torch.zeros(3, h * rows, w * cols)
    mosaic_instance_mask = np.zeros((h * rows, w * cols), dtype=np.int64)

    next_id = 1
    for idx in range(num_images):
        row_idx, col_idx = divmod(idx, cols)
        image, _, instance_mask = val_dataset[idx]
        instance_mask_np = instance_mask.numpy()

        row0, col0 = row_idx * h, col_idx * w
        mosaic_image[:, row0:row0 + h, col0:col0 + w] = image

        remapped = np.zeros_like(instance_mask_np)
        for local_id in np.unique(instance_mask_np):
            if local_id == 0:
                continue
            remapped[instance_mask_np == local_id] = next_id
            next_id += 1
        mosaic_instance_mask[row0:row0 + h, col0:col0 + w] = remapped

    return mosaic_image, mosaic_instance_mask
