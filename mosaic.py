import numpy as np
import torch
from torch.utils.data import random_split

from dataset import DSB2018Dataset


def build_mosaic(num_images=4, image_size=(128, 128), data_dir="data/stage1_train", seed=42):
    """
    Monta uma imagem grande concatenando "num_images" imagens de validacao
    lado a lado (mosaico 1 x N), com uma mascara de instancia combinada
    (ids renumerados para serem unicos entre as sub-imagens).
    """
    full_dataset = DSB2018Dataset(root_dir=data_dir, image_size=image_size, preload_in_memory=True)
    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    _, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    h, w = image_size
    mosaic_image = torch.zeros(3, h, w * num_images)
    mosaic_instance_mask = np.zeros((h, w * num_images), dtype=np.int64)

    next_id = 1
    for i in range(num_images):
        image, _, instance_mask = val_dataset[i]
        instance_mask_np = instance_mask.numpy()

        mosaic_image[:, :, i * w:(i + 1) * w] = image

        remapped = np.zeros_like(instance_mask_np)
        for local_id in np.unique(instance_mask_np):
            if local_id == 0:
                continue
            remapped[instance_mask_np == local_id] = next_id
            next_id += 1
        mosaic_instance_mask[:, i * w:(i + 1) * w] = remapped

    return mosaic_image, mosaic_instance_mask
