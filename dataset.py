import os
from typing import Any
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class DSB2018Dataset(Dataset):
    def __init__(self, root_dir, image_size=(256,256)) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.image_size = image_size

        self.image_ids = next(os.walk(root_dir))[1]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx) -> Any:
        img_id = self.image_ids[idx]
        img_folder = os.path.join(self.root_dir, img_id, 'images')
        mask_folder = os.path.join(self.root_dir, img_id, 'masks')

        img_path = os.path.join(img_folder, img_id + '.png')
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)

        mask = np.zeros(self.image_size, dtype=np.float32)

        for mask_file in os.listdir(mask_folder):
            m_path = os.path.join(mask_folder, mask_file)
            instance_mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            instance_mask = cv2.resize(instance_mask, self.image_size, interpolation=cv2.INTER_NEAREST)

            mask = np.maximum(mask, instance_mask)

        mask = (mask > 0).astype(np.float32)
        image = image.transpose((2, 0, 1)) / 255.0

        image = torch.tensor(image, dtype=torch.float32)

        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask
                