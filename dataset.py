import os
from typing import Any
import numpy as np
from PIL import Image
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

        # 1. Carregar imagem original usando PIL
        img_path = os.path.join(img_folder, img_id + '.png')
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.image_size)
        image = np.array(image, dtype=np.float32)

        mask = np.zeros(self.image_size, dtype=np.float32)
        instance_mask = np.zeros(self.image_size, dtype=np.int32)

        # 2. Carregar máscaras individuais e gerar a máscara binária e a máscara de instâncias
        for i, mask_file in enumerate(os.listdir(mask_folder), start=1):
            m_path = os.path.join(mask_folder, mask_file)
            inst_mask = Image.open(m_path).convert('L')
            inst_mask = inst_mask.resize(self.image_size, Image.NEAREST)
            inst_np = np.array(inst_mask) > 0

            mask = np.maximum(mask, inst_np.astype(np.float32))
            instance_mask[inst_np] = i

        image = image.transpose((2, 0, 1)) / 255.0

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        instance_mask = torch.tensor(instance_mask, dtype=torch.long)

        return image, mask, instance_mask