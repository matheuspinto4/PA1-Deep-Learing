import os
from typing import Any
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset


class DSB2018Dataset(Dataset):
    """
    Dataset real DSB2018 com cache em memoria para velocidade maxima de treinamento.
    """
    def __init__(self, root_dir, image_size=(128, 128), preload_in_memory=True) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.image_size = image_size
        self.preload_in_memory = preload_in_memory

        self.image_ids = next(os.walk(root_dir))[1]
        
        # Cache em RAM para evitar I/O de disco a cada epoca
        self.cache = []
        if self.preload_in_memory:
            print(f"Pre-carregando {len(self.image_ids)} imagens na memoria RAM...")
            for idx in range(len(self.image_ids)):
                self.cache.append(self._load_item(idx))
            print("Pre-carregamento concluido!")

    def __len__(self):
        return len(self.image_ids)

    def _load_item(self, idx):
        img_id = self.image_ids[idx]
        img_folder = os.path.join(self.root_dir, img_id, 'images')
        mask_folder = os.path.join(self.root_dir, img_id, 'masks')

        # 1. Carregar imagem original usando PIL
        img_files = os.listdir(img_folder)
        img_path = os.path.join(img_folder, img_files[0])
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.image_size)
        image = np.array(image, dtype=np.float32)

        mask = np.zeros(self.image_size, dtype=np.float32)
        instance_mask = np.zeros(self.image_size, dtype=np.int32)

        # 2. Carregar mascaras individuais
        if os.path.exists(mask_folder):
            for i, mask_file in enumerate(os.listdir(mask_folder), start=1):
                m_path = os.path.join(mask_folder, mask_file)
                inst_mask = Image.open(m_path).convert('L')
                inst_mask = inst_mask.resize(self.image_size, Image.NEAREST)
                inst_np = np.array(inst_mask) > 0

                mask = np.maximum(mask, inst_np.astype(np.float32))
                instance_mask[inst_np] = i

        image = image.transpose((2, 0, 1)) / 255.0

        image_tensor = torch.tensor(image, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        instance_mask_tensor = torch.tensor(instance_mask, dtype=torch.long)

        return image_tensor, mask_tensor, instance_mask_tensor

    def __getitem__(self, idx) -> Any:
        if self.preload_in_memory:
            return self.cache[idx]
        else:
            return self.load_item(idx)


class SyntheticEllipseDataset(Dataset):
    """
    Dataset sintético gerador de elipses de tamanhos variados para o teste unitário (Parte 0).
    Gera imagens 128x128 com 5 a 20 elipses, ruído e contraste variáveis.
    """
    def __init__(self, num_samples=200, image_size=(128, 128), min_ellipses=5, max_ellipses=20, seed=42):
        super().__init__()
        self.num_samples = num_samples
        self.image_size = image_size
        self.min_ellipses = min_ellipses
        self.max_ellipses = max_ellipses
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Seed individual por índice para reprodutibilidade
        if self.seed is not None:
            np.random.seed(self.seed + idx)

        w, h = self.image_size
        
        # 1. Cor de fundo aleatória (tom de cinza escuro entre 10 e 50)
        bg_color = int(np.random.randint(10, 50))
        img_pil = Image.new("L", (w, h), bg_color)
        instance_pil = Image.new("I", (w, h), 0)

        draw_img = ImageDraw.Draw(img_pil)
        draw_instance = ImageDraw.Draw(instance_pil)

        # 2. Número aleatório de elipses entre min_ellipses e max_ellipses
        num_ellipses = np.random.randint(self.min_ellipses, self.max_ellipses + 1)

        for i in range(1, num_ellipses + 1):
            # Centro da elipse
            cx = np.random.randint(15, w - 15)
            cy = np.random.randint(15, h - 15)
            
            # Raio horizontal e vertical (podem se tocar/sobrepor)
            rx = np.random.randint(6, 18)
            ry = np.random.randint(6, 18)
            
            # Brilho do núcleo (cinza claro entre 140 e 240)
            fg_color = int(np.random.randint(140, 240))

            # Bounding box [x0, y0, x1, y1]
            bbox = [cx - rx, cy - ry, cx + rx, cy + ry]

            # Desenha a elipse na imagem e atribui o ID i na máscara de instâncias
            draw_img.ellipse(bbox, fill=fg_color)
            draw_instance.ellipse(bbox, fill=i)

        img_np = np.array(img_pil, dtype=np.float32)
        instance_np = np.array(instance_pil, dtype=np.int32)

        # 3. Adicionar ruído gaussiano 
        noise_std = np.random.uniform(5.0, 15.0)
        noise = np.random.normal(0, noise_std, (h, w))
        img_np = np.clip(img_np + noise, 0, 255) / 255.0

        # 4. Replicar para 3 canais (RGB) para alimentar o encoder ResNet
        img_rgb = np.stack([img_np] * 3, axis=0) # Formato (3, H, W)

        # 5. Máscara binária (fundo = 0, objeto = 1)
        binary_mask = (instance_np > 0).astype(np.float32)[np.newaxis, :, :] # Formato (1, H, W)

        # Converter para Tensors do PyTorch
        img_tensor = torch.tensor(img_rgb, dtype=torch.float32)
        binary_mask_tensor = torch.tensor(binary_mask, dtype=torch.float32)
        instance_mask_tensor = torch.tensor(instance_np, dtype=torch.long)

        return img_tensor, binary_mask_tensor, instance_mask_tensor


if __name__ == "__main__":
    dataset = SyntheticEllipseDataset(num_samples=10)
    img, bmask, imask = dataset[0]
    print("SyntheticEllipseDataset testado com sucesso!")
    print(f"Dimensão da Imagem: {img.shape}")
    print(f"Dimensão da Máscara Binária: {bmask.shape}")
    print(f"Dimensão da Máscara de Instâncias: {imask.shape}")
    print(f"Número de elipses na amostra 0: {imask.max().item()}")
