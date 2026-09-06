import numpy as np
from scipy import ndimage as ndi

from dataset import DSB2018Dataset
from instance_labels import instance_mask_to_three_class


def diagnose_border_size(data_dir="data/stage1_train", image_size=(128, 128)):
    dataset = DSB2018Dataset(root_dir=data_dir, image_size=image_size, preload_in_memory=True)

    areas = []
    lost_fixed_2 = []
    lost_adaptive = []

    structure = ndi.generate_binary_structure(2, 1)

    for _, _, instance_mask in dataset:
        instance_mask_np = instance_mask.numpy()
        for instance_id in np.unique(instance_mask_np):
            if instance_id == 0:
                continue
            instance_binary = instance_mask_np == instance_id
            area = int(instance_binary.sum())

            # erosao fixa antiga, so para efeito de comparacao
            eroded_fixed = ndi.binary_erosion(instance_binary, structure=structure, iterations=2)

            # rotulo real que o pipeline atual vai gerar (erosao adaptativa)
            three_class = instance_mask_to_three_class(instance_binary.astype(np.int64), border_size=2)
            has_interior_adaptive = (three_class == 1).any()

            areas.append(area)
            lost_fixed_2.append(not eroded_fixed.any())
            lost_adaptive.append(not has_interior_adaptive)

    areas = np.array(areas)
    lost_fixed_2 = np.array(lost_fixed_2)
    lost_adaptive = np.array(lost_adaptive)

    print(f"Total de nucleos analisados: {len(areas)}")
    print(f"Area media: {areas.mean():.1f} px | Area mediana: {np.median(areas):.1f} px")
    print(f"\nErosao fixa (border_size=2)  -> {lost_fixed_2.sum()} nucleos sem interior ({100*lost_fixed_2.mean():.1f}%)")
    print(f"Erosao adaptativa (teto=2)   -> {lost_adaptive.sum()} nucleos sem interior ({100*lost_adaptive.mean():.1f}%)")


if __name__ == "__main__":
    diagnose_border_size()
