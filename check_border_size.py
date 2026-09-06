import numpy as np
from scipy import ndimage as ndi

from dataset import DSB2018Dataset


def diagnose_border_size(data_dir="data/stage1_train", image_size=(128, 128)):
    dataset = DSB2018Dataset(root_dir=data_dir, image_size=image_size, preload_in_memory=True)

    areas = []
    lost_at_2 = []
    lost_at_1 = []

    structure = ndi.generate_binary_structure(2, 1)

    for _, _, instance_mask in dataset:
        instance_mask_np = instance_mask.numpy()
        for instance_id in np.unique(instance_mask_np):
            if instance_id == 0:
                continue
            instance_binary = instance_mask_np == instance_id
            area = int(instance_binary.sum())

            eroded_2 = ndi.binary_erosion(instance_binary, structure=structure, iterations=2)
            eroded_1 = ndi.binary_erosion(instance_binary, structure=structure, iterations=1)

            areas.append(area)
            lost_at_2.append(not eroded_2.any())
            lost_at_1.append(not eroded_1.any())

    areas = np.array(areas)
    lost_at_2 = np.array(lost_at_2)
    lost_at_1 = np.array(lost_at_1)

    print(f"Total de nucleos analisados: {len(areas)}")
    print(f"Area media: {areas.mean():.1f} px | Area mediana: {np.median(areas):.1f} px")
    print(f"\nborder_size=2 -> {lost_at_2.sum()} nucleos perderam o interior por completo ({100*lost_at_2.mean():.1f}%)")
    print(f"border_size=1 -> {lost_at_1.sum()} nucleos perderam o interior por completo ({100*lost_at_1.mean():.1f}%)")

    print(f"\nArea media dos nucleos que perderam o interior (border=2): {areas[lost_at_2].mean():.1f} px")
    print(f"Area media dos nucleos que MANTIVERAM o interior (border=2): {areas[~lost_at_2].mean():.1f} px")


if __name__ == "__main__":
    diagnose_border_size()
