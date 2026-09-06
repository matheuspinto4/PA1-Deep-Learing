import numpy as np
from scipy import ndimage as ndi


def instance_mask_to_three_class(instance_mask, border_size=2):
    label_map = np.zeros_like(instance_mask, dtype=np.int64)
    structure = ndi.generate_binary_structure(2, 1)

    for instance_id in np.unique(instance_mask):
        if instance_id == 0:
            continue

        instance_binary = instance_mask == instance_id
        eroded = ndi.binary_erosion(instance_binary, structure=structure, iterations=border_size)

        border = instance_binary & ~eroded

        label_map[border] = 2
        label_map[eroded] = 1

    return label_map


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from dataset import SyntheticEllipseDataset

    dataset = SyntheticEllipseDataset(num_samples=1)
    _, _, instance_mask = dataset[0]
    instance_mask_np = instance_mask.numpy()

    three_class = instance_mask_to_three_class(instance_mask_np, border_size=2)

    num_instances = int(instance_mask_np.max())
    interior_ids = np.unique(instance_mask_np[three_class == 1])
    instances_sem_interior = num_instances - len(interior_ids[interior_ids > 0])

    print(f"Nucleos da amostra: {num_instances}")
    print(f"Pixels de fundo:   {(three_class == 0).sum()}")
    print(f"Pixels de interior:   {(three_class == 1).sum()}")
    print(f"Pixels de fronteira:   {(three_class == 2).sum()}")
    print(f"Nucleos que perderam o interior por completo: {instances_sem_interior}")

    plt.figure(figsize=(6, 6))
    plt.imshow(three_class, cmap='viridis')
    plt.title("Rotulo de 3 classes (0=fundo, 1=interior, 2=fronteira)")
    plt.colorbar(ticks=[0, 1, 2])
    plt.savefig(r"resultados/imagens/three_class_sanity_check.png")
    print("Visualização salva em 'resultados/imagens/three_class_sanity_check.png'")