import numpy as np

from dataset import DSB2018Dataset


def resnet34_layer_sequence(dilate_layer3=False, dilate_layer4=False, stop_after=None):
    """
    Gera a sequencia de operacoes (kernel, stride, dilation) do "tronco"
    do ResNet34 (a cadeia principal conv1->conv2 de cada bloco), na ordem
    em que sao aplicadas. dilate_layer3/dilate_layer4 reproduzem a mesma
    troca de stride por dilatacao feita em model.py (_make_dilated) para
    a variante atrous_aspp. stop_after corta a sequencia logo apos um
    estagio (ex: "layer2"), para comparar campo receptivo na MESMA
    resolucao de saida com uma rede mais rasa (sem atrous).
    """
    ops = [(7, 2, 1), (3, 2, 1)]  # conv1 (stem) + maxpool

    stages = [
        ("layer1", 3, 1, 1),
        ("layer2", 4, 2, 1),
        ("layer3", 6, 1 if dilate_layer3 else 2, 2 if dilate_layer3 else 1),
        ("layer4", 3, 1 if dilate_layer4 else 2, 4 if dilate_layer4 else 1),
    ]

    for stage_name, num_blocks, first_stride, dilation in stages:
        for block_idx in range(num_blocks):
            conv1_stride = first_stride if block_idx == 0 else 1
            ops.append((3, conv1_stride, dilation))  # conv1
            ops.append((3, 1, dilation))               # conv2
        if stop_after == stage_name:
            break

    return ops


def compute_receptive_field(ops):
    """
    Formula padrao (slides 35-38): cada camada expande o campo receptivo
    pelo tamanho efetivo do seu kernel (considerando dilatacao), escalado
    pelo "salto" acumulado (produto dos strides anteriores).
    """
    rf, jump = 1, 1
    for kernel, stride, dilation in ops:
        effective_kernel = dilation * (kernel - 1) + 1
        rf = rf + (effective_kernel - 1) * jump
        jump = jump * stride
    return rf, jump


def summarize_receptive_fields():
    full_no_atrous = compute_receptive_field(resnet34_layer_sequence(False, False))
    full_atrous = compute_receptive_field(resnet34_layer_sequence(True, True))
    shallow_same_stride = compute_receptive_field(resnet34_layer_sequence(False, False, stop_after="layer2"))

    print("=== Campo receptivo teorico (ResNet34) ===")
    print(f"Encoder completo, SEM atrous (output_stride=32, usado pelo decoder 'skip'):        "
          f"RF={full_no_atrous[0]}px | jump={full_no_atrous[1]}")
    print(f"Encoder completo, COM atrous (output_stride=8, usado pelo decoder 'atrous_aspp'):   "
          f"RF={full_atrous[0]}px | jump={full_atrous[1]}")
    print(f"Encoder raso, SEM atrous, MESMO output_stride=8 (parado apos layer2):               "
          f"RF={shallow_same_stride[0]}px | jump={shallow_same_stride[1]}")
    print(f"\n-> Na MESMA resolucao de saida (output_stride=8), o atrous aumenta o campo receptivo")
    print(f"   de {shallow_same_stride[0]}px (rede rasa, sem atrous) para {full_atrous[0]}px "
          f"(rede funda, com dilatacao), sem perder resolucao espacial.")
    print(f"\n   RESSALVA: campo receptivo TEORICO cresce rapido e tende a superar o tamanho da")
    print(f"   imagem em redes fundas (conhecido na literatura); o campo receptivo EFETIVO, que")
    print(f"   de fato influencia a predicao de forma significativa, e bem menor. O enunciado pede")
    print(f"   o calculo teorico (slides 35-38), que e o que reportamos acima.")


def compute_object_size_distribution(data_dir="data/stage1_train", image_size=(128, 128)):
    """
    Para cada nucleo do dataset (ja redimensionado para image_size, o
    mesmo espaco em que o modelo opera), calcula o maior lado do
    bounding box -- uma medida de "diametro" mais fiel do que a area
    para comparar contra o campo receptivo (uma medida espacial/linear).
    """
    dataset = DSB2018Dataset(root_dir=data_dir, image_size=image_size, preload_in_memory=True)

    sizes = []
    for _, _, instance_mask in dataset:
        instance_mask_np = instance_mask.numpy()
        for instance_id in np.unique(instance_mask_np):
            if instance_id == 0:
                continue
            ys, xs = np.where(instance_mask_np == instance_id)
            height = ys.max() - ys.min() + 1
            width = xs.max() - xs.min() + 1
            sizes.append(max(height, width))

    return np.array(sizes)


def summarize_object_sizes():
    sizes = compute_object_size_distribution()
    print(f"\n=== Distribuicao do tamanho dos nucleos (maior lado do bounding box, em px) ===")
    print(f"Total de nucleos: {len(sizes)}")
    for p in [50, 75, 90, 95, 99, 100]:
        print(f"  percentil {p}: {np.percentile(sizes, p):.1f}px")
    print(f"  maximo observado: {sizes.max()}px")


if __name__ == "__main__":
    summarize_receptive_fields()
    summarize_object_sizes()
