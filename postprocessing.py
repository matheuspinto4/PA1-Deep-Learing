import numpy as np
from scipy.ndimage import label
from skimage.segmentation import watershed


def connected_components_to_instances(binary_pred):
    """
    Metodo ingenuo de extracao de instancias (Parte 1): limiar + componentes
    conexos. Recebe uma mascara binaria (0/1) ja limiarizada e devolve um
    mapa de instancias (cada objeto com um id inteiro > 0), no mesmo formato
    da instance_mask do dataset -- para ser comparado direto por
    metrics.evaluate_image_ap.
    """
    instance_mask, _ = label(binary_pred)
    return instance_mask



def watershed_to_instances(class_probs):
    """
    Decodifica a saida de 3 classes da Trilha A (fundo/interior/fronteira)
    em um mapa de instancias, via watershed controlado por marcadores.

    class_probs: array (3, H, W) com as probabilidades softmax de cada
                 classe -- indice 0 = fundo, 1 = interior, 2 = fronteira.

    Cada componente conexo da classe interior vira um marcador (uma
    "nascente"). A agua de cada nascente sobe seguindo a superficie de
    elevacao -- aqui, 1 - prob(interior) -- e para de se espalhar nos
    pixels classificados como fundo (fora da mascara). Como cada nucleo
    colado tem seu proprio marcador (garantido pela erosao usada para
    gerar o rotulo, ver instance_labels.py), dois nucleos encostados
    terminam decodificados como duas instancias distintas.
    """
    class_pred = np.argmax(class_probs, axis=0)  # (H, W) com valores 0/1/2

    interior_mask = class_pred == 1
    foreground_mask = class_pred != 0  # interior OU fronteira

    markers, num_markers = label(interior_mask)

    if num_markers == 0:
        return np.zeros_like(class_pred, dtype=np.int64)

    elevation = 1.0 - class_probs[1]

    instance_mask = watershed(elevation, markers=markers, mask=foreground_mask)

    return instance_mask.astype(np.int64)