from scipy.ndimage import label


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
