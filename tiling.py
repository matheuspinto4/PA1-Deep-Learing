import numpy as np
import torch

from postprocessing import watershed_to_instances


def compute_tile_starts(total_width, tile_width, stride):
    starts = list(range(0, total_width - tile_width + 1, stride))
    if starts[-1] + tile_width < total_width:
        starts.append(total_width - tile_width)
    return starts


def run_tiled_inference(model, mosaic_image, tile_width, stride, device):
    """
    Roda o modelo em tiles horizontais sobrepostos (altura inteira, so a
    largura e cortada). Cada tile e decodificado por watershed de forma
    INDEPENDENTE -- e essa independencia que causa o problema que a
    Parte 4 pede pra investigar.
    """
    width = mosaic_image.shape[2]
    starts = compute_tile_starts(width, tile_width, stride)

    tiles = []
    model.eval()
    with torch.no_grad():
        for col_start in starts:
            crop = mosaic_image[:, :, col_start:col_start + tile_width]
            logits = model(crop.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            local_instance_mask = watershed_to_instances(probs)

            tiles.append({"col_start": col_start, "instance_mask": local_instance_mask})

    return tiles, starts


def _core_bounds(starts, tile_width):
    """
    Para cada tile, o intervalo [core_start, core_end) que ele "possui" no
    canvas final -- o ponto medio da faixa de sobreposicao com cada
    vizinho, seguindo a pratica do slide 83 (considerar so a parte
    interna de cada tile).
    """
    bounds = []
    for i, col_start in enumerate(starts):
        core_start = col_start if i == 0 else (starts[i - 1] + tile_width + col_start) // 2
        core_end = col_start + tile_width if i == len(starts) - 1 else (col_start + tile_width + starts[i + 1]) // 2
        bounds.append((core_start, core_end))
    return bounds


def stitch_naive(tiles, starts, tile_width, height, total_width):
    """
    Costura ingenua: cada tile escreve so a sua regiao central no canvas,
    com ids renumerados para serem unicos globalmente -- sem nenhuma
    tentativa de reconciliar objetos cortados na costura.
    """
    canvas = np.zeros((height, total_width), dtype=np.int64)
    bounds = _core_bounds(starts, tile_width)
    next_global_id = 1

    for tile, (core_start, core_end) in zip(tiles, bounds):
        local_col0 = core_start - tile["col_start"]
        local_col1 = core_end - tile["col_start"]
        local_crop = tile["instance_mask"][:, local_col0:local_col1]

        remapped = np.zeros_like(local_crop)
        for local_id in np.unique(local_crop):
            if local_id == 0:
                continue
            remapped[local_crop == local_id] = next_global_id
            next_global_id += 1

        canvas[:, core_start:core_end] = remapped

    return canvas


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x != root_y:
            self.parent[root_x] = root_y


def stitch_with_fusion(tiles, starts, tile_width, height, total_width, iou_threshold=0.3):
    """
    Correcao da Parte 4: alem da costura ingenua, olha a faixa de
    sobreposicao entre cada par de tiles vizinhos -- a unica regiao onde
    dois tiles enxergaram o MESMO pedaco de imagem de forma independente
    -- e funde (via Union-Find) os pares de instancias cujo IoU dentro
    dessa faixa e alto o suficiente pra serem, na pratica, o mesmo objeto
    cortado ao meio.
    """
    bounds = _core_bounds(starts, tile_width)

    tile_id_maps = []
    next_global_id = 1
    for tile in tiles:
        id_map = {}
        for local_id in np.unique(tile["instance_mask"]):
            if local_id == 0:
                continue
            id_map[local_id] = next_global_id
            next_global_id += 1
        tile_id_maps.append(id_map)

    uf = UnionFind()
    merges = []

    for i in range(len(tiles) - 1):
        overlap_start = starts[i + 1]
        overlap_end = starts[i] + tile_width

        left_overlap = tiles[i]["instance_mask"][:, overlap_start - tiles[i]["col_start"]:overlap_end - tiles[i]["col_start"]]
        right_overlap = tiles[i + 1]["instance_mask"][:, overlap_start - tiles[i + 1]["col_start"]:overlap_end - tiles[i + 1]["col_start"]]

        for left_local_id in np.unique(left_overlap):
            if left_local_id == 0:
                continue
            left_mask = left_overlap == left_local_id

            for right_local_id in np.unique(right_overlap):
                if right_local_id == 0:
                    continue
                right_mask = right_overlap == right_local_id

                intersection = (left_mask & right_mask).sum()
                union = (left_mask | right_mask).sum()
                iou = intersection / union if union > 0 else 0.0

                if iou >= iou_threshold:
                    global_left = tile_id_maps[i][left_local_id]
                    global_right = tile_id_maps[i + 1][right_local_id]
                    uf.union(global_left, global_right)
                    merges.append((i, global_left, global_right, iou))

    canvas = np.zeros((height, total_width), dtype=np.int64)
    for tile, (core_start, core_end), id_map in zip(tiles, bounds, tile_id_maps):
        local_col0 = core_start - tile["col_start"]
        local_col1 = core_end - tile["col_start"]
        local_crop = tile["instance_mask"][:, local_col0:local_col1]

        remapped = np.zeros_like(local_crop)
        for local_id, global_id in id_map.items():
            remapped[local_crop == local_id] = uf.find(global_id)

        canvas[:, core_start:core_end] = remapped

    unique_ids = [i for i in np.unique(canvas) if i != 0]
    final_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids, start=1)}
    final_canvas = np.zeros_like(canvas)
    for old_id, new_id in final_map.items():
        final_canvas[canvas == old_id] = new_id

    return final_canvas, merges
