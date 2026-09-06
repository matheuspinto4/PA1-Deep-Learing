import numpy as np
import torch

from postprocessing import watershed_to_instances


def compute_tile_starts(total_size, tile_size, stride):
    starts = list(range(0, total_size - tile_size + 1, stride))
    if starts[-1] + tile_size < total_size:
        starts.append(total_size - tile_size)
    return starts


def run_tiled_inference_2d(model, mosaic_image, tile_height, tile_width, stride_h, stride_w, device):
    """
    Roda o modelo numa grade 2D de tiles sobrepostos. Cada tile e
    decodificado por watershed de forma INDEPENDENTE -- e essa
    independencia que causa o problema que a Parte 4 pede pra investigar.
    """
    height, width = mosaic_image.shape[1], mosaic_image.shape[2]
    row_starts = compute_tile_starts(height, tile_height, stride_h)
    col_starts = compute_tile_starts(width, tile_width, stride_w)

    tiles = {}
    model.eval()
    with torch.no_grad():
        for row_start in row_starts:
            for col_start in col_starts:
                crop = mosaic_image[:, row_start:row_start + tile_height, col_start:col_start + tile_width]
                logits = model(crop.unsqueeze(0).to(device))
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                tiles[(row_start, col_start)] = watershed_to_instances(probs)

    return tiles, row_starts, col_starts


def _axis_core_bounds(starts, extent):
    """
    Para cada posicao ao longo de UM eixo, o intervalo [core_start, core_end)
    que ela "possui" no canvas final -- o ponto medio da faixa de
    sobreposicao com cada vizinho (slide 83: considerar so a parte
    interna de cada tile). Funciona igual para linhas ou colunas.
    """
    bounds = []
    for i, start in enumerate(starts):
        core_start = start if i == 0 else (starts[i - 1] + extent + start) // 2
        core_end = start + extent if i == len(starts) - 1 else (start + extent + starts[i + 1]) // 2
        bounds.append((core_start, core_end))
    return bounds


def stitch_naive_2d(tiles, row_starts, col_starts, tile_height, tile_width, canvas_shape):
    """
    Costura ingenua: cada tile escreve so a sua regiao central no canvas,
    com ids renumerados globalmente -- sem reconciliar objetos cortados
    nas costuras (horizontais OU verticais).
    """
    canvas = np.zeros(canvas_shape, dtype=np.int64)
    row_bounds = _axis_core_bounds(row_starts, tile_height)
    col_bounds = _axis_core_bounds(col_starts, tile_width)
    next_global_id = 1

    for r_idx, row_start in enumerate(row_starts):
        row_core_start, row_core_end = row_bounds[r_idx]
        for c_idx, col_start in enumerate(col_starts):
            col_core_start, col_core_end = col_bounds[c_idx]

            local_mask = tiles[(row_start, col_start)]
            local_crop = local_mask[row_core_start - row_start:row_core_end - row_start,
                                     col_core_start - col_start:col_core_end - col_start]

            remapped = np.zeros_like(local_crop)
            for local_id in np.unique(local_crop):
                if local_id == 0:
                    continue
                remapped[local_crop == local_id] = next_global_id
                next_global_id += 1

            canvas[row_core_start:row_core_end, col_core_start:col_core_end] = remapped

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


def _fuse_along_axis(mask_a, offset_a, mask_b, offset_b, boundary, overlap_start, overlap_end,
                      candidate_margin, compare_margin, iou_threshold,
                      id_map_a, id_map_b, uf, merges_log, seam_label):
    """
    Compara duas mascaras locais vizinhas ao longo do eixo das colunas
    (para fusao vertical, o chamador passa as mascaras transpostas, e
    "coluna" aqui vira "linha" na imagem de verdade). Em dois estagios:
    1) candidatos = instancias com pixel bem em cima da linha de corte
       (+/- candidate_margin, poucos pixels de tolerancia);
    2) o IoU desses candidatos e calculado numa janela mais larga
       (+/- compare_margin), para um sinal mais robusto.
    Uma margem larga demais no passo 1 pega objetos inteiros que so por
    acaso estao perto da linha, sem nunca terem sido cortados.
    """
    cand_start = max(boundary - candidate_margin, overlap_start)
    cand_end = min(boundary + candidate_margin, overlap_end)
    cmp_start = max(boundary - compare_margin, overlap_start)
    cmp_end = min(boundary + compare_margin, overlap_end)

    a_cand = mask_a[:, cand_start - offset_a:cand_end - offset_a]
    b_cand = mask_b[:, cand_start - offset_b:cand_end - offset_b]
    a_cmp = mask_a[:, cmp_start - offset_a:cmp_end - offset_a]
    b_cmp = mask_b[:, cmp_start - offset_b:cmp_end - offset_b]

    a_candidates = [v for v in np.unique(a_cand) if v != 0]
    b_candidates = [v for v in np.unique(b_cand) if v != 0]

    for a_local_id in a_candidates:
        a_mask = a_cmp == a_local_id
        for b_local_id in b_candidates:
            b_mask = b_cmp == b_local_id

            intersection = (a_mask & b_mask).sum()
            union = (a_mask | b_mask).sum()
            iou = intersection / union if union > 0 else 0.0

            if iou >= iou_threshold:
                global_a = id_map_a[a_local_id]
                global_b = id_map_b[b_local_id]
                uf.union(global_a, global_b)
                merges_log.append((seam_label, global_a, global_b, iou))


def stitch_with_fusion_2d(tiles, row_starts, col_starts, tile_height, tile_width, canvas_shape,
                           iou_threshold=0.3, candidate_margin=3, compare_margin=20):
    """
    Correcao da Parte 4, em 2D: alem das costuras horizontais (tiles
    vizinhos na mesma linha), tambem funde nas costuras verticais (tiles
    vizinhos na mesma coluna). O caso vertical reaproveita a mesma
    _fuse_along_axis transpondo as mascaras, entao a logica de
    candidato/comparacao so precisa estar certa em um lugar.
    """
    row_bounds = _axis_core_bounds(row_starts, tile_height)
    col_bounds = _axis_core_bounds(col_starts, tile_width)

    tile_id_maps = {}
    next_global_id = 1
    for key, mask in tiles.items():
        id_map = {}
        for local_id in np.unique(mask):
            if local_id == 0:
                continue
            id_map[local_id] = next_global_id
            next_global_id += 1
        tile_id_maps[key] = id_map

    uf = UnionFind()
    merges = []

    # costuras horizontais: tiles vizinhos na mesma linha (mesmo row_start)
    for row_start in row_starts:
        for c_idx in range(len(col_starts) - 1):
            col_left, col_right = col_starts[c_idx], col_starts[c_idx + 1]
            boundary = col_bounds[c_idx][1]
            overlap_start, overlap_end = col_right, col_left + tile_width

            _fuse_along_axis(
                tiles[(row_start, col_left)], col_left,
                tiles[(row_start, col_right)], col_right,
                boundary, overlap_start, overlap_end,
                candidate_margin, compare_margin, iou_threshold,
                tile_id_maps[(row_start, col_left)], tile_id_maps[(row_start, col_right)],
                uf, merges, seam_label=f"H row={row_start} col={col_left}|{col_right}"
            )

    # costuras verticais: tiles vizinhos na mesma coluna (mesmo col_start) -- mascaras transpostas
    for col_start in col_starts:
        for r_idx in range(len(row_starts) - 1):
            row_top, row_bottom = row_starts[r_idx], row_starts[r_idx + 1]
            boundary = row_bounds[r_idx][1]
            overlap_start, overlap_end = row_bottom, row_top + tile_height

            _fuse_along_axis(
                tiles[(row_top, col_start)].T, row_top,
                tiles[(row_bottom, col_start)].T, row_bottom,
                boundary, overlap_start, overlap_end,
                candidate_margin, compare_margin, iou_threshold,
                tile_id_maps[(row_top, col_start)], tile_id_maps[(row_bottom, col_start)],
                uf, merges, seam_label=f"V col={col_start} row={row_top}|{row_bottom}"
            )

    canvas = np.zeros(canvas_shape, dtype=np.int64)
    for r_idx, row_start in enumerate(row_starts):
        row_core_start, row_core_end = row_bounds[r_idx]
        for c_idx, col_start in enumerate(col_starts):
            col_core_start, col_core_end = col_bounds[c_idx]

            local_mask = tiles[(row_start, col_start)]
            id_map = tile_id_maps[(row_start, col_start)]
            local_crop = local_mask[row_core_start - row_start:row_core_end - row_start,
                                     col_core_start - col_start:col_core_end - col_start]

            remapped = np.zeros_like(local_crop)
            for local_id, global_id in id_map.items():
                remapped[local_crop == local_id] = uf.find(global_id)

            canvas[row_core_start:row_core_end, col_core_start:col_core_end] = remapped

    unique_ids = [i for i in np.unique(canvas) if i != 0]
    final_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids, start=1)}
    final_canvas = np.zeros_like(canvas)
    for old_id, new_id in final_map.items():
        final_canvas[canvas == old_id] = new_id

    return final_canvas, merges
