import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


def load_results(path="resultados/ablacoes_resultados.csv"):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row["gamma"] = float(row["gamma"])
            row["balanced"] = row["balanced"] == "True"
            row["seed"] = int(row["seed"])
            for key in ("iou_fundo", "iou_interior", "iou_fronteira", "mAP", "counting_error"):
                row[key] = float(row[key])
            rows.append(row)
    return rows


def group_and_aggregate(rows, key_fn, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)

    summary = {}
    for key, group_rows in groups.items():
        summary[key] = {m: (np.mean([r[m] for r in group_rows]), np.std([r[m] for r in group_rows]))
                         for m in metrics}
    return summary


def print_summary(summary, title, metrics):
    print(f"\n=== {title} ===")
    for key in sorted(summary.keys(), key=str):
        line = f"{key}: " + " | ".join(f"{m}={summary[key][m][0]:.4f}+/-{summary[key][m][1]:.4f}" for m in metrics)
        print(line)


def summarize_eixo1(rows):
    subset = [r for r in rows if r["gamma"] == 0.0 and r["balanced"] is True]
    summary = group_and_aggregate(subset, key_fn=lambda r: r["decoder_type"],
                                   metrics=["mAP", "iou_fronteira", "counting_error"])
    print_summary(summary, "EIXO 1: mecanismo de recuperacao de resolucao (skip vs atrous+ASPP)",
                  metrics=["mAP", "iou_fronteira", "counting_error"])
    return summary


def summarize_eixo2(rows):
    subset = [r for r in rows if r["decoder_type"] == "skip"]
    summary = group_and_aggregate(subset, key_fn=lambda r: (r["gamma"], r["balanced"]),
                                   metrics=["mAP", "iou_fronteira"])
    print_summary(summary, "EIXO 2: funcao de perda (gamma x balanceamento)", metrics=["mAP", "iou_fronteira"])
    return summary


def plot_eixo2(summary, save_path="resultados/imagens/eixo2_loss_ablation.png"):
    fig, ax = plt.subplots(figsize=(8, 5))

    for balanced_value, label, color in [(False, "sem peso", "crimson"), (True, "com peso (balanceada)", "navy")]:
        gammas = sorted(g for (g, b) in summary.keys() if b == balanced_value)
        means = [summary[(g, balanced_value)]["iou_fronteira"][0] for g in gammas]
        stds = [summary[(g, balanced_value)]["iou_fronteira"][1] for g in gammas]
        ax.errorbar(gammas, means, yerr=stds, marker="o", label=label, color=color, capsize=4)

    ax.set_xlabel("gamma (foco na Focal Loss)")
    ax.set_ylabel("IoU da classe fronteira (media +/- desvio, 2 seeds)")
    ax.set_title("Efeito de gamma e do balanceamento na classe minoritaria (fronteira)")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nGrafico salvo em: '{save_path}'")


if __name__ == "__main__":
    rows = load_results()
    summarize_eixo1(rows)
    eixo2_summary = summarize_eixo2(rows)
    plot_eixo2(eixo2_summary)
