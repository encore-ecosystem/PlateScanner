from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def plot_conf_matrix(tp: int, fp: int, fn: int, output_path: Path, title: str, image_name: str = "0_ConfusionMatrix.png"):
    conf_matrix = np.array([[tp, fn], [fp, 0]])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2*(precision*recall)/(precision+recall)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(conf_matrix, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(conf_matrix[i, j]), ha="center", va="center", color="black")

    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f"Precision: {precision:.4f}    Recall: {recall:.4f}\nF1-score: {f1_score:.4f}\n{title}")

    plt.text(-0.75, 0, "TP", fontsize=14,     color="black", va="center")
    plt.text(-0.75, 1, "FP", fontsize=14,     color="black", va="center")
    plt.text(1.6, 0, "FN", fontsize=14, color="black", va="center")
    plt.text(1.6, 1, "TN", fontsize=14, color="black", va="center")

    plt.tight_layout()
    plt.savefig(output_path / image_name)


__all__ = [
    'plot_conf_matrix',
]