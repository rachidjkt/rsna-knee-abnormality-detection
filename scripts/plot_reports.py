"""Publication-style aggregate plots and a clearly synthetic MRI contact sheet."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from knee_ai.constants import TARGETS
from knee_ai.data.dataset import StudyDataset
from knee_ai.data.metadata import load_train, load_series
from knee_ai.preprocessing.dicom import sample_indices


def main():
    output = Path("reports/figures")
    output.mkdir(parents=True, exist_ok=True)
    audit = json.loads(Path("reports/metadata_audit.json").read_text())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [3, 1]})
    stats = audit["label_statistics"]
    values = [stats[target]["positive"] for target in TARGETS]
    axes[0].barh(TARGETS, values, color="#35618f")
    axes[0].set_xlim(0, 58)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Positive studies / 58 explicitly labelled studies")
    axes[0].set_title("Official metadata: labelled-subset prevalence")
    for i, value in enumerate(values):
        axes[0].text(value + .7, i, str(value), va="center")
    axes[1].bar(["Labelled", "Unknown"], [audit["fully_labelled_studies"], audit["unlabelled_studies"]], color=["#35618f", "#aaa"])
    axes[1].set_ylabel("Training studies")
    axes[1].set_title("Label availability")
    for i, value in enumerate([58, 4349]):
        axes[1].text(i, value + 70, f"{value:,}", ha="center")
    fig.tight_layout()
    fig.savefig(output / "metadata_prevalence.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 2.8))
    for row, strategy in enumerate(["uniform", "central"]):
        indices = sample_indices(40, 8, strategy)
        ax.scatter(np.arange(40), np.full(40, row), c="#ddd", s=25)
        ax.scatter(indices, np.full(8, row), c="#35618f" if row == 0 else "#b36135", s=90)
    ax.set_yticks([0, 1], ["Full-span uniform", "Central 60% physical-location prior"])
    ax.set_xlabel("Ordered slice index (equally spaced illustrative series)")
    ax.set_title("Controlled sampling conditions: identical series, eight slices per view")
    fig.tight_layout()
    fig.savefig(output / "sampling_design.png", dpi=180)
    plt.close(fig)

    data = Path("outputs/synthetic_data")
    dataset = StudyDataset(load_train(data / "train.csv").iloc[:1], load_series(data / "train_series.csv"),
                           data / "train_images", slices_per_series=4, image_size=64)
    item = dataset[0]
    fig, axes = plt.subplots(3, 4, figsize=(6, 5))
    for index, axis in enumerate(axes.flat):
        axis.imshow(item["images"][index, 0], cmap="gray", vmin=0, vmax=1)
        axis.axis("off")
    for axis, plane in zip(axes[:, 0], ["Sagittal", "Coronal", "Axial"]):
        axis.set_title(plane, fontsize=9)
    fig.suptitle("Synthetic engineering fixtures — no clinical images")
    fig.tight_layout()
    fig.savefig(output / "synthetic_contact_sheet.png", dpi=180)
    plt.close(fig)

    frame = pd.read_csv("outputs/verified_comparison/comparison.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    for seed, group in frame.groupby("seed"):
        ax.plot(group.condition, group.macro_auc_all_12, marker="o", label=f"Seed {seed}")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Synthetic macro ROC-AUC")
    ax.set_title("Engineering comparison only; not MRI validation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "synthetic_comparison.png", dpi=180)
    plt.close(fig)
    print("Saved four figures to reports/figures")


if __name__ == "__main__":
    main()
