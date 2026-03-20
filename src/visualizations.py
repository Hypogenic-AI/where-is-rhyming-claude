"""
Generate all visualizations for the report.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PLOTS_DIR = Path("results/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_layer_probing():
    """Plot probe F1 across layers."""
    with open("results/layer_probe_results.json") as f:
        results = json.load(f)

    layers = []
    macro_f1 = []
    micro_f1 = []

    for r in results["layer_results"]:
        if r["layer"] == "embedding":
            layers.append(-0.5)  # Plot slightly before 0
        else:
            layers.append(r["layer_idx"] - 1)
        macro_f1.append(r["macro_f1"])
        micro_f1.append(r["micro_f1"])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(layers, macro_f1, "o-", label="Macro F1", color="steelblue", markersize=6)
    ax.plot(layers, micro_f1, "s-", label="Micro F1", color="coral", markersize=6)

    # Mark embedding specially
    ax.axvline(x=-0.5, color="gray", linestyle="--", alpha=0.5, label="Embedding")
    ax.axvline(x=0, color="green", linestyle=":", alpha=0.5, label="Layer 0 (best)")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Phoneme Probing Accuracy Across Layers (GPT-2 Medium)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(-1, 24, 2))
    ax.set_xticklabels(["Emb"] + list(range(1, 24, 2)))

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "layer_probing_f1.png", dpi=150)
    plt.close()
    print("Saved layer_probing_f1.png")


def plot_orthogonality_across_layers():
    """Plot how phoneme direction orthogonality changes across layers."""
    with open("results/subspace_analysis.json") as f:
        analysis = json.load(f)

    layer_names = ["embedding", "layer_0", "layer_11", "layer_23"]
    mean_cos = [analysis[l]["mean_abs_cosine"] for l in layer_names]
    expected = analysis["embedding"]["expected_random"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(layer_names))
    ax.bar(x, mean_cos, color=["steelblue", "coral", "goldenrod", "green"], alpha=0.8)
    ax.axhline(y=expected, color="red", linestyle="--", linewidth=2,
               label=f"Random baseline ({expected:.4f})")
    ax.set_xticks(x)
    ax.set_xticklabels(["Embedding", "Layer 0", "Layer 11", "Layer 23"])
    ax.set_ylabel("Mean |Cosine Similarity|", fontsize=12)
    ax.set_title("Phoneme Direction Alignment Across Layers", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    for i, v in enumerate(mean_cos):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "orthogonality_across_layers.png", dpi=150)
    plt.close()
    print("Saved orthogonality_across_layers.png")


def plot_heteronym_results():
    """Plot context-dependent rhyming results."""
    with open("results/context_rhyming_results.json") as f:
        results = json.load(f)

    het = results["heteronym_results"]
    words = [r["word"] for r in het]
    pron1_correct = [r["pron1_correct"] for r in het]
    pron2_correct = [r["pron2_correct"] for r in het]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(words))
    width = 0.35

    bars1 = ax.bar(x - width/2, pron1_correct, width, label="Pronunciation 1 (primary)",
                   color="steelblue", alpha=0.8)
    bars2 = ax.bar(x + width/2, pron2_correct, width, label="Pronunciation 2 (secondary)",
                   color="coral", alpha=0.8)

    ax.set_xlabel("Heteronym", fontsize=12)
    ax.set_ylabel("Correct (1=Yes, 0=No)", fontsize=12)
    ax.set_title("GPT-4.1 Context-Dependent Rhyming on Heteronyms", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(words, rotation=45, ha="right")
    ax.legend(fontsize=10)
    ax.set_ylim(-0.1, 1.3)
    ax.grid(True, alpha=0.3, axis="y")

    # Add accuracy annotations
    acc1 = sum(pron1_correct) / len(pron1_correct)
    acc2 = sum(pron2_correct) / len(pron2_correct)
    ax.text(0.02, 0.95, f"Primary pron. accuracy: {acc1:.0%}\nSecondary pron. accuracy: {acc2:.0%}",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "heteronym_results.png", dpi=150)
    plt.close()
    print("Saved heteronym_results.png")


def plot_per_phoneme_f1():
    """Plot F1 scores for individual phonemes at best layer."""
    with open("results/layer_probe_results.json") as f:
        results = json.load(f)

    phonemes = results["phoneme_list"]
    # Get embedding and best layer F1s
    embed_f1 = results["layer_results"][0]["per_phoneme_f1"]
    best_layer = max(results["layer_results"], key=lambda x: x["macro_f1"])
    best_f1 = best_layer["per_phoneme_f1"]

    # Sort by best layer F1
    sorted_indices = np.argsort(best_f1)[::-1]
    top_n = 30  # Show top 30 phonemes

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(top_n)
    width = 0.35

    top_idx = sorted_indices[:top_n]
    ax.bar(x - width/2, [embed_f1[i] for i in top_idx], width,
           label="Embedding", color="steelblue", alpha=0.8)
    ax.bar(x + width/2, [best_f1[i] for i in top_idx], width,
           label=f"{best_layer['layer']}", color="coral", alpha=0.8)

    ax.set_xlabel("Phoneme", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title(f"Per-Phoneme Probe F1: Embedding vs {best_layer['layer']} (Top 30)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([phonemes[i] for i in top_idx], fontsize=9, rotation=45, ha="right")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "per_phoneme_f1.png", dpi=150)
    plt.close()
    print("Saved per_phoneme_f1.png")


def plot_summary_comparison():
    """Summary comparison of all experimental findings."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Layer-wise probe accuracy
    with open("results/layer_probe_results.json") as f:
        results = json.load(f)
    layers = list(range(len(results["layer_results"])))
    macro_f1 = [r["macro_f1"] for r in results["layer_results"]]
    axes[0].plot(layers, macro_f1, "o-", color="steelblue", markersize=4)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Macro F1")
    axes[0].set_title("A) Phoneme Probe Accuracy\nAcross Layers")
    axes[0].grid(True, alpha=0.3)
    axes[0].annotate("Peak: Layer 0", xy=(1, max(macro_f1)),
                    fontsize=9, ha="center", color="red")

    # Panel 2: Direction alignment
    with open("results/subspace_analysis.json") as f:
        analysis = json.load(f)
    layer_names = ["embedding", "layer_0", "layer_11", "layer_23"]
    display_names = ["Emb", "L0", "L11", "L23"]
    mean_cos = [analysis[l]["mean_abs_cosine"] for l in layer_names]
    expected = analysis["embedding"]["expected_random"]
    axes[1].bar(range(4), mean_cos, color=["steelblue", "coral", "goldenrod", "green"], alpha=0.8)
    axes[1].axhline(y=expected, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels(display_names)
    axes[1].set_ylabel("Mean |Cos Sim|")
    axes[1].set_title("B) Phoneme Direction\nOrthogonality")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Panel 3: Context-dependent rhyming
    with open("results/context_rhyming_results.json") as f:
        ctx_results = json.load(f)
    s = ctx_results["summary"]
    categories = ["Standard\nRhyming", "Heteronym\n(Primary)", "Heteronym\n(Secondary)"]
    het_results = ctx_results["heteronym_results"]
    pron1_acc = sum(r["pron1_correct"] for r in het_results) / len(het_results)
    pron2_acc = sum(r["pron2_correct"] for r in het_results) / len(het_results)
    accuracies = [s["standard_accuracy"], pron1_acc, pron2_acc]
    colors = ["green", "steelblue", "coral"]
    bars = axes[2].bar(range(3), accuracies, color=colors, alpha=0.8)
    axes[2].set_xticks(range(3))
    axes[2].set_xticklabels(categories, fontsize=9)
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("C) GPT-4.1 Rhyming Accuracy\n(API Experiment)")
    axes[2].set_ylim(0, 1.15)
    axes[2].grid(True, alpha=0.3, axis="y")
    for bar, acc in zip(bars, accuracies):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{acc:.0%}", ha="center", fontsize=11, fontweight="bold")

    plt.suptitle("Where is Rhyming? — Key Experimental Results", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "summary_comparison.png", dpi=150)
    plt.close()
    print("Saved summary_comparison.png")


if __name__ == "__main__":
    plot_layer_probing()
    plot_orthogonality_across_layers()
    plot_heteronym_results()
    plot_per_phoneme_f1()
    plot_summary_comparison()
    print("\nAll visualizations complete!")
