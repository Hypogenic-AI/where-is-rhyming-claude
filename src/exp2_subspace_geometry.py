"""
Experiment 2: Phoneme subspace geometry analysis.
Instead of probe weights (which can be collinear under Ridge), we compute
"phoneme direction vectors" as the difference between mean activations of words
containing vs not containing each phoneme. This directly measures how phonemes
are encoded in the model's representation space.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = Path("results/token_phoneme_data.json")
PROBE_RESULTS_PATH = Path("results/layer_probe_results.json")
WEIGHTS_DIR = Path("results/probe_weights")
PLOTS_DIR = Path("results/plots")

# IPA articulatory feature classification
VOWELS = {
    "ɪ", "iː", "i", "ɛ", "æ", "ʌ", "ɑ", "ɔ", "ʊ", "uː", "u",
    "ə", "ɝ", "ɜ", "e", "o", "a", "ɒ", "ɐ", "aː", "eː", "oː",
    "a͜ɪ", "a͜ʊ", "ɔ͜ɪ",
}

CONSONANT_MANNER = {
    "plosive": {"p", "b", "t", "d", "k", "ɡ", "g", "ʔ", "pʰ", "tʰ", "kʰ"},
    "fricative": {"f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h", "x"},
    "affricate": {"t͡ʃ", "d͡ʒ"},
    "nasal": {"m", "n", "ŋ", "m̩", "n̩"},
    "approximant": {"l", "ɹ", "w", "j", "ɻ", "l̩", "l̴"},
    "tap": {"ɾ"},
}

CONSONANT_VOICING = {
    "voiced": {"b", "d", "ɡ", "g", "v", "ð", "z", "ʒ", "d͡ʒ", "m", "n", "ŋ",
               "l", "ɹ", "w", "j", "ɻ", "ɾ", "m̩", "n̩", "l̩", "l̴"},
    "voiceless": {"p", "t", "k", "f", "θ", "s", "ʃ", "h", "t͡ʃ", "ʔ",
                  "pʰ", "tʰ", "kʰ", "x"},
}

VOWEL_HEIGHT = {
    "high": {"i", "iː", "ɪ", "u", "uː", "ʊ"},
    "mid": {"e", "eː", "ɛ", "ə", "ɝ", "ɜ", "o", "oː", "ɔ", "ʌ"},
    "low": {"æ", "a", "aː", "ɑ", "ɒ", "ɐ"},
}

VOWEL_BACKNESS = {
    "front": {"i", "iː", "ɪ", "e", "eː", "ɛ", "æ"},
    "central": {"ə", "ɝ", "ɜ", "ʌ", "a", "aː", "ɐ"},
    "back": {"u", "uː", "ʊ", "o", "oː", "ɔ", "ɑ", "ɒ"},
}


def classify_phoneme(p):
    if p in VOWELS:
        return "vowel"
    for manner, phones in CONSONANT_MANNER.items():
        if p in phones:
            return manner
    return "other"


def get_voicing(p):
    if p in CONSONANT_VOICING.get("voiced", set()):
        return "voiced"
    if p in CONSONANT_VOICING.get("voiceless", set()):
        return "voiceless"
    return "n/a"


def get_vowel_height(p):
    for h, phones in VOWEL_HEIGHT.items():
        if p in phones:
            return h
    return "n/a"


def get_vowel_backness(p):
    for b, phones in VOWEL_BACKNESS.items():
        if p in phones:
            return b
    return "n/a"


def compute_phoneme_directions(activations, labels, phoneme_list, min_count=30):
    """Compute phoneme direction vectors as mean_present - mean_absent."""
    n_phonemes = labels.shape[1]
    d = activations.shape[1]
    directions = np.zeros((n_phonemes, d), dtype=np.float64)
    active_mask = np.zeros(n_phonemes, dtype=bool)

    for p_idx in range(n_phonemes):
        present = labels[:, p_idx] == 1
        absent = ~present
        if present.sum() < min_count or absent.sum() < min_count:
            continue
        mean_present = activations[present].mean(axis=0)
        mean_absent = activations[absent].mean(axis=0)
        directions[p_idx] = mean_present - mean_absent
        active_mask[p_idx] = True

    return directions, active_mask


def analyze_subspace_geometry():
    print("Loading data...")
    with open(DATA_PATH) as f:
        data = json.load(f)
    with open(PROBE_RESULTS_PATH) as f:
        probe_results = json.load(f)

    phoneme_list = data["phoneme_list"]
    labels = np.array([w["multi_hot"] for w in data["words"]], dtype=np.float32)

    # Load embedding weights for activation-based analysis
    # We need to recompute activations — load from saved or recompute
    # For now, use the embedding layer weights from the probing experiment
    # But actually, let's compute directions from mean activations

    # Load model to get embeddings
    print("Loading model for embedding extraction...")
    import torch
    import transformer_lens
    DEVICE = "cuda:0"
    model = transformer_lens.HookedTransformer.from_pretrained("gpt2-medium", device=DEVICE)

    token_ids = [d["token_id"] for d in data["words"]]
    token_tensor = torch.tensor(token_ids, device=DEVICE)

    # Get embeddings (W_E lookup)
    with torch.no_grad():
        embeddings = model.W_E[token_tensor].cpu().numpy()

    # Also get layer 0 residual
    with torch.no_grad():
        _, cache = model.run_with_cache(
            token_tensor.unsqueeze(1),
            names_filter=lambda name: name in ["hook_embed", "blocks.0.hook_resid_post",
                                                 "blocks.11.hook_resid_post",
                                                 "blocks.23.hook_resid_post"]
        )
        embed_acts = cache["hook_embed"][:, 0, :].cpu().numpy()
        layer0_acts = cache["blocks.0.hook_resid_post"][:, 0, :].cpu().numpy()
        layer11_acts = cache["blocks.11.hook_resid_post"][:, 0, :].cpu().numpy()
        layer23_acts = cache["blocks.23.hook_resid_post"][:, 0, :].cpu().numpy()

    del model, cache
    torch.cuda.empty_cache()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Compute phoneme directions at multiple layers
    layers_data = {
        "embedding": embed_acts,
        "layer_0": layer0_acts,
        "layer_11": layer11_acts,
        "layer_23": layer23_acts,
    }

    all_results = {}
    for layer_name, acts in layers_data.items():
        print(f"\n{'='*60}")
        print(f"Analyzing {layer_name}...")

        directions, active_mask = compute_phoneme_directions(acts, labels, phoneme_list)
        active_phonemes = [p for p, m in zip(phoneme_list, active_mask) if m]
        active_dirs = directions[active_mask]

        print(f"  Active phonemes: {len(active_phonemes)}")

        # Normalize directions
        norms = np.linalg.norm(active_dirs, axis=1, keepdims=True)
        normed = active_dirs / (norms + 1e-10)

        # 1. Cosine similarity matrix
        cos_sim = normed @ normed.T
        abs_cos = np.abs(cos_sim)
        np.fill_diagonal(abs_cos, 0)

        mean_abs_cos = abs_cos.mean()
        d = active_dirs.shape[1]
        expected_random = np.sqrt(2 / (np.pi * d))

        print(f"  Mean |cos sim|: {mean_abs_cos:.4f} (random expected: {expected_random:.4f})")
        print(f"  Max |cos sim|: {abs_cos.max():.4f}")

        # 2. Classify phonemes
        classes = [classify_phoneme(p) for p in active_phonemes]
        is_vowel = np.array([1 if c == "vowel" else 0 for c in classes])

        # Silhouette score (vowel vs consonant)
        if len(set(is_vowel)) >= 2 and min(np.bincount(is_vowel)) >= 2:
            sil_vc = silhouette_score(active_dirs, is_vowel)
            print(f"  Silhouette (vowel/consonant): {sil_vc:.4f}")
        else:
            sil_vc = None

        # Manner class silhouette
        manner_labels = []
        manner_map = {}
        for c in classes:
            if c not in manner_map:
                manner_map[c] = len(manner_map)
            manner_labels.append(manner_map[c])
        manner_labels = np.array(manner_labels)

        unique_ml, ml_counts = np.unique(manner_labels, return_counts=True)
        valid_ml = unique_ml[ml_counts >= 2]
        if len(valid_ml) >= 2:
            ml_mask = np.isin(manner_labels, valid_ml)
            if ml_mask.sum() >= 4:
                sil_manner = silhouette_score(active_dirs[ml_mask], manner_labels[ml_mask])
                print(f"  Silhouette (manner): {sil_manner:.4f}")
            else:
                sil_manner = None
        else:
            sil_manner = None

        all_results[layer_name] = {
            "active_phonemes": active_phonemes,
            "mean_abs_cosine": float(mean_abs_cos),
            "expected_random": float(expected_random),
            "silhouette_vc": float(sil_vc) if sil_vc is not None else None,
            "silhouette_manner": float(sil_manner) if sil_manner is not None else None,
            "classes": classes,
        }

        # Only do detailed plots for embedding and layer_0
        if layer_name in ["embedding", "layer_0"]:
            # Cosine similarity heatmap
            fig, ax = plt.subplots(figsize=(14, 12))
            sns.heatmap(cos_sim, xticklabels=active_phonemes, yticklabels=active_phonemes,
                       cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, square=True)
            ax.set_title(f"Cosine Similarity of Phoneme Direction Vectors ({layer_name})")
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / f"phoneme_cosine_{layer_name}.png", dpi=150)
            plt.close()

            # Dendrogram
            Z = linkage(active_dirs, method="ward", metric="euclidean")
            fig, ax = plt.subplots(figsize=(16, 8))
            # Color by vowel/consonant
            leaf_colors = {i: "red" if is_vowel[i] else "blue" for i in range(len(active_phonemes))}
            dendrogram(Z, labels=active_phonemes, leaf_rotation=90, leaf_font_size=10, ax=ax)
            ax.set_title(f"Hierarchical Clustering of Phoneme Directions ({layer_name})")
            ax.set_ylabel("Ward Distance")
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / f"phoneme_dendrogram_{layer_name}.png", dpi=150)
            plt.close()

            # PCA
            pca = PCA(n_components=5)
            pca_coords = pca.fit_transform(active_dirs)
            explained = pca.explained_variance_ratio_
            print(f"  PCA explained variance: {[f'{v:.3f}' for v in explained[:5]]}")

            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            colors = ["red" if c == "vowel" else "blue" for c in classes]

            for i, (p, xy, cl) in enumerate(zip(active_phonemes, pca_coords, colors)):
                axes[0].scatter(xy[0], xy[1], c=cl, s=100, alpha=0.7, zorder=5)
                axes[0].annotate(p, (xy[0], xy[1]), fontsize=10, ha="center", va="bottom")
            axes[0].set_xlabel(f"PC1 ({explained[0]:.1%})")
            axes[0].set_ylabel(f"PC2 ({explained[1]:.1%})")
            axes[0].set_title("PC1 vs PC2")
            axes[0].grid(True, alpha=0.3)

            for i, (p, xy, cl) in enumerate(zip(active_phonemes, pca_coords, colors)):
                axes[1].scatter(xy[1], xy[2], c=cl, s=100, alpha=0.7, zorder=5)
                axes[1].annotate(p, (xy[1], xy[2]), fontsize=10, ha="center", va="bottom")
            axes[1].set_xlabel(f"PC2 ({explained[1]:.1%})")
            axes[1].set_ylabel(f"PC3 ({explained[2]:.1%})")
            axes[1].set_title("PC2 vs PC3")
            axes[1].grid(True, alpha=0.3)

            from matplotlib.patches import Patch
            for ax in axes:
                ax.legend(handles=[Patch(color="red", label="Vowel"),
                                  Patch(color="blue", label="Consonant")])

            plt.suptitle(f"PCA of Phoneme Direction Vectors ({layer_name})", fontsize=14)
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / f"phoneme_pca_{layer_name}.png", dpi=150)
            plt.close()

            # Vowel-specific PCA
            vowel_mask_arr = np.array([c == "vowel" for c in classes])
            vowel_phonemes = [p for p, m in zip(active_phonemes, vowel_mask_arr) if m]
            vowel_dirs = active_dirs[vowel_mask_arr]

            if len(vowel_phonemes) >= 3:
                pca_v = PCA(n_components=min(3, len(vowel_phonemes)))
                vowel_pca = pca_v.fit_transform(vowel_dirs)
                v_explained = pca_v.explained_variance_ratio_

                fig, ax = plt.subplots(figsize=(10, 8))
                height_colors = {"high": "green", "mid": "orange", "low": "red", "n/a": "gray"}
                for i, p in enumerate(vowel_phonemes):
                    h = get_vowel_height(p)
                    color = height_colors.get(h, "gray")
                    ax.scatter(vowel_pca[i, 0], vowel_pca[i, 1], c=color, s=150, alpha=0.8, zorder=5)
                    ax.annotate(p, (vowel_pca[i, 0], vowel_pca[i, 1]),
                               fontsize=14, ha="center", va="bottom", fontweight="bold")

                ax.set_xlabel(f"PC1 ({v_explained[0]:.1%})")
                ax.set_ylabel(f"PC2 ({v_explained[1]:.1%})")
                ax.set_title(f"Vowel Directions PCA ({layer_name}) — cf. IPA Vowel Chart")
                ax.legend(handles=[Patch(color=c, label=h)
                                  for h, c in height_colors.items() if h != "n/a"],
                         title="Vowel Height")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(PLOTS_DIR / f"vowel_chart_{layer_name}.png", dpi=150)
                plt.close()
                print(f"  Saved vowel chart ({len(vowel_phonemes)} vowels)")

            # Consonant voicing analysis
            cons_mask = np.array([c != "vowel" for c in classes])
            cons_phonemes = [p for p, m in zip(active_phonemes, cons_mask) if m]
            cons_dirs = active_dirs[cons_mask]

            if len(cons_phonemes) >= 4:
                voicing_labels = [get_voicing(p) for p in cons_phonemes]
                voiced_mask = np.array([v == "voiced" for v in voicing_labels])
                voiceless_mask = np.array([v == "voiceless" for v in voicing_labels])

                if voiced_mask.sum() >= 2 and voiceless_mask.sum() >= 2:
                    pca_c = PCA(n_components=3)
                    cons_pca = pca_c.fit_transform(cons_dirs)
                    c_explained = pca_c.explained_variance_ratio_

                    fig, ax = plt.subplots(figsize=(10, 8))
                    for i, p in enumerate(cons_phonemes):
                        color = "blue" if voicing_labels[i] == "voiced" else "orange"
                        ax.scatter(cons_pca[i, 0], cons_pca[i, 1], c=color, s=120, alpha=0.8, zorder=5)
                        ax.annotate(p, (cons_pca[i, 0], cons_pca[i, 1]),
                                   fontsize=11, ha="center", va="bottom")

                    ax.set_xlabel(f"PC1 ({c_explained[0]:.1%})")
                    ax.set_ylabel(f"PC2 ({c_explained[1]:.1%})")
                    ax.set_title(f"Consonant Directions PCA ({layer_name}) — Voicing")
                    ax.legend(handles=[Patch(color="blue", label="Voiced"),
                                      Patch(color="orange", label="Voiceless")])
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(PLOTS_DIR / f"consonant_voicing_{layer_name}.png", dpi=150)
                    plt.close()
                    print(f"  Saved consonant voicing plot ({len(cons_phonemes)} consonants)")

    # Save all analysis results
    with open("results/subspace_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to results/subspace_analysis.json")


if __name__ == "__main__":
    analyze_subspace_geometry()
