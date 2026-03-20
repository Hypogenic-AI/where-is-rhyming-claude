"""
Experiment 1: Layer-wise phoneme probing (optimized).
Extract activations at each layer of GPT-2 medium and train linear probes
to predict IPA phonemes from residual stream representations.
Uses Ridge regression for speed.
"""

import json
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import time
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = Path("results/token_phoneme_data.json")
OUTPUT_PATH = Path("results/layer_probe_results.json")
WEIGHTS_PATH = Path("results/probe_weights")
MODEL_NAME = "gpt2-medium"
BATCH_SIZE = 256
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_data():
    with open(DATA_PATH) as f:
        return json.load(f)


def extract_activations(data, model, batch_size=BATCH_SIZE):
    """Extract embedding and residual stream activations for all tokens."""
    token_ids = [d["token_id"] for d in data["words"]]
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    all_activations = np.zeros((n_layers + 1, len(token_ids), d_model), dtype=np.float32)

    print(f"Extracting activations for {len(token_ids)} tokens across {n_layers} layers...")
    t0 = time.time()

    for batch_start in range(0, len(token_ids), batch_size):
        batch_end = min(batch_start + batch_size, len(token_ids))
        batch_ids = torch.tensor(token_ids[batch_start:batch_end], device=DEVICE).unsqueeze(1)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                batch_ids,
                names_filter=lambda name: name == "hook_embed" or name.endswith("hook_resid_post")
            )

        embed = cache["hook_embed"][:, 0, :].cpu().numpy()
        all_activations[0, batch_start:batch_end] = embed

        for layer in range(n_layers):
            resid = cache[f"blocks.{layer}.hook_resid_post"][:, 0, :].cpu().numpy()
            all_activations[layer + 1, batch_start:batch_end] = resid

        del cache
        torch.cuda.empty_cache()

    print(f"  Done in {time.time() - t0:.1f}s")
    return all_activations


def train_probes(activations, labels, n_layers, phoneme_list):
    """Train per-phoneme Ridge classifiers at each layer."""
    n_samples = activations.shape[1]
    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=SEED)

    results = []
    all_weights = {}
    n_phonemes = labels.shape[1]

    print(f"\nTraining probes (train={len(train_idx)}, test={len(test_idx)}, "
          f"phonemes={n_phonemes})...")

    for layer_idx in range(n_layers + 1):
        layer_name = "embedding" if layer_idx == 0 else f"layer_{layer_idx - 1}"
        X_train = activations[layer_idx, train_idx]
        X_test = activations[layer_idx, test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        # Standardize features per layer
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        t0 = time.time()

        predictions = np.zeros_like(y_test)
        probe_weights = np.zeros((n_phonemes, X_train.shape[1]), dtype=np.float32)
        probe_biases = np.zeros(n_phonemes, dtype=np.float32)
        per_phoneme_f1 = []

        for p_idx in range(n_phonemes):
            pos_count = y_train[:, p_idx].sum()
            if pos_count < 5:
                per_phoneme_f1.append(0.0)
                continue

            clf = RidgeClassifier(alpha=1.0, random_state=SEED)
            clf.fit(X_train_s, y_train[:, p_idx])
            pred = clf.predict(X_test_s)
            predictions[:, p_idx] = pred
            probe_weights[p_idx] = clf.coef_[0]
            probe_biases[p_idx] = clf.intercept_[0]

            f1 = f1_score(y_test[:, p_idx], pred, zero_division=0)
            per_phoneme_f1.append(f1)

        macro_f1 = np.mean([f for f in per_phoneme_f1 if f > 0])
        micro_f1 = f1_score(y_test, predictions, average="micro", zero_division=0)
        sample_acc = np.mean(np.all(predictions == y_test, axis=1))

        elapsed = time.time() - t0
        print(f"  {layer_name}: macro_f1={macro_f1:.4f}, micro_f1={micro_f1:.4f}, "
              f"exact_match={sample_acc:.4f} ({elapsed:.1f}s)")

        results.append({
            "layer": layer_name,
            "layer_idx": layer_idx,
            "macro_f1": float(macro_f1),
            "micro_f1": float(micro_f1),
            "exact_match_accuracy": float(sample_acc),
            "per_phoneme_f1": [float(f) for f in per_phoneme_f1],
            "n_active_phonemes": sum(1 for f in per_phoneme_f1 if f > 0),
        })

        all_weights[layer_name] = {
            "weights": probe_weights,
            "biases": probe_biases
        }

    return results, all_weights, test_idx


def main():
    print(f"Device: {DEVICE}")
    data = load_data()
    labels = np.array([w["multi_hot"] for w in data["words"]], dtype=np.float32)
    print(f"Data: {len(data['words'])} words, {data['num_phonemes']} phonemes")

    print(f"\nLoading {MODEL_NAME} with TransformerLens...")
    import transformer_lens
    model = transformer_lens.HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    print(f"Model: {n_layers} layers, d_model={d_model}")

    activations = extract_activations(data, model, BATCH_SIZE)
    del model
    torch.cuda.empty_cache()

    results, weights, test_idx = train_probes(
        activations, labels, activations.shape[0] - 1, data["phoneme_list"]
    )

    # Save results
    output = {
        "model": MODEL_NAME,
        "seed": SEED,
        "num_words": len(data["words"]),
        "num_phonemes": data["num_phonemes"],
        "n_layers": n_layers,
        "d_model": d_model,
        "phoneme_list": data["phoneme_list"],
        "layer_results": results,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")

    # Save weights for best layer
    WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)
    best_idx = max(range(len(results)), key=lambda i: results[i]["macro_f1"])
    best_layer = results[best_idx]["layer"]
    print(f"Best layer: {best_layer} (macro_f1={results[best_idx]['macro_f1']:.4f})")

    np.save(WEIGHTS_PATH / "best_weights.npy", weights[best_layer]["weights"])
    np.save(WEIGHTS_PATH / "best_biases.npy", weights[best_layer]["biases"])
    np.save(WEIGHTS_PATH / "embedding_weights.npy", weights["embedding"]["weights"])

    # Save all weights for layer comparison
    for layer_name, w in weights.items():
        np.save(WEIGHTS_PATH / f"{layer_name}_weights.npy", w["weights"])

    np.save("results/labels.npy", labels)
    np.save("results/test_idx.npy", test_idx)

    print("Done!")


if __name__ == "__main__":
    main()
