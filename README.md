# Where is Rhyming?

Investigating where rhyming knowledge is encoded in language models and whether each IPA phoneme has a distinct subspace.

## Key Findings

- **Phonetic info is front-loaded:** Linear probes peak at layer 0 (F1=0.41) with no improvement in deeper layers. Rhyming knowledge is mostly in the embedding table.
- **Phoneme directions exist but overlap:** Each phoneme has a distinct direction in embedding space (5× more aligned than random), but they are not orthogonal subspaces. Directions become more compressed in deeper layers.
- **Context-dependent rhyming is hard:** GPT-4.1 achieves 100% on standard rhyming but only 75% on heteronyms, with a "primary pronunciation bias" (92% on primary vs 58% on secondary pronunciations).
- **No clean articulatory clustering in GPT-2:** Unlike Llama-3.2's phoneme mover heads, GPT-2's general residual stream doesn't organize phonemes by articulatory features.

## Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install torch transformers transformer-lens numpy pandas scikit-learn matplotlib seaborn panphon scipy openai

# Run experiments
python src/data_prep.py              # Build word→IPA dataset (13K words)
python src/exp1_layer_probing.py     # Layer-wise phoneme probing (~10 min)
python src/exp2_subspace_geometry.py # Subspace analysis (~2 min)
python src/exp3_context_rhyming.py   # API-based heteronym test (~1 min)
python src/visualizations.py         # Generate plots
```

Requires: GPU (tested on RTX A6000), OpenAI API key for Experiment 3.

## File Structure

```
src/                          # Experiment code
├── data_prep.py              # Build word→IPA dataset
├── exp1_layer_probing.py     # Layer-wise phoneme probing
├── exp2_subspace_geometry.py # Phoneme subspace analysis
├── exp3_context_rhyming.py   # Context-dependent rhyming (API)
└── visualizations.py         # Generate all plots
results/                      # Outputs
├── plots/                    # All visualizations
├── *.json                    # Raw results
└── probe_weights/            # Saved model weights
REPORT.md                     # Full research report
planning.md                   # Research plan
literature_review.md          # Literature survey
```

See [REPORT.md](REPORT.md) for the full research report with methodology, results, and analysis.
