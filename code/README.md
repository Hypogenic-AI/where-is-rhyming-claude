# Cloned Repositories

## Repo 1: PWESuite
- **URL**: https://github.com/zouharvi/pwesuite
- **Purpose**: Phonetic word embedding evaluation suite with rhyme detection benchmarks
- **Location**: code/pwesuite/
- **Key files**:
  - `compute_embeddings.py` - compute various phonetic embeddings
  - `eval_all.py` - run full evaluation suite
  - `data/` - evaluation data
- **Notes**: Provides standardized evaluation for phonetic embeddings. Includes rhyme detection, cognate detection, and sound analogy tasks.

## Recommended Tools (Install via pip)

### TransformerLens
- **URL**: https://github.com/TransformerLensOrg/TransformerLens
- **Purpose**: Mechanistic interpretability library for probing attention heads and residual stream
- **Install**: `pip install transformer-lens`
- **Key for**: Replicating Merullo et al.'s activation patching and logit lens analysis

### nnsight
- **URL**: https://github.com/ndif-team/nnsight
- **Purpose**: Modern neural network interpretation framework
- **Install**: `pip install nnsight`
- **Key for**: Alternative/complement to TransformerLens for probing

### PanPhon
- **URL**: https://github.com/dmort27/panphon
- **Purpose**: Articulatory feature database for IPA segments; computes phonetic distance
- **Install**: `pip install panphon`
- **Key for**: Computing articulatory distances between phonemes

### PyVene
- **URL**: https://github.com/stanfordnlp/pyvene
- **Purpose**: Causal intervention library for PyTorch models
- **Install**: `pip install pyvene`
- **Key for**: Causal interventions on phonetic representations (alternative to manual patching)

## Notes

The Merullo et al. 2025 paper ("I Have No Mouth, and I Must Rhyme") has not yet released code. When available, it should be cloned here as it is the primary reference implementation.

Key experiment runner dependencies:
- `transformer-lens` or `nnsight` for model internals
- `panphon` for articulatory features
- `cmudict` for pronunciation data
- `torch` for model inference
- `scikit-learn` for linear probes
- `matplotlib` for visualization
