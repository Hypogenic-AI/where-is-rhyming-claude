# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "Where is Rhyming?" — investigating where rhyming knowledge is represented within language models and whether there exists a subspace for each IPA sound.

## Papers
Total papers downloaded: 10

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| I Have No Mouth, and I Must Rhyme | McLaughlin, Khurana, Merullo | 2025 | papers/merullo2025_rhyme_llama.pdf | **PRIMARY** - phoneme mover head in Llama 3.2, IPA subspaces |
| PWESuite | Zouhar et al. | 2023 | papers/pwesuite_2023.pdf | Phonetic embedding evaluation with rhyme detection |
| Baby Llamas | Bunzeck et al. | 2024 | papers/baby_llamas_2024.pdf | Grapheme vs phoneme LMs, rhyme prediction probing |
| Interpreting Character Embeddings | Various | 2022 | papers/interpreting_char_embeddings_2022.pdf | Sound info in character embeddings |
| Deep-speare | Lau et al. | 2018 | papers/deep_speare_2018.pdf | Joint language+rhyme+meter model |
| Learning Rhyming Constraints | Jhamtani et al. | 2019 | papers/learning_rhyming_constraints_2019.pdf | Adversarial rhyme learning |
| Supervised Rhyme Detection | Haider, Kuhn | 2018 | papers/rhyme_detection_siamese_2018.pdf | Siamese RNN rhyme detection |
| DeepRapper | Xue et al. | 2021 | papers/deep_rapper_2021.pdf | Transformer rhyme modeling for rap |
| Rhythmic Verse Generation | Hopkins, Kiela | 2017 | papers/rhythmic_verse_2017.pdf | Phonetic encoding for verse |
| Probing Subphonemes | Various | 2025 | papers/probing_subphonemes_2025.pdf | Sub-phonemic probing methodology |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 5

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| WikiPron | GitHub/CUNY-CL | 3.9M entries | Pronunciation reference | datasets/wikipron/ | 306 languages, TSV format |
| Oxford 5000 | GitHub/tyypgzl | ~5000 words | Word filtering | datasets/oxford_5000/ | JSON with pronunciation |
| PWESuite-eval | HuggingFace | 1.7M entries | Phonetic evaluation | datasets/pwesuite_eval/ | 9 languages, IPA+ARPAbet |
| CMU Dict | pip package | 134K words | English pronunciation | pip: cmudict | ARPAbet format |
| PanPhon | pip package | 5000+ segments | Articulatory features | pip: panphon | 21 articulatory features |

See datasets/README.md for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 1 (+4 recommended pip packages)

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| PWESuite | github.com/zouharvi/pwesuite | Phonetic embedding evaluation | code/pwesuite/ | Rhyme detection benchmarks |
| TransformerLens | github.com/TransformerLensOrg/TransformerLens | Mechanistic interpretability | pip: transformer-lens | For probing attention heads |
| nnsight | github.com/ndif-team/nnsight | Model interpretation | pip: nnsight | Alternative to TransformerLens |
| PanPhon | github.com/dmort27/panphon | Articulatory features | pip: panphon | Phoneme distance computation |
| PyVene | github.com/stanfordnlp/pyvene | Causal interventions | pip: pyvene | Model editing/steering |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder tool with diligent mode for two queries covering rhyming/phonetics/LM representations
2. Combined results from 3 search runs yielding 369 unique papers
3. Filtered to 85 papers with relevance >= 2, focused on top 18 most relevant
4. Downloaded 10 papers directly from arXiv and ACL Anthology

### Selection Criteria
- Direct relevance to phonetic representations in text-based LLMs (not speech models)
- Rhyme modeling and detection in neural networks
- Probing methodology for phonological features
- Recency (2017-2025) with emphasis on 2023-2025

### Challenges Encountered
- Two papers downloaded with wrong arXiv IDs (quantum physics and robotic surgery papers instead of NLP papers — phonological_vector_arithmetic_2026 and hidden_folk_2023). These were removed.
- Semantic Scholar API rate-limited/unavailable for many lookups
- Merullo et al. 2025 has no released code yet

### Gaps and Workarounds
- No code from the primary paper (Merullo 2025) — experiment runner will need to reimplement based on paper methodology
- The "phonological vector arithmetic" paper (2026) could not be found with correct arXiv ID

## Recommendations for Experiment Design

Based on gathered resources, recommend:

1. **Primary model**: Llama-3.2-1B-Instruct (as in Merullo et al. 2025)
2. **Primary dataset(s)**: WikiPron (English pronunciations) + Oxford 5000 (word filtering) + CMU Dict (ARPAbet reference)
3. **Baseline methods**:
   - Random embedding probe (42% baseline from Merullo)
   - PanPhon articulatory distance
   - PWESuite phonetic embeddings
4. **Evaluation metrics**:
   - Linear probe accuracy for phoneme prediction
   - Activation patching normalized logit difference
   - PCA visualization of phoneme geometry
   - Rhyme task pass rate with/without ablation
5. **Code to adapt/reuse**:
   - TransformerLens for activation patching, logit lens, and head analysis
   - PanPhon for articulatory feature computation
   - PWESuite for rhyme detection evaluation
   - scikit-learn for linear probing
6. **Experiment priorities**:
   - (a) Replicate Merullo's linear probe on embeddings
   - (b) Extend probe to each layer's residual stream to map where phonetic info emerges
   - (c) Test for per-IPA-sound subspaces using probe weight analysis
   - (d) Investigate context-dependent pronunciation (heteronyms)
   - (e) Compare across model families if time permits
