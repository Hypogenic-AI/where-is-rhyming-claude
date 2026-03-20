# Literature Review: Where is Rhyming?

## Research Area Overview

This literature review investigates how language models (LMs) represent phonetic information, particularly rhyming knowledge, despite receiving no explicit phonetic or auditory input. The core research question asks: **where is rhyming encoded within language models, and does there exist a subspace for each IPA sound?**

The research sits at the intersection of mechanistic interpretability, phonology, and representation learning. Key themes include: (1) probing for phonetic information in LM embeddings and hidden states, (2) identifying specific model components (attention heads, MLPs) that process phonetic information, and (3) understanding the geometric organization of phoneme representations in latent space.

---

## Key Papers

### Paper 1: I Have No Mouth, and I Must Rhyme (McLaughlin, Khurana, Merullo 2025)
- **Source**: ICML 2025 Workshop on Assessing World Models (arXiv:2508.02527)
- **Key Contribution**: First rigorous study of internal phonetic representations in a text-based LLM (Llama-3.2-1B-Instruct)
- **Methodology**:
  - **Linear probing**: Trained multi-hot linear probe on embeddings to predict IPA phonemes. Achieved ~96% accuracy (vs 42% random baseline), showing phonetic info is linearly recoverable from token embeddings.
  - **Causal interventions**: Manipulated embedding-space phoneme vectors (rows of probe matrix) to change rhyming output. Adding c(μ - ξ) to embedding E shifts rhyming predictions from vowel ξ to vowel μ.
  - **Activation patching**: Identified "phoneme mover head" H13L12 (Head 13, Layer 12) with mean normalized logit difference of 0.48 (next highest: 0.19). This head attends to the rhyme target word and promotes phonetically similar tokens via its result vectors.
  - **Geometry analysis**: PCA on H13L12 result vectors reveals organized phonetic structure: PC2/PC3 separate voiced/voiceless consonants; PC1/PC2 separate vowel backness classes. Emergent "Llama vowel chart" partially aligns with but differs from human IPA chart.
- **Key Findings**:
  - Phonetic info exists in both embeddings AND residual stream
  - A set of 3 "phoneme mover heads" (H13L12, H21L14, H22L14) with identical attention patterns form a redundant circuit
  - Ablating all 3 breaks single-token rhyming; keeping any 1 preserves it
  - Cross-lingual phonetic features emerge (Arabic, Hindi, Japanese tokens promoted by result vectors)
  - Coherent result vectors are a prerequisite for successful rhyming (55% pass rate with coherent RV vs 0% with incoherent)
- **Datasets Used**: WikiPron (pronunciation reference), Oxford 5000 (word list)
- **Code Available**: Not yet released
- **Relevance**: **DIRECTLY answers our research question.** This is the primary prior work.

### Paper 2: PWESuite: Phonetic Word Embeddings and Tasks They Facilitate (Zouhar et al., 2023)
- **Source**: LREC-COLING 2024 (arXiv:2304.02541)
- **Key Contribution**: Standardized evaluation suite for phonetic word embeddings, including rhyme detection
- **Methodology**: Develops count-based, autoencoder, and metric/contrastive learning methods using articulatory features (PanPhon). Evaluation includes intrinsic (word retrieval, sound similarity correlation) and extrinsic tasks (rhyme detection, cognate detection, sound analogies).
- **Key Finding**: Intrinsic and extrinsic metrics for phonetic embeddings generally correlate, unlike semantic embeddings.
- **Datasets**: CMU Pronouncing Dictionary, 9-language evaluation suite (1.7M entries)
- **Code**: github.com/zouharvi/pwesuite
- **Relevance**: Provides evaluation benchmarks for phonetic embedding quality, including rhyme detection task.

### Paper 3: Small Language Models Also Work With Small Vocabularies (Bunzeck et al., 2024)
- **Source**: BabyLM Challenge, arXiv:2410.01487
- **Key Contribution**: Compares grapheme-based vs phoneme-based character-level Llama models on phonetic tasks including rhyme prediction
- **Results**:
  - Grapheme models: 88.5-91.5% rhyme prediction accuracy
  - Phoneme models: 78.5-85.0% rhyme prediction accuracy
  - Subword BabyLlama: 92.5% (but worse on lexical decision: 69% vs 99%)
  - Phoneme models slightly better on age prediction (61.1% vs 60.5%)
- **Key Finding**: Character-level models learn phonological representations effectively; grapheme models outperform phoneme models on most tasks, suggesting orthographic biases aid learning.
- **Relevance**: Shows that even small character-level LMs encode rhyme information, complementing Merullo's findings on larger subword models.

### Paper 4: Interpreting Character Embeddings With Perceptual Representations (2022)
- **Source**: ACL 2022
- **Key Contribution**: Shows character embeddings in multilingual models encode perceptual properties including sound
- **Relevance**: Establishes that sub-word representations carry phonetic information, supporting the hypothesis that phonetic knowledge is partially in embeddings.

### Paper 5: Deep-speare: A Joint Neural Model of Poetic Language, Meter and Rhyme (Lau et al., 2018)
- **Source**: ACL 2018 (arXiv:1807.03491)
- **Key Contribution**: Joint architecture capturing language, rhyme, and meter for sonnet generation
- **Methodology**: Encodes stress and rhyme using a phonetic encoding layer alongside language modeling
- **Relevance**: Early work on explicit rhyme modeling in neural networks; provides baseline approach.

### Paper 6: Learning Rhyming Constraints using Structured Adversaries (Jhamtani et al., 2019)
- **Source**: arXiv:1907.00707
- **Key Contribution**: Uses structured adversarial training to enforce rhyming constraints in poetry generation without manual phonetic features
- **Relevance**: Shows neural models can learn rhyme constraints implicitly.

### Paper 7: Supervised Rhyme Detection with Siamese Recurrent Networks (Haider & Kuhn, 2018)
- **Source**: 2018
- **Key Contribution**: Siamese RNN architecture for supervised rhyme detection
- **Relevance**: Provides baseline approach for neural rhyme detection.

### Paper 8: DeepRapper: Neural Rap Generation with Rhyme and Rhythm Modeling (Xue et al., 2021)
- **Source**: arXiv:2107.01875
- **Key Contribution**: Transformer-based rap generation with explicit rhyme representation and constraints; generates lyrics in reverse order with rhyme enhancement
- **Relevance**: Shows how rhyme can be explicitly modeled in transformers.

### Paper 9: Probing Subphonemes in Morphology Models (2025)
- **Source**: arXiv:2501.01685
- **Key Contribution**: Probes for sub-phonemic features in morphology models
- **Relevance**: Methodological reference for probing phonetic features at fine granularity.

---

## Common Methodologies

### Probing / Linear Probing
- Train linear classifiers on model representations to predict linguistic properties
- Used in Merullo et al. 2025 (phoneme prediction from embeddings), Bunzeck et al. 2024 (rhyme/age prediction from sentence embeddings)
- Key insight: if a linear probe succeeds, the information is linearly accessible

### Activation Patching / Causal Interventions
- Replace activations from one forward pass with another to measure causal impact
- Merullo et al. 2025: identified phoneme mover head H13L12 via patching
- Standard tool in mechanistic interpretability (Meng et al., 2023)

### Logit Lens
- Decode intermediate representations into vocabulary space
- Merullo et al. 2025: decoded H13L12 result vectors to find phonetically similar tokens

### PCA / Dimensionality Reduction
- Visualize high-dimensional representations
- Merullo et al. 2025: PCA on result vectors reveals vowel organization

---

## Standard Baselines

- **Random embeddings**: ~42% phoneme prediction accuracy (Merullo et al.)
- **Subword-based BabyLlama**: 92.5% rhyme prediction (Bunzeck et al.)
- **CMU Dict / WikiPron**: Standard pronunciation references
- **PanPhon articulatory distance**: Feature edit distance for phonetic similarity

## Evaluation Metrics

- **Phoneme prediction accuracy** (multi-hot probe accuracy)
- **Rhyme prediction accuracy** (probing-based classification)
- **Normalized logit difference** (activation patching impact)
- **Cosine similarity** of result vectors for phonetically similar words
- **Feature edit distance** (articulatory distance via PanPhon)

## Datasets in the Literature

- **WikiPron**: Used by Merullo et al. for pronunciation reference; 3.9M pronunciations, 306 languages
- **Oxford 5000**: Used by Merullo et al. for word filtering; ~5000 common English words
- **CMU Pronouncing Dictionary**: 134K English words with ARPAbet; standard phonetic resource
- **PWESuite-eval**: 1.7M entries across 9 languages with IPA, ARPAbet, orthography
- **BLiMP**: Syntactic evaluation, adapted for phoneme models (Bunzeck et al.)

---

## Gaps and Opportunities

1. **Model generalization**: Merullo et al. only study Llama-3.2-1B-Instruct. Does the same phoneme mover head structure exist in other models (GPT-2, Pythia, Mistral)?
2. **Context-dependent pronunciation**: English has many context-dependent pronunciations (e.g., "read" /riːd/ vs /rɛd/). Current probing uses single-token words only. How does the model handle multi-token words and homographs?
3. **Consonant-level analysis**: Merullo et al. focus heavily on vowels. The consonant analysis (voicing in PC2/PC3) is less developed.
4. **Per-phoneme subspaces**: The hypothesis of "a subspace for each IPA sound" is partially confirmed by the linear probe approach (each row = a phoneme direction) but more rigorous subspace analysis is needed.
5. **Causal chain**: The full circuit from embedding → phoneme mover heads → output is not fully traced. What happens between the embedding and H13L12?
6. **Training dynamics**: When during training do phonetic representations emerge?

---

## Recommendations for Our Experiment

Based on the literature:

- **Primary model**: Llama-3.2-1B-Instruct (following Merullo et al.), with extensions to other models
- **Primary datasets**: WikiPron + Oxford 5000 for pronunciation data; CMU Dict as fallback
- **Recommended baselines**: Random embeddings, PanPhon articulatory distance
- **Recommended metrics**: Linear probe accuracy, activation patching logit difference, PCA visualization
- **Key tools**: TransformerLens or nnsight for mechanistic interpretability; PanPhon for articulatory features
- **Methodological focus**: (1) Replicate Merullo et al.'s probe on embeddings, (2) Extend to residual stream at each layer, (3) Test whether per-phoneme subspaces exist using DLR or other subspace methods, (4) Investigate context-dependent pronunciation
