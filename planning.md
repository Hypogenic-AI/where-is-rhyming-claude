# Research Plan: Where is Rhyming?

## Motivation & Novelty Assessment

### Why This Research Matters
Language models can identify rhyming words despite never hearing speech — they only process text. Understanding WHERE this phonetic knowledge lives (embeddings vs. deeper layers) and HOW it's structured (per-phoneme subspaces?) illuminates how neural networks learn emergent linguistic representations. This has implications for interpretability, phonology, and cross-lingual transfer.

### Gap in Existing Work
Merullo et al. (2025) is the only rigorous mechanistic study, conducted solely on Llama-3.2-1B-Instruct. Key gaps:
1. **Layer-by-layer analysis**: They probe embeddings and find phoneme mover heads, but don't systematically track how phonetic information evolves across ALL layers of the residual stream
2. **Per-phoneme subspace geometry**: They show PCA structure but don't rigorously test whether each IPA phoneme has a distinct linear subspace
3. **Context-dependent pronunciation**: Only single-token words studied; heteronyms (read/read, lead/lead) are unexplored
4. **Cross-model generalization**: Only one model tested

### Our Novel Contribution
We extend the Merullo et al. work in three ways:
1. **Layer-wise phoneme probing**: Track linear probe accuracy for IPA phonemes across every layer of the residual stream, mapping the emergence and transformation of phonetic representations
2. **Phoneme subspace analysis**: Use probe weight vectors to test whether per-phoneme subspaces exist, measuring orthogonality, clustering by articulatory features, and testing causal relevance via targeted interventions
3. **Context-dependent pronunciation via API**: Test whether modern LLMs can handle heteronyms correctly using real API calls, probing whether rhyming knowledge is purely lexical or contextual

### Experiment Justification
- **Exp 1 (Layer-wise probing)**: Needed to map WHERE phonetic info lives — is it all in embeddings or does it build through layers?
- **Exp 2 (Subspace geometry)**: Needed to test the specific hypothesis about per-IPA-sound subspaces
- **Exp 3 (Context-dependent rhyming)**: Needed to test whether rhyming is purely in static embeddings or requires contextual processing

## Research Question
Where is rhyming knowledge encoded in language models? Is it solely in embeddings, or does it involve deeper layers? Does each IPA phoneme have a distinct subspace?

## Hypothesis Decomposition
H1: Phonetic information is linearly decodable from token embeddings (replication of Merullo)
H2: Phonetic information quality changes across layers — it emerges, peaks, and potentially transforms
H3: Each IPA phoneme corresponds to a distinct direction in embedding/hidden space
H4: Phoneme directions cluster by articulatory features (place, manner, voicing for consonants; height, backness for vowels)
H5: Context-dependent pronunciation requires processing beyond embeddings (heteronyms)

## Proposed Methodology

### Approach
Use Llama-3.2-1B-Instruct with TransformerLens for mechanistic analysis. Build word→IPA mappings from WikiPron/CMU Dict, train linear probes at each layer, analyze probe weight geometry.

### Experimental Steps

**Experiment 1: Layer-wise phoneme probing**
1. Build vocabulary of single-token English words with IPA transcriptions
2. Create multi-hot IPA phoneme labels for each word
3. Extract embeddings and residual stream activations at each layer
4. Train linear probes (logistic regression) at each layer to predict phonemes
5. Plot probe accuracy vs. layer number

**Experiment 2: Phoneme subspace geometry**
1. Extract probe weight vectors (one per phoneme) from best-performing layer
2. Compute cosine similarity matrix between all phoneme vectors
3. Test clustering by articulatory features using PanPhon
4. Measure orthogonality of phoneme subspaces
5. Visualize with PCA/t-SNE, compare to IPA vowel chart

**Experiment 3: Context-dependent rhyming (API-based)**
1. Curate list of heteronyms (read, lead, wind, tear, etc.)
2. Create sentence pairs providing context for each pronunciation
3. Query LLM API to identify rhyming words in context
4. Measure accuracy on context-dependent vs. context-independent rhyming

### Baselines
- Random probe accuracy (~42% from Merullo et al.)
- Shuffled labels (chance performance for our specific setup)
- PanPhon articulatory distance as ground truth for similarity

### Evaluation Metrics
- Multi-label probe accuracy (F1, precision, recall) per layer
- Cosine similarity structure of probe weight vectors
- Silhouette score for phoneme clustering by articulatory class
- Rhyme identification accuracy for API-based context experiment

### Statistical Analysis Plan
- Bootstrap confidence intervals for probe accuracies
- Permutation test for phoneme clustering significance
- McNemar's test for context-dependent vs independent rhyming

## Expected Outcomes
- H1: Probe accuracy >90% on embeddings (replicating Merullo)
- H2: Accuracy rises in early layers, peaks mid-network, possibly decreases in later layers as semantics dominate
- H3: Phoneme vectors show moderate but not perfect orthogonality
- H4: Clear clustering by articulatory features in probe weight space
- H5: LLMs handle heteronyms better than static embeddings would predict

## Timeline
- Setup + data prep: 15 min
- Exp 1 (layer-wise probing): 45 min
- Exp 2 (subspace geometry): 30 min
- Exp 3 (context-dependent): 20 min
- Analysis + visualization: 30 min
- Documentation: 20 min

## Potential Challenges
- Llama-3.2-1B may be too large for full TransformerLens hooks on all layers → use batching
- Single-token word constraint limits vocabulary size
- WikiPron IPA may not perfectly match model's implicit phonetic knowledge
