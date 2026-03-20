# Downloaded Datasets

This directory contains datasets for the research project. Large data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: WikiPron (Multilingual Pronunciations)

### Overview
- **Source**: https://github.com/CUNY-CL/wikipron
- **Size**: 3.9M pronunciations across 306 languages
- **Format**: TSV files (word → IPA transcription)
- **License**: Apache 2.0
- **Used by**: Merullo et al. 2025 as pronunciation reference

### Download Instructions
Already cloned to `datasets/wikipron/`. English data at:
```
datasets/wikipron/data/scrape/tsv/eng_latn_us_broad.tsv
```

### Loading
```python
import csv
with open("datasets/wikipron/data/scrape/tsv/eng_latn_us_broad.tsv") as f:
    reader = csv.reader(f, delimiter='\t')
    for word, ipa in reader:
        print(word, ipa)
```

---

## Dataset 2: Oxford 5000 Words

### Overview
- **Source**: https://github.com/tyypgzl/Oxford-5000-words
- **Size**: ~5000 common English words with pronunciation info
- **Format**: JSON
- **Used by**: Merullo et al. 2025 for filtering rhyme task words

### Download Instructions
Already cloned to `datasets/oxford_5000/`.

### Loading
```python
import json
with open("datasets/oxford_5000/full-word.json") as f:
    words = json.load(f)
```

---

## Dataset 3: PWESuite Evaluation Dataset

### Overview
- **Source**: https://huggingface.co/datasets/zouharvi/pwesuite-eval
- **Size**: 1,738,496 entries across 9 languages
- **Format**: HuggingFace Dataset (Arrow)
- **Fields**: token_ort, token_ipa, token_arp, lang, purpose, extra_index
- **License**: Apache 2.0

### Download Instructions
Already downloaded to `datasets/pwesuite_eval/`.

**To re-download:**
```python
from datasets import load_dataset
ds = load_dataset("zouharvi/pwesuite-eval", split="train")
ds.save_to_disk("datasets/pwesuite_eval")
```

### Loading
```python
from datasets import load_from_disk
ds = load_from_disk("datasets/pwesuite_eval")
print(ds[0])  # {'token_ort': 'a', 'token_ipa': 'ə', 'token_arp': 'AH0', ...}
```

---

## Dataset 4: CMU Pronouncing Dictionary

### Overview
- **Source**: https://github.com/cmusphinx/cmudict
- **Size**: 134,000+ English words with ARPAbet transcriptions
- **Format**: Python package (pip install cmudict)
- **License**: Unrestricted use

### Download Instructions
Installed as Python package:
```bash
pip install cmudict
```

### Loading
```python
import cmudict
d = cmudict.dict()
print(d['hello'])  # [['HH', 'AH0', 'L', 'OW1'], ['HH', 'EH0', 'L', 'OW1']]
```

---

## Dataset 5: PanPhon (Articulatory Features)

### Overview
- **Source**: https://github.com/dmort27/panphon
- **Size**: 5000+ IPA segments mapped to 21 articulatory features
- **Format**: Python package (pip install panphon)
- **License**: MIT

### Download Instructions
Installed as Python package:
```bash
pip install panphon
```

### Loading
```python
import panphon
ft = panphon.FeatureTable()
print(ft.word_fts('bat'))  # articulatory features for each phoneme
```

---

## Notes
- WikiPron and Oxford 5000 are cloned as git repos (small enough)
- PWESuite eval is downloaded as HuggingFace dataset
- CMU Dict and PanPhon are installed as pip packages
- All datasets are accessible without authentication
