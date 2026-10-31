# Indonesian Data Augmentation Using Transformers and Similarity Measures

This repository contains the complete source code adaptation of the methodology introduced by **Refai, Abu-Soud, & Abdel-Rahman (2023)** (*"Data Augmentation Using Transformers and Similarity Measures for Improving Arabic Text Classification"*, *IEEE Access*, vol. 11, pp. 132516–132531), specifically re-engineered and optimized for **Bahasa Indonesia** and tailored to four comparable datasets available within this project workspace.

---

## 1. Architectural Adaptation Overview

| Component | Original Paper (Arabic) | Adapted Indonesian Implementation |
| :--- | :--- | :--- |
| **Generative Model** | `aubmindlab/aragpt2-base` | `cahya/gpt2-small-indonesian-522M` *(or `flax-community/indonesian-gpt2`)* |
| **Sentence Embedding** | `distiluse-base-multilingual-cased` | `sentence-transformers/distiluse-base-multilingual-cased-v2` |
| **Downstream Classifier** | `aubmindlab/bert-base-arabertv02-twitter` | `indobenchmark/indobert-base-p1` *(IndoBERT)* |
| **Text Preprocessing** | `ArabertPreprocessor` (Farasa, Arabic diacritics, Tatweel) | `IndonesianPreprocessor` (Slang normalizer, repeated char reduction, mentions, URLs) |
| **Evaluation Framework** | 5-Fold Cross Validation / Paired t-test | Stratified Train/Test & 5-Fold CV / Paired t-test |

---

## 2. Selected Comparable Datasets

The 4 datasets accommodate the exact experimental diversity of the original research:

1. **SmSA (`smsa_doc-sentiment-prosa`)** $\leftrightarrow$ **AraSarcasm-v1**
   - **Scale**: Large (~11,000 train / 12,760 total records)
   - **Domain**: Consumer reviews and document feedback (Prosa.ai)
   - **Classes**: 3 classes (`positive`, `negative`, `neutral`)
   - **Minority Classes to Augment**: `negative` (31.2%), `neutral` (10.4%)
   - **Source**: `data/raw/smsa_doc-sentiment-prosa/`

2. **EmoT (`emot_emotion-twitter`)** $\leftrightarrow$ **ASTD**
   - **Scale**: Medium (~3,521 train / 4,401 total records)
   - **Domain**: Social Media (Twitter tweets with informal Indonesian slang)
   - **Classes**: 5 emotion categories (`anger`, `happy`, `sadness`, `fear`, `love`)
   - **Minority Classes to Augment**: `fear` (14.7%), `love` (14.5%)
   - **Source**: `data/raw/emot_emotion-twitter/`

3. **HoASA (`hoasa_absa-airy`)** $\leftrightarrow$ **ATT**
   - **Scale**: Medium (~2,283 train / 2,854 total records)
   - **Domain**: Travel, Lodging & Hospitality reviews (Airy Rooms hotel customer feedback)
   - **Classes**: Aspect-based review polarity (`positive`, `negative`, `neutral`)
   - **Minority Classes to Augment**: `neg` (1.3%), `pos` (10.1%) (where `neut` is the majority at 88.6%)
   - **Source**: `data/raw/hoasa_absa-airy/`

4. **MovieOpini (`dataset_tweet_sentiment_opini_film.csv`)** $\leftrightarrow$ **MOVIE**
   - **Scale**: Small (~200 records)
   - **Domain**: Entertainment & Film review tweets ("Opini Film")
   - **Classes**: 2 classes (`positive`, `negative`)
   - **Minority Classes to Augment**: `negative`, `positive` (low-resource augmentation)
   - **Source**: `data/raw/dataset_tweet_sentiment_opini_film.csv`

---

## 3. Directory Structure

```text
Indonesian-Data-Augmentation/
├── README.md                           # Documentation and usage guide
├── requirements.txt                    # Project dependencies
├── IndonesianPreprocessor.py           # Indonesian text cleaning, elongation, and slang normalization
├── ClassificationDataset.py            # PyTorch Dataset for IndoBERT tokenization & label mapping
├── DatasetsManager.py                  # Dataset loader for SmSA, EmoT, HoASA, and MovieOpini
├── SimilarityMeasures.py               # Euclidean, Cosine, Jaccard, and BLEU-1..4 metrics
├── Augmenter.py                        # Core generator & similarity scoring engine
├── run_generation.py                   # CLI: Phase 1 (Text Generation & Similarity scoring)
├── run_augmentation_filter.py          # CLI: Phase 2 (Similarity threshold filtering & dataset export)
├── run_classification_eval.py          # CLI: Phase 3 (IndoBERT fine-tuning & paired t-test evaluation)
├── run_all_benchmarks.py               # CLI: Unified benchmark runner & visualizer for SmSA, MovieOpini, HoASA, EmoT
├── notebooks/
│   ├── SmSA_Indonesian_Augmentation_and_Classification.ipynb
│   ├── EmoT_Indonesian_Augmentation_and_Classification.ipynb
│   ├── HoASA_Indonesian_Augmentation_and_Classification.ipynb
│   └── MovieOpini_Indonesian_Augmentation_and_Classification.ipynb
└── data/
    ├── raw/                            # Intermediate generated samples with similarity scores
    ├── augmented/
    │   ├── all_text/                   # Original + Generated text datasets (ECU, COS, JAC, BLEU)
    │   └── new_text/                   # Generated text datasets (ECU, COS, JAC, BLEU)
    └── results/                        # Final classification metrics, F1 comparisons, and paired t-tests
```

---

## 4. The 3-Phase Augmentation Workflow

### Phase 1: Text Generation & Similarity Scoring
Generate continuation sequences for dataset records using an Indonesian causal language model and compute embeddings:
```bash
python run_generation.py --dataset SmSA --sample_limit 500
```
Outputs intermediate data with `ecu_similarity`, `cos_similarity`, `jacc_similarity`, and `bleu_sim_1` to `data/raw/SmSA-Augmented-raw.csv`.

### Phase 2: Similarity Threshold Filtering & Class Augmentation
Calculate average metric thresholds and augment only under-represented minority classes:
```bash
python run_augmentation_filter.py \
  --input_file data/raw/SmSA-Augmented-raw.csv \
  --dataset_name SmSA \
  --minority_classes negative neutral
```
Produces the 4 final augmented datasets:
- `SmSA-Augmented-ECU-ALL-Text-Final.csv`
- `SmSA-Augmented-COS-ALL-Text-Final.csv`
- `SmSA-Augmented-JAC-ALL-Text-Final.csv`
- `SmSA-Augmented-BLE-ALL-Text-Final.csv`

### Phase 3: IndoBERT Classifier Fine-Tuning & Evaluation
Train and evaluate `indobenchmark/indobert-base-p1` on baseline vs. all 4 similarity-filtered datasets:
```bash
python run_classification_eval.py
```
Outputs Accuracy, Macro-Precision, Macro-Recall, Macro-F1 improvement, and paired t-test results to `data/results/`.

### Phase 4: Unified Benchmark Runner & Visualization Suite
To run all investigations across **SmSA**, **MovieOpini**, **HoASA**, and **EmoT**, and generate publication-ready performance comparison plots and confusion matrices:
```bash
# Run complete benchmark on all 4 datasets
python run_all_benchmarks.py

# Or run specific dataset(s)
python run_all_benchmarks.py --datasets SmSA EmoT

# Or generate/update all visualization and confusion matrix plots instantly from existing results
python run_all_benchmarks.py --plot_only
```
Generated artifacts in `data/results/figures/`:
* `{dataset}_metrics_comparison.png` & `.pdf`: 4-metric bar chart (Accuracy, Precision, Recall, Macro-F1) comparing Baseline vs. ECU vs. COS vs. JAC vs. BLEU.
* `{dataset}_confusion_matrices.png` & `.pdf`: Side-by-side Confusion Matrices (Baseline vs. Best Augmented) with raw counts and percentages.
* `all_datasets_benchmark_comparison.png` & `.pdf`: 4-panel dashboard summarizing performance across all datasets.

---

## 5. Jupyter Notebooks

For interactive experimentation, open any of the tailored notebooks in `notebooks/`:
- **SmSA**: [`notebooks/SmSA_Indonesian_Augmentation_and_Classification.ipynb`](notebooks/SmSA_Indonesian_Augmentation_and_Classification.ipynb)
- **EmoT**: [`notebooks/EmoT_Indonesian_Augmentation_and_Classification.ipynb`](notebooks/EmoT_Indonesian_Augmentation_and_Classification.ipynb)
- **HoASA**: [`notebooks/HoASA_Indonesian_Augmentation_and_Classification.ipynb`](notebooks/HoASA_Indonesian_Augmentation_and_Classification.ipynb)
- **MovieOpini**: [`notebooks/MovieOpini_Indonesian_Augmentation_and_Classification.ipynb`](notebooks/MovieOpini_Indonesian_Augmentation_and_Classification.ipynb)
