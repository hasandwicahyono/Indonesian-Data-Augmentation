# Indonesian Data Augmentation Using Transformers and Similarity Measures

This repository contains the complete source code adaptation of the methodology introduced by **Refai, Abu-Soud, & Abdel-Rahman (2023)** (*"Data Augmentation Using Transformers and Similarity Measures for Improving Arabic Text Classification"*, *IEEE Access*, vol. 11, pp. 132516–132531), specifically re-engineered and optimized for **Bahasa Indonesia** and tailored to four comparable datasets available within this project workspace.

---

## 1. Architectural Adaptation Overview

| Component | Original Paper (Arabic) | Adapted Indonesian Implementation |
| :--- | :--- | :--- |
| **Generative Model** | `aubmindlab/aragpt2-base` | `cahya/gpt2-small-indonesian-522M` *(or `flax-community/gpt2-small-indonesian`)* |
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
├── run_all_benchmarks.py               # Master CLI: Unified benchmark runner, ablation suite & visualizer
├── configs/                            # Modular YAML configuration files
│   ├── datasets.yaml                   # Dataset hyperparameters, imbalance calibrations & training settings
│   ├── models.yaml                     # Model registry for generative backbone candidates
│   ├── preprocessor.yaml               # Slang dictionary, emoji sentiment mappings & emoticons
│   └── benchmarks.yaml                 # Prompts, SOTA baselines & experimental configurations
├── src/                                # Core modular Python package
│   ├── data/                           # Data handling and preprocessing
│   │   ├── preprocessor.py             # Text cleaning, slang normalization, emoji translation
│   │   ├── dataset_manager.py          # Unified loader for SmSA, EmoT, HoASA, MovieOpini
│   │   └── classification.py           # PyTorch Dataset for IndoBERT tokenization & label mapping
│   ├── augmentation/                   # Augmentation algorithms & filtering
│   │   ├── registry.py                 # Generative model registry
│   │   ├── similarity.py               # Euclidean, Cosine, Jaccard, and BLEU-1..4 metrics
│   │   ├── baselines.py                # EDA and backtranslation baseline augmenters
│   │   ├── augmenter.py                # Core generative augmentation & similarity engine
│   │   ├── filtering.py                # Threshold calculation, hallucination purge & dataset export
│   │   └── generative_models.py        # Generative model comparison & benchmarking
│   ├── modeling/                       # Model training and calibration
│   │   ├── calibrator.py               # Imbalance calibrator, logit adjustment & loss functions
│   │   ├── adversarial.py              # Fast Gradient Method (FGM) regularization
│   │   └── evaluator.py                # IndoBERT classifier training and evaluation
│   ├── benchmarks/                     # Benchmark suites and table generators
│   │   ├── cross_validation.py         # 5-fold cross-validation with paired t-tests
│   │   ├── table_generator.py          # Dynamic LaTeX tables generator for IEEE Access & TPAMI
│   │   ├── tpami_suite.py              # TPAMI experimental suite (ECE, margins, t-SNE, etc.)
│   │   └── extended_suite.py           # Extended benchmarks (diversity, ablations, baselines)
│   └── utils/                          # Shared general-purpose utilities
│       ├── config.py                   # Central YAML configuration loader
│       ├── path_utils.py               # Standardized repository paths
│       ├── metrics.py                  # Statistical metrics, Distinct-N, entropy
│       ├── hardware.py                 # Hardware and execution environment detectors
│       └── latex.py                    # LaTeX manuscript compilation utilities
├── notebooks/                          # Interactive exploratory notebooks
├── paper/                              # LaTeX manuscript, tables, and figures
└── data/                               # Raw, augmented, and experimental results
```

---

## 4. The Augmentation & Benchmark Workflow

### Configuration-Driven Architecture
All hyperparameters, model registries, slang normalizations, and dataset specifications are externalized in `configs/*.yaml`:
* `configs/datasets.yaml`: Learning rates, batch sizes, epochs, logit adjustment $\tau$, oversampling factors, label smoothing, and FGM $\epsilon$.
* `configs/models.yaml`: First-Gen (2020-2021) and Contemporary Foundation (2024) causal language models.
* `configs/preprocessor.yaml`: Indonesian slang mapping dictionary and sentiment emoji translations.
* `configs/benchmarks.yaml`: Benchmark prompts, SOTA baseline parameters, and experimental configurations.

### Running Benchmarks & Augmentations
The unified runner `run_all_benchmarks.py` coordinates the complete end-to-end pipeline:

```bash
# Run complete benchmark on all 4 datasets
python run_all_benchmarks.py

# Or run generation phase with a specific generative model
python run_all_benchmarks.py --run_generation --gen_model sailor-0.5b

# Or run 5-fold cross-validation with statistical paired t-tests
python run_all_benchmarks.py --run_cv

# Or generate/update all figures, dynamic LaTeX tables, and compile manuscripts instantly
python run_all_benchmarks.py --plot_only
```

You can also execute individual modular components directly:
```bash
# Benchmark candidate generative backbones
python -m src.augmentation.generative_models

# Run 5-fold cross validation suite
python -m src.benchmarks.cross_validation

# Generate dynamic LaTeX tables and compile IEEE Access & TPAMI manuscripts
python -m src.benchmarks.table_generator
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
