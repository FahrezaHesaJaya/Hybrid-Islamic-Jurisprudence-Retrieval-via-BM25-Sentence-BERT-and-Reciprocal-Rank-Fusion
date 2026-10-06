# 🕌 Quranica: Hybrid Information Retrieval System for Comparative Islamic Jurisprudence (Fiqh)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Scientific Research & Publication Repository**  
> **Title:** *Relevance Optimization in Everyday Islamic Jurisprudence Retrieval Using a Hybrid Retrieval Architecture Based on Sentence-BERT and Okapi BM25*  
> **Author & Lead Researcher:** Fahreza Hesa Jaya  
> **Affiliation:** Department of Informatics, Faculty of Engineering, Universitas Malikussaleh, Indonesia  
> **Advisors & Co-Authors:** Safwandi, S.T., M.Kom & Nunsina, S.T., M.Kom  
> **Target Journal:** *Jurnal Informatika Mulawarman (JIM)* — Nationally Accredited SINTA 3  

---

## 📌 Executive Summary

**Quranica** is an artificial intelligence-powered legal information retrieval system designed to search and analyze comparative everyday Islamic jurisprudence (*fiqh ubudiyyah*) rulings between Indonesia's two largest Islamic mass organizations: **Nahdlatul Ulama (LBM-PBNU)** and **Muhammadiyah (Majelis Tarjih PP Muhammadiyah)**.

In conventional search engines, retrieving Islamic legal literature faces two major technical bottlenecks:
1. **Term Mismatch in Lexical Retrieval (BM25):** Fails when queries contain colloquial contractions, regional vernacular (*slang*), vowel-dropped chat SMS forms (`wdhu`, `poso`, `drh`, `btl`), or typographical errors.
2. **Semantic Drift in Dense Semantic Retrieval (Sentence-BERT):** Pretrained dense transformers often confuse closely related fiqh terms with opposing legal outcomes (e.g., distinguishing between *madzi* and *mani*, or wudhu invalidation vs. mandatory major ritual baths).

Quranica resolves these issues through a **Hybrid Retrieval Architecture** that fuses sparse lexical signals (**Okapi BM25**) and dense semantic representations (**Multilingual Sentence-BERT**, `paraphrase-multilingual-MiniLM-L12-v2`) via balanced 50:50 score fusion, strengthened by:
- **Dynamic Per-Query Min-Max Score Normalization:** Eliminates baseline score inflation (~0.70) in cosine similarity space.
- **Advanced Colloquial Normalization Engine:** Normalizes dialectal contractions while strictly preserving ~100 common Indonesian function words.
- **Non-Destructive Soft Category Gating:** Prevents cross-domain semantic leakage (e.g., distinguishing blood donation under *Thaharah* vs. *Puasa*) without rigid candidate filtering.
- **Specificity-Aware Subject Bonus:** Protects granular topics from being eclipsed by broader parent categories.
- **Automated Fatwa Consensus Classifier:** Determines legal consensus badges (*Mubah*, *Permissible*, *Invalidating*, *Obligatory*, *Prohibited*) across NU and Muhammadiyah rulings.

---

## 📊 Comprehensive Empirical Benchmarks (Cranfield Evaluation Paradigm)

Evaluated across **500 authentic comparative fatwa documents** under three rigorous testing regimes using international Information Retrieval metrics: **Precision@1 (P@1)**, **Recall@3 (R@3)**, **Recall@5 (R@5)**, and **Mean Reciprocal Rank (MRR)**.

### Metric Formulas
$$\text{Precision@1} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \mathbb{I}(\text{rank}_i = 1)$$
$$\text{Recall@K} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \mathbb{I}(\text{rank}_i \le K), \quad K \in \{3, 5\}$$
$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

---

### Comparative Evaluation Results (500 Test Queries)

| Benchmark Scenario | Pure BM25 (Lexical) | Pure Indo S-BERT (Semantic) | Proposed Hybrid Retrieval | Performance Gain vs. BM25 | Performance Gain vs. S-BERT |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. 500 Out-of-Sample Unseen Queries** | P@1: 73.0%<br>MRR: 0.7913 | P@1: 35.2%<br>MRR: 0.4590 | **P@1: 92.8%<br>R@5: 98.6%<br>MRR: 0.9535** | **+27.12%** | **+163.64%** |
| **2. 500 Colloquial Chat & Typo Queries** | P@1: 31.0%<br>MRR: 0.4539 | P@1: 19.6%<br>MRR: 0.2929 | **P@1: 60.2%<br>R@5: 92.6%<br>MRR: 0.7425** | **+94.19%** | **+207.14%** |
| **3. 500 Inverted Syntax Queries (Ultimate Stress Test)** | P@1: 26.8%<br>MRR: 0.4167 | P@1: 11.8%<br>MRR: 0.1988 | **P@1: 57.4%<br>R@5: 85.4%<br>MRR: 0.6916** | **+114.18%** | **+386.44%** |

### Key Scientific Findings:
1. **Mathematical Synergy:** Fusing sparse and dense signals consistently outperforms standalone models across all queries, validating the hypothesis that lexical and semantic retrieval address mutually exclusive error modes.
2. **Stress-Test Resilience:** Under severe word-order permutations (inverted conversational queries) and extreme chat slang, the proposed Hybrid system successfully captures **85.4% of relevant fatwa documents within the Top-5 positions** with an average query latency of only ~133 ms.

---

## 🛠️ Architecture & Core Components

```
┌────────────────────────────────────────────────────────┐
│               User Query (Text / Slang)                │
└───────────────────────────┬────────────────────────────┘
                            │
               ┌────────────▼───────────┐
               │ Preprocessing & Normal │
               │ Slang Dict + Boundary  │
               └────────────┬───────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
┌───────────▼───────────┐       ┌───────────▼───────────┐
│   Sparse Lexical      │       │     Dense Semantic    │
│    (Okapi BM25)       │       │    (Sentence-BERT)    │
│  Sastrawi Tokenizer   │       │  384-dim Dense Vector │
└───────────┬───────────┘       └───────────┬───────────┘
            │                               │
            └───────────────┬───────────────┘
                            │
               ┌────────────▼───────────┐
               │ Dynamic Min-Max Normal │
               │  Balanced Fusion 50:50 │
               └────────────┬───────────┘
                            │
               ┌────────────▼───────────┐
               │ Soft Category Gating   │
               │ Specificity Bonus      │
               └────────────┬───────────┘
                            │
               ┌────────────▼───────────┐
               │ Ranked Results (Top-K) │
               │ NU & Muhammadiyah Card │
               │ Consensus Legal Badge  │
               └────────────────────────┘
```

---

## 🚀 Quickstart & Local Installation

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/FahrezaHesaJaya/Hybrid-Islamic-Jurisprudence-Retrieval-via-BM25-Sentence-BERT-and-Reciprocal-Rank-Fusion.git
cd Hybrid-Islamic-Jurisprudence-Retrieval-via-BM25-Sentence-BERT-and-Reciprocal-Rank-Fusion
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📂 Repository Structure

```text
├── app.py                             # Main Streamlit Application Controller & UI Orchestration
├── retrieval.py                       # Information Retrieval Core (BM25, S-BERT, Intent Expansion, RRF)
├── classifier.py                      # Context-Aware Fiqh Legal Status Classifier & Executive Synthesis
├── ui_styles.py                       # Custom Islamic Theme Design System & CSS Styling
├── utils.py                           # Utilities (Prayer Times API, Search Logger, Trivia & Recommendations)
├── requirements.txt                   # Production Python Dependencies
├── .streamlit/
│   └── config.toml                    # Clean Google-Style Theme Configurations
├── assets/
│   └── quranica_hero.jpg              # Official Visual Header Banner
├── Data_ubudiyah_final.parquet        # Fast Binary Corpus Representation (500 Fatwas)
├── Data_ubudiyah_final.csv            # Structured Master Dataset (NU & Muhammadiyah)
├── bm25_model.pkl                     # Precomputed Okapi BM25 Index Cache
├── corpus_emb.npy                     # Precomputed 384-dim Sentence-BERT Embedding Matrix
├── .gitignore                         # Git Production Exclusion Rules
└── README.md                          # Comprehensive Academic Documentation
```

---

## ⚖️ Comparative Fatwa Scope

The dataset covers **500 verified rulings** across five essential daily worship (*ubudiyyah*) domains:
1. **Thaharah (Purification):** 100 rulings (Ablution, Tayammum, Impurities, Janabah, Medical fluids).
2. **Shalat (Prayer):** 200 rulings (Obligatory prayers, Congregarional etiquette, Travel rukhsah, Forgetfulness prostrations).
3. **Puasa (Fasting):** 100 rulings (Ramadan requirements, Involuntary swallowing, Medical injections, Inhalers).
4. **Zakat (Almsgiving):** 80 rulings (Zakat Fitrah, Zakat Mal, Digital payments / QRIS, Nisab & Haul).
5. **Haji & Umrah (Pilgrimage):** 20 rulings (Ihram prohibitions, Manasik violations, Dam compensations).

---

## 📜 Citation & Academic Attribution

If you utilize this repository, codebase, or dataset in your academic research, please cite as follows:

```bibtex
@misc{jaya2026quranica,
  author       = {Fahreza Hesa Jaya and Safwandi and Nunsina},
  title        = {Relevance Optimization in Everyday Islamic Jurisprudence Retrieval Using a Hybrid Retrieval Architecture Based on Sentence-BERT and Okapi BM25},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/FahrezaHesaJaya/Hybrid-Islamic-Jurisprudence-Retrieval-via-BM25-Sentence-BERT-and-Reciprocal-Rank-Fusion}},
  institution  = {Department of Informatics, Universitas Malikussaleh}
}
```

---

## 👨‍💻 Author & Contact

**Fahreza Hesa Jaya**  
Department of Informatics, Faculty of Engineering, Universitas Malikussaleh  
Email: [fahrezzahesajaya@gmail.com](mailto:fahrezzahesajaya@gmail.com)  
GitHub: [https://github.com/FahrezaHesaJaya](https://github.com/FahrezaHesaJaya)  
License: [MIT License](LICENSE)
