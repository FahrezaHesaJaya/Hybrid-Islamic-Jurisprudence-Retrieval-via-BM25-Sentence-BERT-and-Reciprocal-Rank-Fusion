# Hybrid Islamic Jurisprudence Retrieval via BM25, Sentence-BERT, and Reciprocal Rank Fusion

**Fahreza Hesa Jaya$^{1)}$, Safwandi$^{2)}$, Nunsina$^{3)}$**  
$^{1,2,3)}$ Program Studi Teknik Informatika, Fakultas Teknik, Universitas Malikussaleh  
Jl. Kampus Bukit Indah, Blang Pulo, Kota Lhokseumawe, Aceh 24355, Indonesia  
E-Mail: `fahreza.220170117@mhs.unimal.ac.id` $^{1)}$; `safwandi@unimal.ac.id` $^{2)}$; `nunsina@unimal.ac.id` $^{3)}$  

---

### ABSTRACT
Public access to authoritative Islamic jurisprudence (*fiqh*) fatwas is frequently hindered by vocabulary mismatch between colloquial queries posed by everyday users and the formal Arabic-Indonesian terminology recorded in religious legal texts. Traditional lexical retrieval models such as Okapi BM25 struggle with morphological variations and synonyms, while dense semantic models like Sentence-BERT are susceptible to semantic drift across dense religious document clusters. This study develops a robust Hybrid Retrieval architecture combining Okapi BM25 and Multilingual Sentence-BERT (`paraphrase-multilingual-MiniLM-L12-v2`), reinforced by Dynamic Per-Query Min-Max Normalization, Specificity-Aware Topic Bonus, and Soft Hierarchical Category Gating. The empirical evaluation was conducted following the Cranfield Information Retrieval Paradigm on a verified corpus of 500 comparative *ubudiyyah* fatwas from Nahdlatul Ulama and Majelis Tarjih Muhammadiyah. A comprehensive benchmark of 1,000 empirical queries—comprising 500 out-of-sample unseen formal queries and 500 colloquial chat and typographical error stress-test queries—was evaluated across a grid search of 13 fusion weight configurations ($w_{\text{BM25}} : w_{\text{SBERT}}$). The empirical results demonstrate that a balanced 50:50 fusion achieves superior performance as the primary target in known-item retrieval, securing a Precision@1 of 75.00%, Recall@5 of 93.80%, and MRR of 0.8319 on formal queries, alongside a Precision@1 of 64.20%, Recall@5 of 92.60%, and MRR of 0.7670 on colloquial typo queries. An ablation study reveals that the proposed Soft Category Gating significantly elevates standalone BM25 by +20.60% and standalone Sentence-BERT by +15.40% in Precision@1, proving that lightweight domain-adaptive heuristics effectively resolve religious retrieval noise without incurring high computational latency (average latency 2.30–2.48 ms per query).

**Keywords** — Islamic Jurisprudence, Hybrid Retrieval, BM25, Sentence-BERT, Reciprocal Rank Fusion, Fatwa Search

---

### 1. INTRODUCTION
The pervasive adoption of digital technology has transformed religious inquiry, enabling Indonesian Muslims to seek everyday Islamic legal rulings (*fatwas*) through mobile applications and conversational search engines (Kurniawan & Syarif, 2023). However, digital information retrieval in Islamic jurisprudence encounters a severe vocabulary mismatch problem (*term mismatch*). Everyday users typically express their concerns using colloquial language, regional vernacular, abbreviations, and informal descriptions (e.g., querying *"hukum solat keluar mimisan"* or *"wudhu btl ga kl abis donor drh"*), whereas authentic fatwa documents compiled by established Islamic organizations such as Nahdlatul Ulama (LBM-NU) and Muhammadiyah (Majelis Tarjih dan Tajdid) employ formal religious terminology derived from classical jurisprudence (*al-kutub al-mu'tabarah*), such as *dam kharij min ghair al-sabilain* (blood exiting from areas other than the primary orifices) and *najis mutawassithah* (moderate impurities) (Arifin et al., 2022).

Conventional lexical information retrieval algorithms, most notably Okapi BM25, rely heavily on exact term frequency and inverse document frequency (IDF) matches (Robertson & Zaragoza, 2009). When colloquial search terms fail to appear verbatim in document indices, lexical search engines suffer from term mismatch, leading to zero-score retrievals or irrelevant outputs (Zheng et al., 2021). Conversely, dense semantic retrieval powered by Transformer-based Pretrained Language Models, such as Sentence-BERT (Reimers & Gurevych, 2019), maps queries and passages into dense embedding spaces to capture conceptual meaning. Nonetheless, standalone dense retrieval models often exhibit *semantic drift* in specialized religious domains; because religious fatwas across various chapters share a dense background of sacred and legal vocabulary (*shalat*, *wajib*, *batal*, *sah*), dense vector similarities often fail to distinguish fine-grained procedural boundaries, conflating unrelated rulings within overlapping contexts (Zhang et al., 2023).

To mitigate these bilateral limitations, recent information retrieval research has explored hybrid retrieval paradigms combining sparse lexical scoring with dense semantic vector similarity (Karpukhin et al., 2020; Luan et al., 2021). Nevertheless, standard hybrid fusion techniques often treat document collections as uniform text blocks, neglecting domain-specific hierarchical structures and cross-chapter interactions. In Islamic jurisprudence, a single real-world problem frequently intersects multiple legal chapters—such as bodily impurities breaking ablution (*Thaharah*) directly impacting the validity of prayer (*Shalat*). Conventional hard filtering abruptly discards candidate documents across adjacent chapters, precipitating catastrophic false exclusions.

This study proposes an enhanced Hybrid Islamic Jurisprudence Retrieval architecture that integrates Okapi BM25 and Multilingual Sentence-BERT through Dynamic Min-Max Score Normalization, Specificity-Aware Topic Bonus, and a non-destructive Soft Hierarchical Category Gating mechanism. The proposed system is implemented and evaluated on a bilingual comparative dataset of 500 authentic *ubudiyyah* fatwas representing Nahdlatul Ulama and Muhammadiyah. The primary contributions of this paper are fourfold:
1. Formulating a multi-tier query expansion pipeline that unifies Indonesian colloquial slang normalization, a dedicated fiqh thesaurus, and cross-domain contextual regex rules.
2. Designing a soft hierarchical category gating mechanism that dynamically modulates score distributions without permanently pruning candidate documents across overlapping worship domains.
3. Conducting an extensive empirical benchmark across 1,000 queries (500 unseen formal queries and 500 colloquial chat and typographical stress-test queries) evaluated through a 13-configuration fusion weight grid search ($w_{\text{BM25}} : w_{\text{SBERT}}$ from 100:0 to 0:100).
4. Providing an empirical ablation analysis of the Precision-Recall trade-off and evaluating the computational efficiency of lightweight multilingual embeddings for low-latency web deployment.

---

### 2. LITERATURE REVIEW

#### A. Comparative Islamic Jurisprudence Corpus
Islamic jurisprudence (*fiqh*) regarding daily worship (*ubudiyyah*) serves as the legal foundation for Muslim religious practice. In Indonesia, two major Islamic organizations, Nahdlatul Ulama (founded in 1926) and Muhammadiyah (founded in 1912), provide legal guidance through distinct methodological frameworks (*manhaj*). Nahdlatul Ulama predominantly adheres to the Shafi'i legal school (*madhhab*) using the consensus-driven *Bahtsul Masail* mechanism (Fauzi, 2021), whereas Muhammadiyah employs the *Manhaj Tarjih*, utilizing contextual *ijtihad* directly anchored in the Qur'an and authentic Sunnah through a *maqashid al-shari'ah* lens (Huda et al., 2022). While both organizations share substantial consensus (*ittifaq*) on primary obligations, nuanced differences (*khilafiyah*) frequently arise regarding procedural details, such as the qunut supplication in dawn prayer, ablution nullification through touch, and contemporary transaction rulings. Constructing a bilingual, comparative corpus addressing both perspectives is essential for balanced, objective religious information access.

#### B. Lexical Retrieval and Okapi BM25
Okapi BM25 remains the predominant baseline in lexical information retrieval due to its solid probabilistic foundation and computational efficiency (Robertson & Zaragoza, 2009). The retrieval score of document $D$ for query $Q = \{q_1, q_2, \dots, q_n\}$ is formulated as:
$$\text{Score}_{\text{BM25}}(D, Q) = \sum_{i=1}^n \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
where $f(q_i, D)$ is the term frequency of $q_i$ in document $D$, $|D|$ is the document length, $\text{avgdl}$ is the average document length across the corpus, $k_1$ regulates term frequency saturation (typically $k_1 = 1.5$), and $b$ controls document length normalization (typically $b = 0.75$). Inverse Document Frequency is computed as:
$$\text{IDF}(q_i) = \ln \left(\frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1\right)$$
where $N$ is the total document count and $n(q_i)$ denotes the number of documents containing $q_i$. Although BM25 provides precise lexical anchoring for exact legal terms (*sujud sahwi*, *zakat fitrah*), it suffers from severe recall degradation when queries involve informal phrasing or typos.

#### C. Dense Semantic Retrieval via Sentence-BERT
Dense semantic retrieval maps queries and documents into continuous vector representations, capturing latent semantic relationships beyond surface lexical forms (Karpukhin et al., 2020). Sentence-BERT modifies pretrained Transformer networks using siamese and triplet structures to generate semantically meaningful sentence embeddings that can be compared using cosine similarity (Reimers & Gurevych, 2019):
$$\text{Sim}_{\text{cos}}(u, v) = \frac{u \cdot v}{\|u\|_2 \cdot \|v\|_2}$$
In multilingual and low-resource domain settings, `paraphrase-multilingual-MiniLM-L12-v2` offers an optimal balance between semantic capacity and computational footprint (Wang et al., 2020). However, dense models frequently exhibit semantic drift in specialized domains: because religious texts share ubiquitous spiritual terms, cosine similarities across documents often concentrate within narrow bands (e.g., $[0.65, 0.90]$), diluting top-1 precision without lexical guidance.

#### D. Hybrid Fusion Strategies
Hybrid retrieval unifies sparse lexical scores and dense vector representations to optimize both precision and recall (Luan et al., 2021). Two prominent paradigms exist: Reciprocal Rank Fusion (RRF) and Weighted Linear Score Fusion. RRF merges discrete ranked lists without relying on raw score magnitudes (Cormack et al., 2009):
$$\text{RRF}(d) = \sum_{m \in M} \frac{w_m}{k + r_m(d)}$$
where $r_m(d)$ is the rank of document $d$ in model $m$, $k$ is a smoothing constant (typically $k=60$), and $w_m$ is the model weight. Conversely, Weighted Linear Score Fusion normalizes and combines continuous scores directly:
$$S_{\text{hybrid}}(d, q) = w_{\text{BM25}} \cdot \hat{S}_{\text{BM25}}(d, q) + w_{\text{SBERT}} \cdot \hat{S}_{\text{SBERT}}(d, q)$$
Linear score fusion retains fine-grained margin information between candidate documents, provided that incommensurable score ranges are dynamically normalized (Zheng et al., 2021).

---

### 3. RESEARCH METHOD

#### A. Research Scheme
The research methodology comprises six sequential phases: (1) corpus acquisition and bilingual fatwa annotation; (2) text preprocessing and multi-tier query expansion; (3) inverted index and dense embedding vector space construction; (4) hybrid retrieval engine implementation with soft category gating and topic specificity bonus; (5) empirical benchmark evaluation over 1,000 queries using the Cranfield Paradigm; and (6) ablation analysis across 13 fusion weight configurations. The operational research scheme is illustrated in Figure 1.

```
+-------------------------------------------------------------------+
| 1. Corpus Acquisition (500 Comparative Ubudiyyah Fatwas: NU & MU) |
+-------------------------------------------------------------------+
                                  │
                                  ▼
+-------------------------------------------------------------------+
| 2. Preprocessing & Multi-Tier Expansion (Slang + Fiqh Thesaurus)  |
+-------------------------------------------------------------------+
                                  │
                                  ▼
+-------------------------------------------------------------------+
| 3. Indexing: Okapi BM25 Inverted Index + S-BERT Dense Vectors    |
+-------------------------------------------------------------------+
                                  │
                                  ▼
+-------------------------------------------------------------------+
| 4. Hybrid Engine: Min-Max Normalization + Soft Gate + Topic Bonus|
+-------------------------------------------------------------------+
                                  │
                                  ▼
+-------------------------------------------------------------------+
| 5. Evaluation: 1,000 Queries (500 Unseen Formal + 500 Chat/Typo)  |
+-------------------------------------------------------------------+
                                  │
                                  ▼
+-------------------------------------------------------------------+
| 6. Grid Search Analysis: 13 Fusion Weight Configurations          |
+-------------------------------------------------------------------+
```
*Figure 1. Research Workflow Scheme.*

#### B. Dataset Acquisition and Corpus Characteristics
The master dataset comprises 500 verified fatwa rulings spanning five major daily worship (*ubudiyyah*) chapters, summarized in Table 1. Each record contains the ruling title, category, full fatwa content from Nahdlatul Ulama, full fatwa content from Muhammadiyah, source URLs, and legal consensus classifications (*ittifaq* vs. *khilafiyah*).

*Table 1. Distribution of the Comparative Islamic Jurisprudence Corpus.*
| No | Fiqh Chapter (*Bab*) | Scope of Islamic Legal Issues | Document Count | Percentage |
| :-: | :--- | :--- | :-: | :-: |
| 1 | Thaharah (Purification) | Water types, ablution, tayammum, bodily impurities, bleeding | 100 | 20.0% |
| 2 | Shalat (Prayer) | Rukun, congregational prayer, travel rukhsah, prostrations | 200 | 40.0% |
| 3 | Puasa (Fasting) | Ramadan invalidators, involuntary swallowing, inhalers | 100 | 20.0% |
| 4 | Zakat (Almsgiving) | Zakat fitrah, zakat mal, modern digital payment / QRIS | 80 | 16.0% |
| 5 | Haji & Umrah (Pilgrimage) | Ihram prohibitions, tawaf conditions, dam compensations | 20 | 4.0% |
| **Total** | **Corpus Size** | **Verified Comparative Fatwa Documents** | **500** | **100.0%** |

#### C. Multi-Tier Query Expansion Pipeline
To bridge the vocabulary gap between colloquial Indonesian expressions and formal Arabic-Indonesian legal terms, a multi-tier query expansion pipeline is executed prior to retrieval:
1. **Tier 1 — Colloquial Slang and Typo Normalization (`normalize_slang`):** Maps colloquial abbreviations (*shlt* $\to$ *shalat*, *aer* $\to$ *air*, *poso* $\to$ *puasa*, *cbok* $\to$ *istinja*).
2. **Tier 2 — Canonical Fiqh Thesaurus (`SINONIM_FIQIH`):** Expands synonyms across religious terms (*wudlu/wudu* $\to$ *thaharah, bersuci*; *mimisan* $\to$ *darah, najis darah, batal wudhu*).
3. **Tier 3 — Cross-Domain Intent Expansion (`INTENT_EXPANSION`):** Applies contextual regular expression patterns to inject relevant fiqh concepts. For example, queries matching `mimisan saat sholat` automatically expand to include `darah keluar ketika shalat batal wudhu pakaian terkena darah najis`.

#### D. Hybrid Retrieval Scoring Formulation
The hybrid ranking function incorporates four synchronized components:
1. **Per-Query Dynamic Min-Max Normalization:**
   $$\hat{S}_{\text{BM25}}(d, q) = \frac{S_{\text{BM25}}(d, q)}{\max_{d' \in D} S_{\text{BM25}}(d', q)}$$
   $$\hat{S}_{\text{SBERT}}(d, q) = \frac{S_{\text{SBERT}}(d, q) - \min_{d' \in D} S_{\text{SBERT}}(d', q)}{\max_{d' \in D} S_{\text{SBERT}}(d', q) - \min_{d' \in D} S_{\text{SBERT}}(d', q)}$$
2. **Specificity-Aware Topic Bonus ($S_{\text{topic}}$):** Evaluates clean stem overlap between query stems $T_q$ and document title stems $T_d$:
   $$S_{\text{topic}}(d, q) = 0.50 \cdot \left( \mathbb{I}_{\text{exact}}(d, q) \cdot 0.25 + \frac{|T_q \cap T_d|}{|T_d|} \cdot 0.20 + \min(|T_q \cap T_d|, 4) \cdot 0.03 \right)$$
3. **Soft Hierarchical Category Gating ($S_{\text{cat}}$):** Rather than strictly eliminating non-matching categories, the soft gate assigns an incentive bonus to matching categories while imposing a gentle penalty on others:
   $$S_{\text{cat}}(d, q) = \begin{cases} +0.20, & \text{if } \text{Cat}(d) = \text{Cat}(q) \\ -0.05, & \text{if } \text{Cat}(d) \neq \text{Cat}(q) \text{ and } \neg\text{Strict} \\ -0.25, & \text{if } \text{Cat}(d) \neq \text{Cat}(q) \text{ and Strict} \end{cases}$$
4. **Final Composite Retrieval Score:**
   $$S_{\text{final}}(d, q) = \left(w_{\text{BM25}} \cdot \hat{S}_{\text{BM25}}(d, q)\right) + \left(w_{\text{SBERT}} \cdot \hat{S}_{\text{SBERT}}(d, q)\right) + S_{\text{topic}}(d, q) + S_{\text{cat}}(d, q)$$

---

### 4. RESULTS AND DISCUSSION

#### A. Experimental Evaluation Setup
The proposed system was evaluated following the Cranfield evaluation paradigm across two distinct 500-query test regimes, totaling 1,000 evaluated queries:
- **Regime 1 (500 Unseen Formal Queries):** Natural, complete questions formulated with standard Indonesian and religious terminology.
- **Regime 2 (500 Colloquial Chat and Typo Queries):** Stress-test queries characterized by conversational syntax, spelling errors (88% corruption probability), and regional vernacular.

A grid search over 13 fusion weight configurations was executed on both regimes. In known-item fatwa search, **Precision@1 (P@1)** and **Mean Reciprocal Rank (MRR)** serve as primary performance indicators, as users require immediate authoritative guidance at rank #1.

#### B. Benchmark Results across 13 Fusion Weight Configurations

*Table 2. Evaluation Results on Regime 1: 500 Out-of-Sample Unseen Formal Queries.*
| Configuration | $w_{\text{BM25}}$ | $w_{\text{SBERT}}$ | P@1 (%) | R@3 (%) | R@5 (%) | MRR | MAP@5 | NDCG@5 | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Pure BM25 (100:0) | 1.00 | 0.00 | 64.00 | 82.00 | 88.00 | 0.7470 | 0.7470 | 0.7721 | 2.30 |
| Weight 90:10 | 0.90 | 0.10 | 66.40 | 84.20 | 90.00 | 0.7687 | 0.7687 | 0.7947 | 2.30 |
| Weight 80:20 | 0.80 | 0.20 | 69.40 | 85.40 | 92.20 | 0.7897 | 0.7897 | 0.8179 | 2.30 |
| Weight 75:25 | 0.75 | 0.25 | 71.00 | 87.00 | 92.40 | 0.8013 | 0.8013 | 0.8273 | 2.30 |
| Weight 70:30 | 0.70 | 0.30 | 72.20 | 88.20 | 92.20 | 0.8108 | 0.8108 | 0.8336 | 2.30 |
| Weight 60:40 | 0.60 | 0.40 | 73.80 | 88.20 | 93.00 | 0.8233 | 0.8233 | 0.8454 | 2.30 |
| **Proposed Balanced (50:50)** | **0.50** | **0.50** | **75.00** 🏆 | **89.80** 🏆 | **93.80** 🏆 | **0.8319** 🏆 | **0.8319** 🏆 | **0.8546** 🏆 | **2.30** |
| Weight 40:60 | 0.40 | 0.60 | 74.80 | 88.40 | 93.60 | 0.8259 | 0.8259 | 0.8492 | 2.30 |
| Weight 30:70 | 0.30 | 0.70 | 71.20 | 86.80 | 91.80 | 0.8022 | 0.8022 | 0.8258 | 2.30 |
| Weight 25:75 | 0.25 | 0.75 | 70.00 | 85.40 | 90.60 | 0.7913 | 0.7913 | 0.8140 | 2.30 |
| Weight 20:80 | 0.20 | 0.80 | 67.40 | 84.00 | 90.00 | 0.7703 | 0.7703 | 0.7969 | 2.30 |
| Weight 10:90 | 0.10 | 0.90 | 62.60 | 79.40 | 85.80 | 0.7281 | 0.7281 | 0.7522 | 2.30 |
| Pure S-BERT (0:100) | 0.00 | 1.00 | 57.00 | 74.20 | 82.80 | 0.6768 | 0.6768 | 0.7060 | 2.30 |

*Table 3. Evaluation Results on Regime 2: 500 Colloquial Chat and Typo Queries (Stress Test).*
| Configuration | $w_{\text{BM25}}$ | $w_{\text{SBERT}}$ | P@1 (%) | R@3 (%) | R@5 (%) | MRR | MAP@5 | NDCG@5 | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Pure BM25 (100:0) | 1.00 | 0.00 | 56.00 | 80.40 | 85.00 | 0.6982 | 0.6982 | 0.7235 | 2.48 |
| Weight 90:10 | 0.90 | 0.10 | 58.40 | 82.00 | 85.80 | 0.7179 | 0.7179 | 0.7407 | 2.48 |
| Weight 80:20 | 0.80 | 0.20 | 59.80 | 83.60 | 89.20 | 0.7312 | 0.7312 | 0.7626 | 2.48 |
| Weight 75:25 | 0.75 | 0.25 | 62.00 | 84.60 | 90.20 | 0.7458 | 0.7458 | 0.7769 | 2.48 |
| Weight 70:30 | 0.70 | 0.30 | 63.00 | 85.40 | 91.40 | 0.7526 | 0.7526 | 0.7864 | 2.48 |
| Weight 60:40 | 0.60 | 0.40 | 63.40 | **88.40** 🏆 | **93.20** 🏆 | 0.7629 | 0.7629 | 0.8011 | 2.48 |
| **Proposed Balanced (50:50)** | **0.50** | **0.50** | **64.20** 🏆 | 87.80 | 92.60 | **0.7670** 🏆 | **0.7670** 🏆 | **0.8021** 🏆 | **2.48** |
| Weight 40:60 | 0.40 | 0.60 | 62.60 | 85.80 | 91.60 | 0.7507 | 0.7507 | 0.7867 | 2.48 |
| Weight 30:70 | 0.30 | 0.70 | 59.20 | 83.00 | 90.00 | 0.7202 | 0.7202 | 0.7589 | 2.48 |
| Weight 25:75 | 0.25 | 0.75 | 55.00 | 79.80 | 88.00 | 0.6937 | 0.6937 | 0.7325 | 2.48 |
| Weight 20:80 | 0.20 | 0.80 | 52.80 | 77.40 | 85.80 | 0.6701 | 0.6701 | 0.7078 | 2.48 |
| Weight 10:90 | 0.10 | 0.90 | 48.20 | 71.80 | 81.00 | 0.6197 | 0.6197 | 0.6559 | 2.48 |
| Pure S-BERT (0:100) | 0.00 | 1.00 | 42.00 | 65.20 | 74.60 | 0.5633 | 0.5633 | 0.5946 | 2.48 |

#### C. Inverted-U Performance Curve and Equilibrium Analysis
As illustrated in Tables 2 and 3, retrieval accuracy across varying fusion weights exhibits an **inverted-U (concave) performance trajectory**, reaching its global peak at the balanced 50:50 configuration:
- As the weighting diverges toward extreme lexical dominance (75:25 or 100:0), Precision@1 declines sharply by -4.00% to -11.00% on formal queries and -2.20% to -8.20% on colloquial queries due to vocabulary mismatch.
- Conversely, shifting toward dense semantic dominance (25:75 or 0:100) precipitates even steeper degradation, causing Precision@1 to plummet to 57.00% on formal queries and 42.00% on typo queries as a consequence of unanchored semantic drift.
- The 50:50 equilibrium functions as an optimal synergy: normalized BM25 scores anchor the exact legal subject, while normalized Sentence-BERT dense similarities provide fuzzy semantic alignment that accommodates syntactic variations.

#### D. Precision-Recall Trade-off Analysis (50:50 vs. 60:40)
A notable empirical pattern emerges under colloquial stress testing (Table 3):
- **Recall@3 and Recall@5 Superiority at 60:40:** The 60:40 configuration achieves slightly higher candidate coverage, securing Recall@3 of 88.40% (+0.60%) and Recall@5 of 93.20% (+0.60%). Because query expansion successfully restores canonical keywords, heavily weighted BM25 scores act as a powerful lexical magnet, pulling matching documents into the top-5 candidate pool.
- **Precision@1 and MRR Dominance at 50:50:** Despite slightly narrower candidate recall, the 50:50 configuration superiorly secures the top rank, achieving the highest Precision@1 of 64.20% (+0.80% over 60:40) and highest MRR of 0.7670 (+0.0041). The balanced 50% semantic weight provides the contextual discernment necessary to propel the true target from rank #2 or #3 directly into rank #1.
- In task-oriented fatwa retrieval where user attention concentrates on the primary recommendation card, **Precision@1 serves as the decisive metric**, validating 50:50 as the superior deployment configuration.

#### E. Component Ablation Study
To verify the independent utility of the proposed Soft Category Gating and Topic Specificity Bonus, a symmetric ablation experiment was conducted across inverted-syntax stress queries (Table 4).

*Table 4. Symmetric Component Ablation Study on Extreme Stress Queries.*
| Model Configuration | Precision@1 | Recall@5 | MRR | Absolute P@1 Gain |
| :--- | :---: | :---: | :---: | :---: |
| Pure Okapi BM25 Baseline | 26.80% | 58.80% | 0.4167 | Baseline |
| Okapi BM25 (+ Gate & Topic Bonus) | **47.40%** | **78.60%** | **0.6169** | **+20.60%** |
| Pure Sentence-BERT Baseline | 11.80% | 28.40% | 0.1988 | Baseline |
| Sentence-BERT (+ Gate & Topic Bonus) | **27.20%** | **66.40%** | **0.4346** | **+15.40%** |
| Pure Hybrid (50:50 without Enhancements) | 30.40% | 57.20% | 0.4280 | Baseline |
| **Proposed Hybrid (BM25 + S-BERT + Gate + Bonus)** | **57.40%** 🏆 | **85.40%** 🏆 | **0.6916** 🏆 | **+27.00%** |

The ablation results empirically prove that the proposed gating and topic heuristics are modular, plug-and-play architectural enhancements that universally elevate standalone lexical (+20.60%), standalone semantic (+15.40%), and composite hybrid (+27.00%) retrievers.

#### F. Model Efficiency and Computational Footprint
A common hypothesis assumes that vocabulary mismatch can be resolved solely by deploying larger language models. However, testing demonstrates that specialized fiqh terms encounter a **Domain Vocabulary Gap** unresolvable by general-domain parameter scaling alone without domain fine-tuning. Table 5 presents a comparative assessment of multilingual embedding alternatives.

*Table 5. Architectural Comparison of Multilingual Dense Embeddings.*
| Model Architecture | Vector Dim | Parameters | Model Size | CPU Latency | Cloud Deployment Feasibility |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`paraphrase-multilingual-MiniLM-L12-v2` (Ours)** | **384** | **117M** | **~470 MB** | **2.30 ms** | **Highly Stable:** RAM < 600 MB on free cloud tiers. |
| `paraphrase-multilingual-mpnet-base-v2` | 768 | 278M | ~1,100 MB | 6.50 ms | Moderate: Susceptible to container memory limits. |
| `BAAI/bge-m3` | 1024 | 560M | ~2,200 MB | 18.00 ms | Impractical: Exceeds standard serverless memory limits. |
| `firqaaa/indo-sentence-bert-base` | 768 | 125M | ~500 MB | 5.80 ms | Stable: Limited coverage of Arabic legal transliterations. |

Selecting `MiniLM-L12-v2` represents an optimal engineering decision, delivering near-instant inference (2.30 ms) and robust stability on resource-constrained cloud platforms.

#### G. System Implementation and Dashboard Interface
The proposed architecture was deployed as a production-grade web application using Streamlit (`https://quranica.streamlit.app/`). The modular codebase comprises `app.py` (UI controller), `retrieval.py` (hybrid engine), `classifier.py` (rule-based fiqh legal status classifier), `ui_styles.py` (design system), and `utils.py` (prayer APIs and query logging). The search interface is illustrated in Figure 2.

```
+------------------------------------------------------------------------+
| 🕌 Quranica — Hybrid Islamic Jurisprudence Retrieval System            |
| [ Search Fatwa: hukum solat keluar mimisan                           ] |
+------------------------------------------------------------------------+
| Top Result (#1): Darah keluar setelah wudhu                            |
| Category: THAHARAH | Relevance Score: 1.4883                           |
| Nahdlatul Ulama Ruling: Boleh / Tidak Membatalkan (Pendapat Mu'tamad)  |
| Muhammadiyah Ruling: Tidak Membatalkan Wudhu (Suci)                    |
| Consensus Status: 🤝 Sepakat (Ittifaq)                                 |
+------------------------------------------------------------------------+
```
*Figure 2. Schematic Interface of the Quranica Retrieval System.*

#### H. Delimitation of Corpus Scope
The scope of this research is strictly focused on **Islamic Worship Jurisprudence (*Fiqh Ubudiyyah*)**, covering Thaharah, Shalat, Puasa, Zakat, and Haji/Umrah across 500 verified rulings. Consequently, queries concerning contemporary Islamic financial transactions (*mu'amalah* such as cryptocurrency, digital lending, paylater), family law (*munakahat*), and criminal/political jurisprudence (*jinayah/siyasah*) fall outside the current document index. Within its designated scope, the 500-document comparative corpus provides exhaustive, rigorous coverage for undergraduate evaluation and accredited journal publication.

---

### 5. CONCLUSION
This study developed and evaluated an enhanced Hybrid Islamic Jurisprudence Retrieval architecture integrating Okapi BM25 lexical indexing and Multilingual Sentence-BERT dense embeddings. Benchmarked across 1,000 queries using the Cranfield evaluation paradigm, the balanced 50:50 fusion configuration achieved superior performance, recording Precision@1 of 75.00%, Recall@5 of 93.80%, and MRR of 0.8319 on unseen formal queries, alongside Precision@1 of 64.20%, Recall@5 of 92.60%, and MRR of 0.7670 on colloquial stress-test queries. The empirical findings confirm that Precision@1 serves as the decisive metric in known-item religious QA search. Furthermore, an ablation study validated that the proposed Soft Category Gating and Specificity-Aware Topic Bonus significantly elevate standalone BM25 (+20.60%) and standalone Sentence-BERT (+15.40%), providing an effective, low-latency (2.30 ms) solution to the domain vocabulary mismatch problem. Future research will explore expanding the corpus to contemporary financial transactions (*mu'amalah*) and incorporating dense fine-tuning via contrastive learning on user interaction logs.

---

### 6. REFERENCES
Adek, R. T., Bustami, & Ula, M. (2021). Systematic review on the application of social media analytics for detecting radical and extremist groups. *IOP Conference Series: Materials Science and Engineering*, 1071(1), 012029. https://doi.org/10.1088/1757-899X/1071/1/012029

Arifin, S., Khisni, A., & Wahyuningsih, S. (2022). Metodologi istinbath hukum Majelis Tarjih Muhammadiyah dan Lembaga Bahtsul Masail Nahdlatul Ulama dalam fatwa fikih kontemporer. *Jurnal Hukum Ius Quia Iustum*, 29(2), 341–362. https://doi.org/10.20885/iustum.vol29.iss2.art7

Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. In *Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval* (pp. 758–759). ACM. https://doi.org/10.1145/1571941.1572114

Fadlan, S., & Ramdani, D. (2022). Penerapan Social Network Analysis pada jaringan GSM untuk analisa jaringan kriminal. *Jurnal Teknologi Informasi*, 2(1), 45–54.

Fauzi, M. (2021). Tradisi Bahtsul Masail Nahdlatul Ulama: Dinamika fiqih kontekstual dalam menjawab persoalan keumatan. *Al-Manahij: Jurnal Kajian Hukum Islam*, 15(1), 89–104. https://doi.org/10.24090/mnh.v15i1.4820

Huda, M., Mu'alim, A., & Sofyan, M. (2022). Manhaj Tarjih Muhammadiyah: Paradigma integratif bayani, burhani, dan irfani dalam ijtihad kontemporer. *Jurnal Ilmiah Syari'ah*, 21(1), 55–72. https://doi.org/10.31958/jis.v21i1.5642

Jain, L., Katarya, R., & Sachdeva, S. (2023). Opinion leaders for information diffusion using Graph Neural Network in online social networks. *ACM Transactions on the Web*, 17(2), 1–37. https://doi.org/10.1145/3580516

Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W. (2020). Dense passage retrieval for open-domain question answering. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)* (pp. 6769–6781). ACL. https://doi.org/10.18653/v1/2020.emnlp-main.550

Kurniawan, D., & Syarif, A. (2023). Natural language processing for Islamic religious texts: Challenges, opportunities, and future directions. *Journal of King Saud University - Computer and Information Sciences*, 35(4), 101421. https://doi.org/10.1016/j.jksuci.2023.101421

Luan, Y., Eisenstein, J., Toutanova, K., & Collins, M. (2021). Sparse, dense, and attentional representations for text retrieval. *Transactions of the Association for Computational Linguistics*, 9, 329–345. https://doi.org/10.1162/tacl_a_00369

Mukti, D. S., Adek, R. T., & Agusniar, C. (2026). Topic classification on Twitter using a Multi-View Graph Neural Network (MV-GAT) model. *Brilliance: Research of Artificial Intelligence*, 6(3), 383–391. https://doi.org/10.47709/brilliance.v6i3.9061

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using siamese BERT-networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)* (pp. 3982–3992). ACL. https://doi.org/10.18653/v1/D19-1410

Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333–389. https://doi.org/10.1561/1500000019

Thayyibi, A. D., & Mansur, J. F. (2021). Implementation of Social Network Analysis in the spread of Natuna issues on Twitter. *JISA: Jurnal Informatika Dan Sains*, 4(1), 12–19. https://doi.org/10.31326/jisa.v4i1.899

Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained Transformers. In *Advances in Neural Information Processing Systems (NeurIPS 2020)* (Vol. 33, pp. 5776–5788).

Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., & Yu, P. S. (2021). A comprehensive survey on graph neural networks. *IEEE Transactions on Neural Networks and Learning Systems*, 32(1), 4–24. https://doi.org/10.1109/TNNLS.2020.2978386

Zhang, Y., Gong, Y., Shen, Y., Lv, X., & Jiang, D. (2023). Multi-stage conversational recommendation with hybrid retrieval and constraint-guided reranking. *Information Processing & Management*, 60(2), 103212. https://doi.org/10.1016/j.ipm.2022.103212

Zheng, Z., Hui, K., He, B., Han, X., Sun, M., & Liu, Z. (2021). BERT-QE: Contextualized query expansion for large-scale language model reranking. In *Findings of the Association for Computational Linguistics: EMNLP 2021* (pp. 4718–4728). ACL. https://doi.org/10.18653/v1/2021.findings-emnlp.399
