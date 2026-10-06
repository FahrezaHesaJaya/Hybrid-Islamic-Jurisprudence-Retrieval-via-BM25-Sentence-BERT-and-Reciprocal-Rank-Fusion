# LAPORAN TEKNIS EVALUASI KINERJA SISTEM TEMU KEMBALI HIBRIDA
## Evaluasi Empiris Okapi BM25 dan Sentence-BERT pada Korpus Fatwa Fikih Ubudiyah Menggunakan 1.000 Kueri Uji

**Peneliti:** Fahreza Hesa Jaya  
**Program Studi:** Teknik Informatika, Fakultas Teknik, Universitas Malikussaleh  
**Dokumen:** Laporan Teknis Evaluasi Akhir Sistem  
**Tanggal:** 7 Oktober 2026  
**Status Dokumen:** Final  

---

### ABSTRAK

Laporan teknis ini menyajikan hasil evaluasi kinerja komprehensif terhadap arsitektur *Hybrid Information Retrieval* yang menggabungkan representasi leksikal Okapi BM25 dan representasi semantik Multilingual Sentence-BERT (`paraphrase-multilingual-MiniLM-L12-v2`). Pengujian dilakukan mengikuti paradigma evaluasi Cranfield terhadap korpus 500 dokumen fatwa fikih ubudiyah komparatif (Nahdlatul Ulama dan Majelis Tarjih Muhammadiyah). Evaluasi melibatkan 1.000 kueri uji yang terbagi ke dalam dua rezim pengujian: 500 kueri formal *out-of-sample* (*unseen*) dan 500 kueri kolokial ber-typo (*stress test*). Melalui *grid search* 13 variasi bobot fusi ($w_{\text{BM25}} : w_{\text{SBERT}}$), penelitian ini mengkaji perilaku sistem terhadap metrik *Precision@1* (P@1), *Recall@3* (R@3), *Recall@5* (R@5), *Mean Reciprocal Rank* (MRR), *Mean Average Precision* (MAP@5), dan *Normalized Discounted Cumulative Gain* (NDCG@5). Hasil pengujian membuktikan bahwa konfigurasi bobot seimbang 50:50 menghasilkan kinerja optimal dengan P@1 tertinggi sebesar 75,00% pada kueri formal dan 64,20% pada kueri kolokial, serta MRR 0,8319 dan 0,7670.

---

### 1. PENDAHULUAN DAN TUJUAN PENGUJIAN

Tujuan dari evaluasi teknis ini adalah:
1. Mengukur performa retrieval secara objektif pada tingkat peringkat pertama (*Precision@1*) dan perangkingan kumulatif (*MRR*, *NDCG@5*).
2. Menentukan rasio bobot linear optimal antara komponen leksikal BM25 dan komponen semantik Sentence-BERT melalui pengujian *grid search* 13 variasi konfigurasi.
3. Menguji ketahanan sistem (*robustness*) terhadap variasi bahasa non-baku, salah ketik (*typographical errors*), dan istilah percakapan sehari-hari (*slang*).
4. Menganalisis trade-off antara ukuran model embedding (efisiensi memori) terhadap penanganan kesenjangan istilah (*domain vocabulary gap*).

Dalam konteks sistem pencarian fatwa tanya-jawab (*Known-Item Search*), metrik utama keberhasilan adalah **Precision@1 (P@1)** dan **Mean Reciprocal Rank (MRR)**. Pengguna sistem konsultasi fikih menghendaki ketetapan fatwa yang paling relevan langsung berada pada posisi teratas tanpa harus menelusuri hasil pencarian lebih lanjut.

---

### 2. METODOLOGI DAN FORMULASI EVALUASI

#### 2.1 Spesifikasi Korpus dan Pipeline Retrieval
- **Korpus Data:** 500 dokumen fatwa ubudiyah (*Data_ubudiyah_final.parquet*), mencakup Thaharah (100), Shalat (200), Puasa (100), Zakat (80), dan Haji/Umrah (20).
- **Komponen Leksikal:** Okapi BM25 ($k_1 = 1.5, b = 0.75$) dengan pemrosesan awal tokenisasi, stemming Sastrawi, dan penghapusan stopword bahasa Indonesia.
- **Komponen Semantik:** Dense vector embedding 384 dimensi dari model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- **Fungsi Skor Akhir:**
  $$S_{\text{final}}(d, q) = w_{\text{BM25}} \cdot \hat{S}_{\text{BM25}}(d, q) + w_{\text{SBERT}} \cdot \hat{S}_{\text{SBERT}}(d, q) + S_{\text{topic}}(d, q) + S_{\text{cat}}(d, q)$$
  di mana $\hat{S}$ merupakan skor yang telah melalui normalisasi Min-Max per kueri:
  $$\hat{S} = \frac{S - S_{\min}}{S_{\max} - S_{\min}}$$

#### 2.2 Formulasi Metrik Evaluasi
1. **Precision@k (P@k):**
   $$\text{P@}k = \frac{|\text{Dokumen Relevan} \cap \text{Top-}k|}{k}$$
   Untuk pengujian *known-item* dengan 1 dokumen target: $\text{P@1} \in \{0, 1\}$.
2. **Recall@k (R@k):**
   $$\text{R@}k = \frac{|\text{Dokumen Relevan} \cap \text{Top-}k|}{|\text{Total Dokumen Relevan}|}$$
3. **Mean Reciprocal Rank (MRR):**
   $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
4. **Normalized Discounted Cumulative Gain (NDCG@k):**
   $$\text{DCG@}k = \sum_{i=1}^k \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}, \quad \text{NDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}$$

---

### 3. HASIL EKSPERIMEN

#### 3.1 Rezim 1: Evaluasi 500 Kueri Formal Out-of-Sample (Unseen Queries)

Pengujian dilakukan pada kueri alami dengan struktur kalimat lengkap dan istilah fikih formal yang belum pernah diproses selama perancangan arsitektur.

| Konfigurasi Bobot | $w_{\text{BM25}}$ | $w_{\text{SBERT}}$ | P@1 (%) | R@3 (%) | R@5 (%) | MRR | MAP@5 | NDCG@5 | Latensi (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BM25 Murni (100:0) | 1,00 | 0,00 | 64,00 | 82,00 | 88,00 | 0,7470 | 0,7470 | 0,7721 | 2,30 |
| Bobot 90:10 | 0,90 | 0,10 | 66,40 | 84,20 | 90,00 | 0,7687 | 0,7687 | 0,7947 | 2,30 |
| Bobot 80:20 | 0,80 | 0,20 | 69,40 | 85,40 | 92,20 | 0,7897 | 0,7897 | 0,8179 | 2,30 |
| Bobot 75:25 | 0,75 | 0,25 | 71,00 | 87,00 | 92,40 | 0,8013 | 0,8013 | 0,8273 | 2,30 |
| Bobot 70:30 | 0,70 | 0,30 | 72,20 | 88,20 | 92,20 | 0,8108 | 0,8108 | 0,8336 | 2,30 |
| Bobot 60:40 | 0,60 | 0,40 | 73,80 | 88,20 | 93,00 | 0,8233 | 0,8233 | 0,8454 | 2,30 |
| **Bobot 50:50 (Optimal)** | **0,50** | **0,50** | **75,00 🏆** | **89,80 🏆** | **93,80 🏆** | **0,8319 🏆** | **0,8319 🏆** | **0,8546 🏆** | **2,30** |
| Bobot 40:60 | 0,40 | 0,60 | 74,80 | 88,40 | 93,60 | 0,8259 | 0,8259 | 0,8492 | 2,30 |
| Bobot 30:70 | 0,30 | 0,70 | 71,20 | 86,80 | 91,80 | 0,8022 | 0,8022 | 0,8258 | 2,30 |
| Bobot 25:75 | 0,25 | 0,75 | 70,00 | 85,40 | 90,60 | 0,7913 | 0,7913 | 0,8140 | 2,30 |
| Bobot 20:80 | 0,20 | 0,80 | 67,40 | 84,00 | 90,00 | 0,7703 | 0,7703 | 0,7969 | 2,30 |
| Bobot 10:90 | 0,10 | 0,90 | 62,60 | 79,40 | 85,80 | 0,7281 | 0,7281 | 0,7522 | 2,30 |
| S-BERT Murni (0:100) | 0,00 | 1,00 | 57,00 | 74,20 | 82,80 | 0,6768 | 0,6768 | 0,7060 | 2,30 |

*\*Tanda 🏆 mengindikasikan skor kinerja tertinggi pada setiap kolom metrik evaluasi.*

---

#### 3.2 Rezim 2: Evaluasi 500 Kueri Kolokial, Typo, dan Slang (Stress Test)

Pengujian ketahanan terhadap kueri percakapan sehari-hari yang memuat singkatan kata, variasi dialek, dan salah ketik leksikal secara acak dengan probabilitas 88%.

| Konfigurasi Bobot | $w_{\text{BM25}}$ | $w_{\text{SBERT}}$ | P@1 (%) | R@3 (%) | R@5 (%) | MRR | MAP@5 | NDCG@5 | Latensi (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BM25 Murni (100:0) | 1,00 | 0,00 | 56,00 | 80,40 | 85,00 | 0,6982 | 0,6982 | 0,7235 | 2,48 |
| Bobot 90:10 | 0,90 | 0,10 | 58,40 | 82,00 | 85,80 | 0,7179 | 0,7179 | 0,7407 | 2,48 |
| Bobot 80:20 | 0,80 | 0,20 | 59,80 | 83,60 | 89,20 | 0,7312 | 0,7312 | 0,7626 | 2,48 |
| Bobot 75:25 | 0,75 | 0,25 | 62,00 | 84,60 | 90,20 | 0,7458 | 0,7458 | 0,7769 | 2,48 |
| Bobot 70:30 | 0,70 | 0,30 | 63,00 | 85,40 | 91,40 | 0,7526 | 0,7526 | 0,7864 | 2,48 |
| Bobot 60:40 | 0,60 | 0,40 | 63,40 | **88,40 🏆** | **93,20 🏆** | 0,7629 | 0,7629 | 0,8011 | 2,48 |
| **Bobot 50:50 (Optimal)** | **0,50** | **0,50** | **64,20 🏆** | 87,80 | 92,60 | **0,7670 🏆** | **0,7670 🏆** | **0,8021 🏆** | **2,48** |
| Bobot 40:60 | 0,40 | 0,60 | 62,60 | 85,80 | 91,60 | 0,7507 | 0,7507 | 0,7867 | 2,48 |
| Bobot 30:70 | 0,30 | 0,70 | 59,20 | 83,00 | 90,00 | 0,7202 | 0,7202 | 0,7589 | 2,48 |
| Bobot 25:75 | 0,25 | 0,75 | 55,00 | 79,80 | 88,00 | 0,6937 | 0,6937 | 0,7325 | 2,48 |
| Bobot 20:80 | 0,20 | 0,80 | 52,80 | 77,40 | 85,80 | 0,6701 | 0,6701 | 0,7078 | 2,48 |
| Bobot 10:90 | 0,10 | 0,90 | 48,20 | 71,80 | 81,00 | 0,6197 | 0,6197 | 0,6559 | 2,48 |
| S-BERT Murni (0:100) | 0,00 | 1,00 | 42,00 | 65,20 | 74,60 | 0,5633 | 0,5633 | 0,5946 | 2,48 |

*\*Tanda 🏆 mengindikasikan skor kinerja tertinggi pada setiap kolom metrik evaluasi.*

---

### 4. ANALISIS DAN PEMBAHASAN

#### 4.1 Prioritas Precision@1 dalam Known-Item Retrieval
Dalam Information Retrieval untuk sistem tanya-jawab agama, pengguna membutuhkan jawaban otoritatif instan pada peringkat pertama. Peringkat kedua atau ketiga memerlukan interaksi pengguna tambahan (*cognitive load*). Oleh karena itu:
- **Precision@1 (P@1)** menjadi tolok ukur utama keberhasilan sistem.
- **Mean Reciprocal Rank (MRR)** merefleksikan seberapa dekat dokumen target dengan posisi teratas ketika gagal berada di peringkat #1.
- Berdasarkan data empiris pada Rezim 1 dan Rezim 2, konfigurasi bobot 50:50 menghasilkan nilai P@1 tertinggi di kedua rezim (**75,00%** dan **64,20%**) serta MRR tertinggi (**0,8319** dan **0,7670**).

#### 4.2 Dinamika Trade-Off Bobot 50:50 vs 60:40
Pada pengujian kueri kolokial dan typo (Tabel 3.2), teridentifikasi perbedaan perilaku yang terukur antara bobot 50:50 dan 60:40:
1. **Recall@3 dan Recall@5 pada 60:40:** Konfigurasi 60:40 mencatatkan R@3 sebesar 88,40% (+0,60% di atas 50:50) dan R@5 sebesar 93,20% (+0,60% di atas 50:50). Hal ini disebabkan oleh modul `normalize_slang()` dan `expand_query()` yang berhasil mengembalikan token kunci baku. Skor BM25 yang dominan (60%) bertindak sebagai penarik leksikal kuat sehingga dokumen target terjaring ke dalam rentang lima besar.
2. **Keunggulan Peringkat Pertama pada 50:50:** Meskipun 60:40 sedikit lebih unggul dalam menjaring kandidat ke Top-5, penempatan dokumen tepat pada peringkat pertama (P@1) tetap dipimpin oleh 50:50 (64,20% vs 63,40%). Bobot semantik 50% memberikan daya pembeda kontekstual yang cukup untuk mendorong dokumen relevan dari peringkat #2 atau #3 naik ke peringkat #1.
3. **Kesimpulan Trade-Off:** Konfigurasi 50:50 tetap ditetapkan sebagai konfigurasi rujukan utama (*proposed model*) karena mengoptimalkan metrik prioritas sistem (P@1, MRR, dan NDCG@5).

#### 4.3 Karakteristik Keruntuhan Model Tunggal
- **Okapi BM25 Murni (100:0):** Mengalami penurunan performa akibat *vocabulary mismatch*. Ketika kueri menggunakan sinonim di luar indeks atau memuat salah ketik yang tidak terpetakan, skor BM25 menghasilkan 0, menyebabkan penurunan P@1 dari 64,00% menjadi 56,00% pada kueri typo.
- **Sentence-BERT Murni (0:100):** Mengalami degradasi lebih parah (P@1 57,00% pada kueri formal dan 42,00% pada kueri typo). Pada domain teks hukum agama, dokumen-dokumen memiliki kesamaan latar semantik yang tinggi (*dense cluster*). Tanpa pembobotan frekuensi istilah leksikal dari BM25, representasi vektor mengalami *semantic drift* antar-subtopik dalam bab yang sama.

---

### 5. ANALISIS ARSITEKTUR MODEL EMBEDDING DAN BATASAN KORPUS

#### 5.1 Evaluasi Model Multilingual Sentence-BERT
Analisis terhadap kasus retrieval istilah spesifik (seperti pencarian hukum mimisan saat shalat) menunjukkan bahwa kendala penelusuran bukan disebabkan oleh dimensi model `MiniLM-L12-v2` (384 dimensi), melainkan oleh **Domain Vocabulary Gap**:
- Model bahasa pra-latih umum memetakan istilah pendarahan hidung ke domain medis.
- Dalam ushul fikih, fenomena tersebut diklasifikasikan ke dalam kategori *darah yang keluar dari selain sabilain* dan hubungannya dengan pembatalan wudhu atau status najis.
- Integrasi modul **Query Expansion** dan **Kamus Tezaurus Fikih** terbukti menjadi solusi yang lebih efisien dan terarah dibandingkan melakukan komputasi model berparameter besar tanpa fine-tuning spesifik.

#### 5.2 Perbandingan Karakteristik Model Embedding

| Model | Dimensi | Parameter | Ukuran File | Latensi CPU | Status Kelayakan Deployment Cloud |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`paraphrase-multilingual-MiniLM-L12-v2`** | **384** | **117 Juta** | **470 MB** | **2,3 ms** | **Sangat Stabil:** Konsumsi RAM < 600 MB pada lingkungan cloud terdistribusi. |
| `paraphrase-multilingual-mpnet-base-v2` | 768 | 278 Juta | 1.100 MB | 6,5 ms | Cukup Stabil: Berisiko terhadap batas memori container pada lonjakan trafik. |
| `BAAI/bge-m3` | 1024 | 560 Juta | 2.200 MB | 18,0 ms | Tidak Disarankan: Melampaui kapasitas alokasi memori instance standar. |
| `firqaaa/indo-sentence-bert-base` | 768 | 125 Juta | 500 MB | 5,8 ms | Stabil: Representasi bahasa Indonesia baik, namun kosakata transliterasi Arab terbatas. |

Pemilihan `MiniLM-L12-v2` merupakan keputusan komputasi yang tepat dengan mempertimbangkan trade-off efisiensi memori, latensi inferensi rendah, dan stabilitas server hosting.

#### 5.3 Batasan Korpus Penelitian (Delimitation of Corpus)
Korpus data saat ini terdiri atas 500 fatwa komparatif yang terfokus secara khusus pada ranah **Fikih Ubudiyah (Ibadah Keseharian)**:
1. Thaharah (Bersuci dan Najis)
2. Shalat (Fardhu, Sunnah, dan Jamaah)
3. Puasa (Ramadhan dan Kafarat)
4. Zakat (Fitrah, Mal, dan Pembayaran Digital)
5. Haji dan Umrah (Rukun, Larangan, dan Dam)

**Batasan Masalah:** Sistem belum mencakup domain fikih di luar ibadah ubudiyah, seperti Fikih Muamalah Finansial Kontemporer (transaksi perbankan modern, aset kripto, pinjaman digital), Fikih Munakahat (hukum keluarga), dan Fikih Jinayah/Siyasah. Untuk ruang lingkup tugas akhir sarjana dan publikasi jurnal terakreditasi, korpus 500 dokumen ini telah mencukupi syarat representasi domain yang mendalam.

---

### 6. KESIMPULAN

1. Pengujian empiris terhadap 1.000 kueri membuktikan bahwa arsitektur hibrida dengan rasio bobot **50:50** menghasilkan efektivitas retrieval tertinggi secara konsisten, mencapai Precision@1 sebesar 75,00% (kueri formal) dan 64,20% (kueri kolokial/typo), serta Recall@5 sebesar 93,80% dan 92,60%.
2. Precision@1 terkonfirmasi sebagai metrik kinerja utama untuk sistem penelusuran fatwa. Konfigurasi 50:50 mengungguli seluruh variasi bobot asimetris lainnya pada penempatan target tepat di peringkat pertama.
3. Kesenjangan istilah kolokial terhadap bahasa fikih formal diselesaikan secara efektif melalui kombinasi *Query Expansion*, normalisasi leksikal, dan penyesuaian kategori hierarkis (*Category Gating*), tanpa memerlukan model embedding berukuran besar yang memberatkan infrastruktur hosting.
