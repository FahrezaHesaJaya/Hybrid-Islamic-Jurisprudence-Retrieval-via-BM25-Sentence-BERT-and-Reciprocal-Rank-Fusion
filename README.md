# 🕌 Quranica — Sistem Temu Balik Fatwa Fikih Ubudiyah Komparatif Berbasis Hybrid Retrieval

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Repositori Penelitian Skripsi & Publikasi Ilmiah:**  
> *"Optimalisasi Relevansi Penelusuran Hukum Fikih Keseharian Menggunakan Metode Hybrid Retrieval Berbasis Sentence-BERT dan BM25"*  
> **Penulis:** Fahreza Hesa Jaya (NIM: 220170117)  
> **Program Studi:** Teknik Informatika, Fakultas Teknik, Universitas Malikussaleh  
> **Dosen Pembimbing:** Safwandi, S.T., M.Kom & Nunsina, S.T., M.Kom  

---

## 📖 Ringkasan Proyek

**Quranica** adalah sistem penelusuran temu balik informasi (*Information Retrieval*) khusus literatur hukum fikih Islam komparatif antara dua ormas Islam terbesar di Indonesia: **Nahdlatul Ulama (LBM-PBNU)** dan **Muhammadiyah (Majelis Tarjih PP Muhammadiyah)**.

Sistem ini memecahkan dua masalah klasik dalam *Information Retrieval*:
1. **Term Mismatch Problem pada BM25:** Kegagalan saat pengguna memasukkan kueri sehari-hari yang bersifat percakapan (*slang*), singkatan chat, atau salah ketik (*typo*).
2. **Semantic Drift pada Sentence-BERT:** Model *dense embedding* umum kerap tertukar membedakan istilah fikih khusus yang bertetangga dekat secara semantik (*madzi* vs *mani*, wudhu vs mandi wajib).

Dengan menggabungkan representasi leksikal **Okapi BM25** dan representasi semantik **Multilingual Sentence-BERT** (bobot seimbang 50:50) yang diperkuat dengan:
- **Dynamic Min-Max Normalization** per kueri.
- **Advanced Slang & Chat Normalizer** dengan proteksi ~100 kata umum bahasa Indonesia.
- **Soft Hierarchical Category Gate** non-destruktif.
- **Specificity-Aware Topic Bonus** untuk keabsahan hukum spesifik.
- **Deteksi Konsensus Hukum Otomatis** (Mubah, Boleh, Batal, Wajib, Haram).

---

## 📊 Hasil Evaluasi Kuantitatif (Cranfield Paradigm)

Evaluasi kuantitatif penuh dilakukan pada **500 dokumen fatwa** dengan 3 skenario kueri uji:

| Skenario Pengujian | BM25 Murni | Indo S-BERT Murni | Proposed Hybrid Retrieval | Peningkatan vs BM25 |
| :--- | :---: | :---: | :---: | :---: |
| **500 Kueri Bersih (Out-of-Sample)** | P@1: 73.0% (MRR: 0.7913) | P@1: 35.2% (MRR: 0.4590) | **P@1: 92.8% \| R@5: 98.6% (MRR: 0.9535)** | **+27.12%** |
| **500 Kueri Chat, Slang & Typo** | P@1: 31.0% (MRR: 0.4539) | P@1: 19.6% (MRR: 0.2929) | **P@1: 60.2% \| R@5: 92.6% (MRR: 0.7425)** | **+94.19%** |
| **500 Kueri Terbalik (Inverted Syntax)** | P@1: 26.8% (MRR: 0.4167) | P@1: 11.8% (MRR: 0.1988) | **P@1: 57.4% \| R@5: 85.4% (MRR: 0.6916)** | **+114.18%** |

---

## 🚀 Panduan Menjalankan Aplikasi Secara Lokal

1. **Clone repositori:**
   ```bash
   git clone https://github.com/FahrezaHesaJaya/quranica-fikih-search.git
   cd quranica-fikih-search
   ```

2. **Buat virtual environment dan pasang dependensi:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Di Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Jalankan aplikasi Streamlit:**
   ```bash
   streamlit run app.py
   ```
   Aplikasi akan terbuka otomatis di peramban pada alamat `http://localhost:8501`.

---

## 📂 Struktur Repositori

```text
├── app.py                             # Kode utama Streamlit (Search Engine & UI)
├── requirements.txt                   # Daftar dependensi pustaka Python
├── .streamlit/
│   └── config.toml                    # Konfigurasi tema Quranica (Google-style)
├── assets/
│   └── quranica_hero.jpg              # Banner aset visual resmi
├── Data_ubudiyah_final.csv            # Korpus 500 fatwa ubudiyah komparatif NU & Muhammadiyah
├── Data_ubudiyah_final.parquet        # Representasi korpus cepat (Parquet)
├── bm25_model.pkl                     # Model indeks Okapi BM25 tersimpan
├── corpus_emb.npy                     # Vektor embedding Sentence-BERT korpus
├── DRAFT_SKRIPSI_DAN_JURNAL_SINTA3.md # Naskah Bab IV, Bab V, dan Artikel Jurnal SINTA 3
├── hasil_evaluasi_skripsi_500.csv     # Rekap hasil evaluasi kueri bersih
├── hasil_evaluasi_500_chat_typo.csv   # Rekap hasil evaluasi kueri slang & typo
├── hasil_evaluasi_500_inverted_hard.csv # Rekap hasil evaluasi kueri sintaksis terbalik
└── README.md                          # Dokumentasi proyek
```

---

## 👨‍💻 Pengembang

**Fahreza Hesa Jaya**  
NIM: 220170117  
Jurusan Teknik Informatika, Universitas Malikussaleh  
Email: [fahrezzahesajaya@gmail.com](mailto:fahrezzahesajaya@gmail.com)
