# =============================================================
# app.py — Quranica Unimal
# Fikih Hybrid Search: BM25 + Indo S-BERT + Weighted RRF
# Premium UI — Clean Google Style with Quranica Theme
# =============================================================

import os
import random
import html as html_lib
import urllib.parse
from datetime import datetime
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ── Import Modular Components ─────────────────────────────────
from ui_styles import render_html, apply_custom_styles, get_hero_image_b64
from retrieval import (
    DATASET_PATH, PARQUET_PATH, BM25_CACHE_PATH, EMB_CACHE_PATH,
    load_sastrawi, load_sbert, build_indexes, hybrid_search,
    search_bm25, search_sbert, expand_query, detect_query_category,
    preprocess_text, SINONIM_FIQIH
)
from classifier import (
    deteksi_hukum, ringkasan_konsensus_hukum,
    ekstrak_intisari_fatwa, buat_sintesis_eksekutif_top4
)
from utils import (
    fetch_prayer_times, get_next_prayer, highlight,
    log_pencarian, log_feedback,
    TRIVIA_FIKIH, FATWA_RECS, QUICK_TOPICS, HARI_ID, BULAN_ID
)

# ── Paths & Configuration ─────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
HERO_IMG_PATH = os.path.join(BASE_DIR, "assets", "quranica_hero.jpg")

st.set_page_config(
    page_title="Quranica — Pencarian Fikih Islam",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Terapkan Desain Sistem UI & Tema ──────────────────────────
apply_custom_styles()

if "w_sbert" not in st.session_state:
    st.session_state["w_sbert"] = 0.50
if "top_k" not in st.session_state:
    st.session_state["top_k"] = 5
if "quick_query" not in st.session_state:
    st.session_state["quick_query"] = ""


# =============================================================
# TOP NAVIGATION
# =============================================================
render_html("""
<div class="top-nav">
  <div class="top-nav-brand">
    <span class="q-accent">🕌</span>
    <span>Quranica Unimal</span>
  </div>
  <div class="top-nav-info">
    <span>Pencarian Fikih Ubudiyah Komparatif NU & Muhammadiyah</span>
  </div>
</div>
""")


# =============================================================
# JADWAL SHOLAT BAR
# =============================================================
prayer_times = fetch_prayer_times("Lhokseumawe", "ID")
next_prayer  = get_next_prayer(prayer_times)

prayer_icons = {
    "Subuh": "🌌", "Dzuhur": "☀️", "Ashar": "🌤️", "Maghrib": "🌅", "Isya": "🌙"
}
HARI_ID = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
BULAN_ID = ["", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
            "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
_now = datetime.now()
hari_id = HARI_ID[_now.weekday()]
bulan_id = BULAN_ID[_now.month]
today_str = f"{hari_id}, {_now.day} {bulan_id} {_now.year}"

prayer_chips_html = ""
for p_name, p_time in prayer_times.items():
    is_act = (p_name == next_prayer)
    act_cls = "active" if is_act else ""
    ico = prayer_icons.get(p_name, "🕌")
    prayer_chips_html += f"""<div class="prayer-chip {act_cls}"><span class="p-badge">⏰ Berikutnya</span><div class="p-name">{ico} {p_name}</div><div class="p-time">{p_time}</div></div>"""

render_html(f"""
<div class="prayer-bar-wrap">
  <div class="prayer-bar-header">🕌 Jadwal Sholat</div>
  <div class="prayer-cards-group">
    {prayer_chips_html}
  </div>
  <div class="prayer-loc-date">📍 Lhokseumawe &nbsp;·&nbsp; {today_str}</div>
</div>
""")


# =============================================================
# LOAD DATASET & AI INDEXES
# =============================================================
if not os.path.exists(DATASET_PATH):
    st.error(f"❌ File dataset tidak ditemukan di: `{DATASET_PATH}`")
    st.stop()

try:
    ret_idx = build_indexes(DATASET_PATH)
    df, bm25_model, corpus_emb, cols = ret_idx[0], ret_idx[1], ret_idx[2], ret_idx[3]
    sbert_model = load_sbert()
    stemmer, stopword = load_sastrawi()
except Exception as e:
    st.error(f"⚠️ Gagal memuat dataset: {e}")
    st.stop()


# =============================================================
# HERO SECTION
# =============================================================
hero_b64 = get_hero_image_b64()
banner_img_html = f"""<div class="hero-banner-frame"><img src="data:image/jpeg;base64,{hero_b64}" class="hero-banner-image" alt="Quranica Banner" /></div>""" if hero_b64 else ""

render_html(f"""
<div class="hero-container">
  {banner_img_html}
  <h1 class="brand-title">
    <span class="q-Q">Q</span><span class="q-u">u</span><span class="q-r">r</span><span class="q-a">a</span><span class="q-n">n</span><span class="q-i">i</span><span class="q-c">c</span><span class="q-aa">a</span>
  </h1>
  <div class="subtitle-badge">
    <span class="sub-lead">Pencarian Fatwa Fikih Berbasis AI</span>
    <span class="sub-sep">•</span>
    <span class="org-nu">Nahdlatul Ulama</span>
    <span class="sub-and">&amp;</span>
    <span class="org-mu">Muhammadiyah</span>
    <span class="sub-sep">•</span>
    <span class="org-unimal">Universitas Malikussaleh</span>
  </div>
  <div class="stats-pills-row">
    <div class="stat-chip"><span>📚</span> 500 Dokumen Fikih</div>
    <div class="stat-chip"><span>⚖️</span> Komparasi NU &amp; MU</div>
    <div class="stat-chip"><span>⚡</span> Hybrid BM25 + S-BERT</div>
  </div>
</div>
""")


# =============================================================
# GOOGLE-STYLE SEARCH FORM
# =============================================================
with st.form("google_search_form", clear_on_submit=False):
    query_input = st.text_input(
        "Pertanyaan",
        value=st.session_state.get("quick_query", ""),
        placeholder="Tanyakan persoalan fikih Anda... (contoh: apakah sah shalat tanpa wudhu karena lupa?)",
        label_visibility="collapsed",
    )
    col_sp1, col_center, col_sp2 = st.columns([1, 2, 1])
    with col_center:
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            btn_search = st.form_submit_button("🔍 Cari Fatwa", use_container_width=True, type="primary")
        with btn_c2:
            btn_lucky = st.form_submit_button("🎲 Fatwa Acak", use_container_width=True)

if btn_lucky:
    picked = random.choice(FATWA_RECS)
    st.session_state["hasil_hybrid"] = [(picked["doc_idx"], 1.0)]
    st.session_state["query_terakhir"] = picked["title"]
    st.session_state["quick_query"] = ""
    st.rerun()

# Quick Topic Chips
q_cols = st.columns(len(QUICK_TOPICS))
for i, (label, query_text) in enumerate(QUICK_TOPICS):
    with q_cols[i]:
        if st.button(label, key=f"quick_chip_{i}", use_container_width=True):
            st.session_state["quick_query"] = query_text
            st.session_state.pop("hasil_hybrid", None)
            st.rerun()


# =============================================================
# SETTINGS EXPANDER
# =============================================================
with st.expander("⚙️ Pengaturan Bobot Algoritma Hybrid", expanded=False):
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        w_sbert = st.slider("Bobot S-BERT (Semantik)", 0.0, 1.0, st.session_state["w_sbert"], 0.05)
        st.session_state["w_sbert"] = w_sbert
    with col_s2:
        top_k = st.slider("Jumlah Dokumen Ditampilkan", 3, 10, st.session_state["top_k"], 1)
        st.session_state["top_k"] = top_k
    w_bm25 = round(1.0 - w_sbert, 2)
    st.caption(f"Bobot BM25 (Leksikal): **{w_bm25}** · Bobot S-BERT: **{w_sbert}** · RRF Constant k=60")

w_sbert = st.session_state.get("w_sbert", 0.50)
w_bm25  = round(1.0 - w_sbert, 2)
top_k   = st.session_state.get("top_k", 5)


# =============================================================
# SEARCH EXECUTION
# =============================================================
if btn_search:
    if not query_input.strip():
        st.warning("⚠️ Silakan ketikkan pertanyaan atau kata kunci fikih terlebih dahulu.")
    else:
        st.session_state["quick_query"] = ""
        try:
            with st.spinner("🔍 Menelusuri ribuan dalil dan fatwa dengan AI Hybrid Retrieval..."):
                exp_q = expand_query(query_input)
                detected_gate = detect_query_category(exp_q)
                st.session_state["gated_kategori"] = detected_gate
                hasil, bm_res, sb_res = hybrid_search(
                    query_input, bm25_model, sbert_model, corpus_emb,
                    stemmer, stopword, df=df, cols=cols, top_k=top_k,
                    w_bm25=w_bm25, w_sbert=w_sbert, return_details=True
                )
            if not hasil:
                st.session_state.pop("hasil_hybrid", None)
                st.session_state.pop("hasil_bm25_only", None)
                st.session_state.pop("hasil_sbert_only", None)
                st.session_state["query_terakhir"] = query_input
                st.info("ℹ️ Tidak ada fatwa yang cocok dengan kata kunci tersebut. Coba gunakan istilah umum (misal: 'wudhu', 'puasa', 'shalat').")
            else:
                hasil_limit = hasil[:4]
                st.session_state["hasil_hybrid"]     = hasil_limit
                st.session_state["hasil_bm25_only"]  = bm_res[:3]
                st.session_state["hasil_sbert_only"] = sb_res[:3]
                st.session_state["scroll_to_results"] = True
                st.session_state["query_terakhir"]   = query_input
                topik_list = []
                for ip, _ in hasil_limit:
                    row = df.iloc[ip]
                    t = row.get(cols["top"], None) if cols["top"] else None
                    topik_list.append(str(t) if t and str(t).lower() != "nan" else f"Dokumen #{ip+1}")
                log_pencarian(query_input, topik_list)
                st.rerun()
        except Exception as e:
            st.error(f"⚠️ Terjadi kesalahan saat mencari: {e}")


# =============================================================
# DISPLAY RESULTS OR HOMEPAGE
# =============================================================
if "hasil_hybrid" in st.session_state and st.session_state["hasil_hybrid"]:
    q_last = st.session_state.get("query_terakhir", "")
    is_direct = (len(st.session_state["hasil_hybrid"]) == 1 and st.session_state["hasil_hybrid"][0][1] == 1.0)

    heading_title = "📖 Fatwa Pilihan Terverifikasi" if is_direct else "✨ Hasil Penelusuran Fatwa"
    caption_text = f"Menampilkan dokumen fatwa: &ldquo;<strong>{html_lib.escape(q_last)}</strong>&rdquo;" if is_direct else f"Menampilkan fatwa terverifikasi untuk &ldquo;<strong>{html_lib.escape(q_last)}</strong>&rdquo; diurutkan skor relevansi AI"

    gate_badge = ""
    g_cat = st.session_state.get("gated_kategori")
    if g_cat:
        gate_badge = f"""
        <div style="background: rgba(45,106,79,0.08); border: 1.5px solid #2D6A4F; border-radius: 12px; padding: 0.55rem 1rem; margin-bottom: 1.25rem; display: flex; align-items: center; justify-content: space-between; gap: 0.75rem;">
            <div style="display: flex; align-items: center; gap: 0.6rem;">
                <span style="font-size: 1.2rem;">🏛️</span>
                <span style="font-size: 0.88rem; color: #1B4332; font-weight: 700;">
                    Hierarchical Category Gate: Terkunci pada <b>Bab {g_cat}</b> (Mencegah Kebocoran Topik Luar)
                </span>
            </div>
            <span style="background: #2D6A4F; color: #FFFFFF; font-size: 0.75rem; font-weight: 800; padding: 0.2rem 0.65rem; border-radius: 8px; letter-spacing: 0.04em;">GATE TERVERIFIKASI</span>
        </div>
        """

    col_hdr, col_btn = st.columns([3, 1])
    with col_hdr:
        render_html(f"""
        <div id="hasil-pencarian-anchor"></div>
        <div class="section-heading">{heading_title}</div>
        <div class="section-caption">{caption_text}</div>
        <div class="section-accent-line line-emerald"></div>
        {gate_badge}
        """)
    with col_btn:
        components.html("""
        <div style="display:flex; justify-content:flex-end; align-items:flex-start; padding-top:14px;">
          <button onclick="window.parent.print()" style="
            background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 100%);
            color: #FFFFFF;
            border: none;
            border-radius: 10px;
            padding: 9px 16px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 13px;
            font-weight: 700;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 7px;
            box-shadow: 0 4px 12px rgba(27,67,50,0.18);
          ">
            <span>🖨️</span> Cetak / Simpan PDF
          </button>
        </div>
        """, height=55)

    # ── Live Benchmark Komparasi 3 Model (Model Comparative Analysis) ──
    bm_res_list = st.session_state.get("hasil_bm25_only", [])
    sb_res_list = st.session_state.get("hasil_sbert_only", [])
    hy_res_list = st.session_state.get("hasil_hybrid", [])

    with st.expander("🔬 Mode Komparasi 3 Model (Live Model Comparative Analysis)", expanded=False):
        render_html("""
        <div style="background: #F8FAFC; border: 1.5px solid #E2E8F0; border-radius: 12px; padding: 0.9rem 1.1rem; margin-bottom: 1rem;">
            <div style="font-weight: 800; color: #1E293B; font-size: 0.95rem; margin-bottom: 0.25rem;">
                🎯 Demonstrasi Komparasi Model Real-Time (Live Query Evaluation)
            </div>
            <div style="font-size: 0.82rem; color: #64748B; line-height: 1.5;">
                Panel ini menampilkan 3 dokumen teratas yang dihasilkan secara independen oleh masing-masing metode penelusuran. 
                Perhatikan bagaimana <b>Hybrid Retrieval</b> memadukan keunggulan leksikal <b>BM25</b> dengan pemahaman semantik <b>S-BERT</b>.
            </div>
        </div>
        """)

        c_bm, c_sb, c_hy = st.columns(3, gap="medium")
        
        # Kolom BM25
        with c_bm:
            render_html("""
            <div style="font-weight: 800; color: #1B4332; font-size: 0.9rem; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.4rem;">
                <span>📚</span> BM25 Only (Leksikal)
            </div>
            """)
            if bm_res_list:
                for b_rank, (b_idx, b_sc) in enumerate(bm_res_list[:3], 1):
                    b_row = df.iloc[b_idx]
                    b_topik = str(b_row.get(cols["top"], f"Dokumen #{b_idx+1}"))
                    b_kat = str(b_row.get(cols["kat"], "Ubudiyah"))
                    rank_icon = "🥇" if b_rank == 1 else ("🥈" if b_rank == 2 else "🥉")
                    render_html(f"""
                    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:9px; padding:0.6rem 0.8rem; margin-bottom:0.5rem; box-shadow:0 1px 4px rgba(0,0,0,0.03);">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.2rem;">
                            <span style="font-size:0.75rem; font-weight:800; color:#2D6A4F;">{rank_icon} Peringkat #{b_rank}</span>
                            <span style="font-size:0.7rem; color:#64748B; background:#F1F5F9; padding:0.1rem 0.4rem; border-radius:4px;">Skor: {b_sc:.2f}</span>
                        </div>
                        <div style="font-size:0.8rem; font-weight:700; color:#1E293B; line-height:1.35;">{html_lib.escape(b_topik)}</div>
                        <div style="font-size:0.7rem; color:#64748B; margin-top:0.25rem;">🕌 {html_lib.escape(b_kat)}</div>
                    </div>
                    """)
            else:
                st.caption("Tidak ada data.")

        # Kolom S-BERT
        with c_sb:
            render_html("""
            <div style="font-weight: 800; color: #1B4332; font-size: 0.9rem; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.4rem;">
                <span>🧠</span> Indo S-BERT Only (Semantik)
            </div>
            """)
            if sb_res_list:
                for s_rank, (s_idx, s_sc) in enumerate(sb_res_list[:3], 1):
                    s_row = df.iloc[s_idx]
                    s_topik = str(s_row.get(cols["top"], f"Dokumen #{s_idx+1}"))
                    s_kat = str(s_row.get(cols["kat"], "Ubudiyah"))
                    rank_icon = "🥇" if s_rank == 1 else ("🥈" if s_rank == 2 else "🥉")
                    render_html(f"""
                    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:9px; padding:0.6rem 0.8rem; margin-bottom:0.5rem; box-shadow:0 1px 4px rgba(0,0,0,0.03);">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.2rem;">
                            <span style="font-size:0.75rem; font-weight:800; color:#2D6A4F;">{rank_icon} Peringkat #{s_rank}</span>
                            <span style="font-size:0.7rem; color:#64748B; background:#F1F5F9; padding:0.1rem 0.4rem; border-radius:4px;">Kosinus: {s_sc*100:.1f}%</span>
                        </div>
                        <div style="font-size:0.8rem; font-weight:700; color:#1E293B; line-height:1.35;">{html_lib.escape(s_topik)}</div>
                        <div style="font-size:0.7rem; color:#64748B; margin-top:0.25rem;">🕌 {html_lib.escape(s_kat)}</div>
                    </div>
                    """)
            else:
                st.caption("Tidak ada data.")

        # Kolom Hybrid
        with c_hy:
            render_html("""
            <div style="font-weight: 800; color: #1B4332; font-size: 0.9rem; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.4rem;">
                <span>✨</span> Hybrid Retrieval (Sistem Kita)
            </div>
            """)
            if hy_res_list:
                for h_rank, (h_idx, h_sc) in enumerate(hy_res_list[:3], 1):
                    h_row = df.iloc[h_idx]
                    h_topik = str(h_row.get(cols["top"], f"Dokumen #{h_idx+1}"))
                    h_kat = str(h_row.get(cols["kat"], "Ubudiyah"))
                    rank_icon = "🥇" if h_rank == 1 else ("🥈" if h_rank == 2 else "🥉")
                    sc_display = f"{h_sc*100:.1f}%" if h_sc <= 1.0 else f"{h_sc:.2f}"
                    render_html(f"""
                    <div style="background:#ECFDF5; border:1px solid #A7F3D0; border-radius:9px; padding:0.6rem 0.8rem; margin-bottom:0.5rem; box-shadow:0 1px 4px rgba(0,0,0,0.03);">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.2rem;">
                            <span style="font-size:0.75rem; font-weight:800; color:#065F46;">{rank_icon} Peringkat #{h_rank}</span>
                            <span style="font-size:0.7rem; color:#065F46; background:#D1FAE5; padding:0.1rem 0.4rem; border-radius:4px; font-weight:700;">Relevansi: {sc_display}</span>
                        </div>
                        <div style="font-size:0.8rem; font-weight:700; color:#064E3B; line-height:1.35;">{html_lib.escape(h_topik)}</div>
                        <div style="font-size:0.7rem; color:#047857; margin-top:0.25rem;">🕌 {html_lib.escape(h_kat)}</div>
                    </div>
                    """)
            else:
                st.caption("Tidak ada data.")

        render_html("""
        <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 8px; padding: 0.65rem 0.9rem; margin-top: 0.5rem; font-size: 0.78rem; color: #166534; line-height: 1.45;">
            💡 <b>Insight Evaluasi:</b> Model Hybrid menyatukan keunggulan leksikal eksak (BM25) dan kedekatan semantik (S-BERT) secara seimbang 50:50 dengan <i>Category Gate</i>, mengantarkan fatwa paling relevan ke peringkat teratas.
        </div>
        """)

    # ── Multi-Document Executive Fiqh Synthesis (Top 4 Fatwas) ──
    top_docs_data = []
    for r_idx, (ip, score_val) in enumerate(st.session_state["hasil_hybrid"][:4]):
        row_d = df.iloc[ip]
        t_topik = str(row_d.get(cols["top"], "")) if cols["top"] else f"Dokumen #{ip+1}"
        t_kat   = str(row_d.get(cols["kat"], "Ubudiyah")) if cols["kat"] else ""
        t_j_nu  = str(row_d.get(cols["nu"],  "")) if cols["nu"]  else ""
        t_j_mu  = str(row_d.get(cols["mu"],  "")) if cols["mu"]  else ""

        t_hl_nu, t_hw_nu, t_cat_nu = deteksi_hukum(t_j_nu)
        t_hl_mu, t_hw_mu, t_cat_mu = deteksi_hukum(t_j_mu)

        t_norm_nu = t_cat_nu.lower().replace(" (rukhshah)", "").replace(" / sah", "").replace(" / batal", "")
        t_norm_mu = t_cat_mu.lower().replace(" (rukhshah)", "").replace(" / sah", "").replace(" / batal", "")
        t_is_agree = (t_norm_nu == t_norm_mu)

        top_docs_data.append({
            "rank": r_idx + 1,
            "topik": t_topik,
            "kategori": t_kat,
            "hl_nu": t_hl_nu,
            "hw_nu": t_hw_nu,
            "cat_nu": t_cat_nu,
            "hl_mu": t_hl_mu,
            "hw_mu": t_hw_mu,
            "cat_mu": t_cat_mu,
            "inti_nu": ekstrak_intisari_fatwa(t_j_nu, max_len=180),
            "inti_mu": ekstrak_intisari_fatwa(t_j_mu, max_len=180),
            "is_agree": t_is_agree,
            "score_val": score_val,
        })

    top_results = st.session_state["hasil_hybrid"]
    if top_results:
        # ── 1. TAMPILKAN JAWABAN UTAMA (PERINGKAT #1) TERLEBIH DAHULU ──
        ip, rrf_score = top_results[0]
        row      = df.iloc[ip]
        topik    = str(row.get(cols["top"], "")) if cols["top"] else ""
        kategori = str(row.get(cols["kat"], "")) if cols["kat"] else ""
        j_nu     = str(row.get(cols["nu"],  "")) if cols["nu"]  else ""
        j_mu     = str(row.get(cols["mu"],  "")) if cols["mu"]  else ""
        url_nu   = str(row.get(cols["url_nu"], "")) if cols["url_nu"] else ""
        url_mu   = str(row.get(cols["url_mu"], "")) if cols["url_mu"] else ""

        hl_nu, hw_nu, cat_nu = deteksi_hukum(j_nu)
        hl_mu, hw_mu, cat_mu = deteksi_hukum(j_mu)
        badge_nu = f'<span class="badge-hukum" style="background:{hw_nu}22;color:{hw_nu};border:1px solid {hw_nu}55">{hl_nu}</span>' if hl_nu else ""
        badge_mu = f'<span class="badge-hukum" style="background:{hw_mu}22;color:{hw_mu};border:1px solid {hw_mu}55">{hl_mu}</span>' if hl_mu else ""
        consensus_html = ringkasan_konsensus_hukum(hl_nu, cat_nu, hl_mu, cat_mu)

        score_pct = round(rrf_score * 100, 1) if rrf_score <= 1.0 else round(rrf_score, 1)
        rank_tag  = "📌 Fatwa Terpilih" if is_direct else "🥇 Jawaban Fatwa Utama (Peringkat #1)"
        pill_score_html = '<span class="pill-score">⚡ Fatwa Pilihan Langsung</span>' if is_direct else f'<span class="pill-score">⚡ Relevansi AI: {score_pct}%</span>'

        render_html(f"""
        <div id="jawaban-utama-target"></div>
        <div class="result-box" style="border:2px solid #2D6A4F;background:#FCFDFB;">
          <div class="result-rank-tag" style="background:#2D6A4F;color:#FFFFFF;padding:0.3rem 0.8rem;border-radius:8px;font-weight:800;">{rank_tag}</div>
          <div class="result-title-text" style="font-size:1.55rem;color:#1B4332;">{html_lib.escape(topik)}</div>
          <div class="result-meta-pills">
            <span class="pill-kat">🕌 Kategori: {html_lib.escape(kategori)}</span>
            {pill_score_html}
          </div>
          {consensus_html}
        </div>
        """)

        components.html("""
        <script>
        function tryScroll() {
            try {
                var el = window.parent.document.getElementById("hasil-pencarian-anchor") || window.parent.document.getElementById("jawaban-utama-target");
                if (el) {
                    el.scrollIntoView({ behavior: "smooth", block: "start" });
                    return;
                }
            } catch(e) {}
            setTimeout(tryScroll, 100);
        }
        setTimeout(tryScroll, 120);
        </script>
        """, height=0, width=0)

        j_nu_prev = j_nu[:650] + ("…" if len(j_nu) > 650 else "")
        j_mu_prev = j_mu[:650] + ("…" if len(j_mu) > 650 else "")
        hl_nu_text = highlight(j_nu_prev, q_last)
        hl_mu_text = highlight(j_mu_prev, q_last)

        col_nu, col_mu = st.columns(2, gap="medium")
        with col_nu:
            render_html(f"""
            <div class="jawaban-card nu">
              <div class="jawaban-header nu">
                <span>🟢 Nahdlatul Ulama</span>
                {badge_nu}
              </div>
              <div class="jawaban-body">{hl_nu_text}</div>
            </div>
            """)
            if len(j_nu) > 650:
                with st.expander("📖 Baca Selengkapnya (NU)"):
                    render_html(f'<div class="jawaban-body">{highlight(j_nu, q_last)}</div>')
            if url_nu.startswith("http"):
                st.markdown(f"[🔗 Buka Sumber Resmi NU]({url_nu})", unsafe_allow_html=False)

        with col_mu:
            render_html(f"""
            <div class="jawaban-card mu">
              <div class="jawaban-header mu">
                <span>🔵 Muhammadiyah</span>
                {badge_mu}
              </div>
              <div class="jawaban-body">{hl_mu_text}</div>
            </div>
            """)
            if len(j_mu) > 650:
                with st.expander("📖 Baca Selengkapnya (Muhammadiyah)"):
                    render_html(f'<div class="jawaban-body">{highlight(j_mu, q_last)}</div>')
            if url_mu.startswith("http"):
                st.markdown(f"[🔗 Buka Sumber Resmi Muhammadiyah]({url_mu})", unsafe_allow_html=False)

        fb_key = f"fb_{ip}_0"
        if fb_key not in st.session_state:
            st.session_state[fb_key] = None
        fb_c1, fb_c2, fb_sp = st.columns([1, 1, 6])
        with fb_c1:
            if st.button("👍 Bermanfaat", key=f"pos_{ip}_0"):
                st.session_state[fb_key] = "helpful"
                log_feedback(q_last, topik, "helpful")
        with fb_c2:
            if st.button("👎 Kurang", key=f"neg_{ip}_0"):
                st.session_state[fb_key] = "not_helpful"
                log_feedback(q_last, topik, "not_helpful")

        if st.session_state[fb_key] == "helpful":
            st.caption("✅ Terima kasih atas masukan Anda!")
        elif st.session_state[fb_key] == "not_helpful":
            st.caption("📝 Masukan tercatat untuk peningkatan model.")

        # Banner Tidak Cocok? Temukan Dokumen Relevan Lainnya
        render_html(f"""
        <div class="unmatched-feedback-box" style="margin:1.8rem 0 1.2rem 0;padding:1.15rem 1.4rem;background:#F8F9FA;border-left:4.5px solid #F4B400;border-radius:14px;display:flex;align-items:center;justify-content:space-between;box-shadow:0 3px 10px rgba(0,0,0,0.04);gap:1rem;flex-wrap:wrap;">
          <div style="display:flex;align-items:center;gap:0.85rem;">
            <span style="font-size:1.6rem;">💡</span>
            <div>
              <div style="font-family:'Outfit',sans-serif;font-weight:800;color:#1A202C;font-size:1.05rem;">
                Tidak cocok? Temukan beberapa dokumen yang mungkin relevan
              </div>
              <div style="font-size:0.86rem;color:#64748B;margin-top:0.2rem;">
                Telusuri ringkasan sintesis komparasi di bawah atau klik tombol untuk langsung melihat fatwa pilihan lainnya
              </div>
            </div>
          </div>
          <a href="#fatwa-lainnya-anchor" style="font-size:0.85rem;font-weight:800;color:#1B4332;background:#E8F5E9;padding:0.5rem 1.1rem;border-radius:10px;text-decoration:none;white-space:nowrap;display:inline-flex;align-items:center;gap:0.35rem;border:1px solid #A7F3D0;">
            Temukan Dokumen Lain ↓
          </a>
        </div>
        """)

    # ── 2. TAMPILKAN KESIMPULAN & SINTESIS EKSEKUTIF (4 FATWA TERKAIT) ──
    sintesis_html = buat_sintesis_eksekutif_top4(top_docs_data, q_last)
    render_html(f"""
    <div style="margin-top:2.2rem;margin-bottom:0.8rem;">
      <div style="font-family:'Outfit',sans-serif;font-size:1.35rem;font-weight:800;color:#1A202C;display:flex;align-items:center;gap:0.5rem;">
        <span>📊</span> Kesimpulan & Rangkuman Komparasi (4 Temuan Terpadu)
      </div>
      <div style="font-size:0.88rem;color:#64748B;margin-top:0.2rem;">
        Sintesis komparatif otomatis pandangan fikih NU dan Muhammadiyah dari 4 fatwa paling relevan
      </div>
    </div>
    """)
    render_html(sintesis_html)

    # ── 3. TAMPILKAN FATWA TERKAIT LAINNYA (PERINGKAT #2, #3, DST) ──
    if len(top_results) > 1:
        render_html(f"""
        <div id="fatwa-lainnya-anchor"></div>
        <div style="margin-top:2.5rem;margin-bottom:1rem;">
          <div style="font-family:'Outfit',sans-serif;font-size:1.35rem;font-weight:800;color:#1A202C;display:flex;align-items:center;gap:0.5rem;">
            <span>📚</span> Rekomendasi Fatwa Terkait Lainnya
          </div>
          <div style="font-size:0.88rem;color:#64748B;margin-top:0.2rem;">
            Pembahasan fiqih lain yang juga berhubungan dengan pertanyaan Anda
          </div>
        </div>
        """)

        for rank, (ip, rrf_score) in enumerate(top_results[1:], 1):
            row      = df.iloc[ip]
            topik    = str(row.get(cols["top"], "")) if cols["top"] else ""
            kategori = str(row.get(cols["kat"], "")) if cols["kat"] else ""
            j_nu     = str(row.get(cols["nu"],  "")) if cols["nu"]  else ""
            j_mu     = str(row.get(cols["mu"],  "")) if cols["mu"]  else ""
            url_nu   = str(row.get(cols["url_nu"], "")) if cols["url_nu"] else ""
            url_mu   = str(row.get(cols["url_mu"], "")) if cols["url_mu"] else ""

            hl_nu, hw_nu, cat_nu = deteksi_hukum(j_nu)
            hl_mu, hw_mu, cat_mu = deteksi_hukum(j_mu)
            badge_nu = f'<span class="badge-hukum" style="background:{hw_nu}22;color:{hw_nu};border:1px solid {hw_nu}55">{hl_nu}</span>' if hl_nu else ""
            badge_mu = f'<span class="badge-hukum" style="background:{hw_mu}22;color:{hw_mu};border:1px solid {hw_mu}55">{hl_mu}</span>' if hl_mu else ""
            consensus_html = ringkasan_konsensus_hukum(hl_nu, cat_nu, hl_mu, cat_mu)

            score_pct = round(rrf_score * 100, 1) if rrf_score <= 1.0 else round(rrf_score, 1)
            medals    = ["🥇", "🥈", "🥉", "4️⃣"]
            medal     = medals[rank] if rank < len(medals) else f"#{rank+1}"
            rank_tag  = f"{medal} Peringkat #{rank+1}"
            pill_score_html = f'<span class="pill-score">⚡ Relevansi AI: {score_pct}%</span>'

            render_html(f"""
            <div class="result-box">
              <div class="result-rank-tag">{rank_tag}</div>
              <div class="result-title-text">{html_lib.escape(topik)}</div>
              <div class="result-meta-pills">
                <span class="pill-kat">🕌 Kategori: {html_lib.escape(kategori)}</span>
                {pill_score_html}
              </div>
              {consensus_html}
            </div>
            """)

            j_nu_prev = j_nu[:650] + ("…" if len(j_nu) > 650 else "")
            j_mu_prev = j_mu[:650] + ("…" if len(j_mu) > 650 else "")
            hl_nu_text = highlight(j_nu_prev, q_last)
            hl_mu_text = highlight(j_mu_prev, q_last)

            col_nu, col_mu = st.columns(2, gap="medium")
            with col_nu:
                render_html(f"""
                <div class="jawaban-card nu">
                  <div class="jawaban-header nu">
                    <span>🟢 Nahdlatul Ulama</span>
                    {badge_nu}
                  </div>
                  <div class="jawaban-body">{hl_nu_text}</div>
                </div>
                """)
                if len(j_nu) > 650:
                    with st.expander(f"📖 Baca Selengkapnya (NU) - {rank_tag}"):
                        render_html(f'<div class="jawaban-body">{highlight(j_nu, q_last)}</div>')
                if url_nu.startswith("http"):
                    st.markdown(f"[🔗 Buka Sumber Resmi NU]({url_nu})", unsafe_allow_html=False)

            with col_mu:
                render_html(f"""
                <div class="jawaban-card mu">
                  <div class="jawaban-header mu">
                    <span>🔵 Muhammadiyah</span>
                    {badge_mu}
                  </div>
                  <div class="jawaban-body">{hl_mu_text}</div>
                </div>
                """)
                if len(j_mu) > 650:
                    with st.expander(f"📖 Baca Selengkapnya (Muhammadiyah) - {rank_tag}"):
                        render_html(f'<div class="jawaban-body">{highlight(j_mu, q_last)}</div>')
                if url_mu.startswith("http"):
                    st.markdown(f"[🔗 Buka Sumber Resmi Muhammadiyah]({url_mu})", unsafe_allow_html=False)

            fb_key = f"fb_{ip}_{rank}"
            if fb_key not in st.session_state:
                st.session_state[fb_key] = None
            fb_c1, fb_c2, fb_sp = st.columns([1, 1, 6])
            with fb_c1:
                if st.button("👍 Bermanfaat", key=f"pos_{ip}_{rank}"):
                    st.session_state[fb_key] = "helpful"
                    log_feedback(q_last, topik, "helpful")
            with fb_c2:
                if st.button("👎 Kurang", key=f"neg_{ip}_{rank}"):
                    st.session_state[fb_key] = "not_helpful"
                    log_feedback(q_last, topik, "not_helpful")

            if st.session_state[fb_key] == "helpful":
                st.caption("✅ Terima kasih atas masukan Anda!")
            elif st.session_state[fb_key] == "not_helpful":
                st.caption("📝 Masukan tercatat untuk peningkatan model.")

            render_html("<hr style='border-color:#EAE5D8;margin:1.5rem 0;'>")

    if st.button("← Kembali ke Beranda", key="btn_back_home", type="primary"):
        st.session_state.pop("hasil_hybrid", None)
        st.session_state["quick_query"] = ""
        st.rerun()

else:
    # =========================================================
    # HOMEPAGE CONTENT
    # =========================================================

    # Feature mini cards
    render_html("""
    <div class="section-heading">🌟 Fitur Unggulan Quranica</div>
    <div class="section-caption">Teknologi pencarian fikih generasi baru yang cepat, transparan, dan mudah dipahami</div>
    <div class="section-accent-line line-emerald"></div>
    <div class="feature-box-grid">
      <div class="feature-mini-card">
        <div class="feature-mini-icon">🤖</div>
        <div class="feature-mini-title">AI Hybrid Search</div>
        <div class="feature-mini-desc">BM25 leksikal + Indo S-BERT semantik dengan Weighted RRF fusion</div>
      </div>
      <div class="feature-mini-card">
        <div class="feature-mini-icon">⚖️</div>
        <div class="feature-mini-title">Komparasi 2 Ormas</div>
        <div class="feature-mini-desc">Perbandingan fatwa NU dan Muhammadiyah langsung berdampingan</div>
      </div>
      <div class="feature-mini-card">
        <div class="feature-mini-icon">📚</div>
        <div class="feature-mini-title">500+ Dokumen</div>
        <div class="feature-mini-desc">Database ibadah komprehensif dari sumber kredibel terverifikasi</div>
      </div>
      <div class="feature-mini-card">
        <div class="feature-mini-icon">🏷️</div>
        <div class="feature-mini-title">Deteksi Hukum</div>
        <div class="feature-mini-desc">Status hukum (Wajib, Sunnah, Boleh, Makruh, Haram) otomatis terdeteksi</div>
      </div>
      <div class="feature-mini-card">
        <div class="feature-mini-icon">🔍</div>
        <div class="feature-mini-title">Query Expansion</div>
        <div class="feature-mini-desc">Mengenali sinonim fiqih otomatis: shalat/salat/sholat, wudhu/wudlu</div>
      </div>
    </div>
    """)

    # Trivia Section - Tahukah Kamu?
    render_html("""
    <div class="section-heading">💡 Tahukah Kamu? (Tanya Jawab Fikih Populer)</div>
    <div class="section-caption">Pertanyaan fikih praktis sehari-hari yang sering menjadi keraguan umat — klik tombol untuk langsung membuka fatwanya</div>
    <div class="section-accent-line line-amber"></div>
    """)

    triv_cols = st.columns(len(TRIVIA_FIKIH))
    for ti, triv in enumerate(TRIVIA_FIKIH):
        with triv_cols[ti]:
            render_html(f"""
            <div class="trivia-box">
              <div class="trivia-box-q">❓ {triv['q']}</div>
              <div class="trivia-box-hint">{triv['hint']}</div>
            </div>
            """)
            if st.button(f"🔍 Buka Fatwa #{ti+1}", key=f"triv_action_{ti}", use_container_width=True):
                st.session_state["hasil_hybrid"] = [(triv["doc_idx"], 1.0)]
                st.session_state["query_terakhir"] = triv["title"]
                st.session_state["quick_query"] = ""
                st.rerun()

    # Fatwa Recommendations
    render_html("""
    <div class="section-heading" style="margin-top:2.5rem">✨ Rekomendasi Fatwa Pilihan</div>
    <div class="section-caption">Pilihan fatwa komparatif yang paling sering dicari dan dipelajari &mdash; klik tombol untuk langsung membaca perbandingan fatwa</div>
    <div class="section-accent-line line-blue"></div>
    """)

    # Category filter pills
    cat_filter = st.pills(
        "Kategori",
        options=["Semua", "Sholat", "Thaharah", "Puasa", "Zakat", "Haji"],
        default="Semua",
        label_visibility="collapsed"
    )

    filtered_recs = [
        f for f in FATWA_RECS
        if cat_filter == "Semua" or f["category"].lower() == cat_filter.lower()
    ]

    rec_cols = st.columns(3)
    for idx, f in enumerate(filtered_recs):
        with rec_cols[idx % 3]:
            render_html(f"""
            <div class="fatwa-box">
              <div class="fatwa-badge-row">
                <span class="fatwa-cat-tag">{f['icon']} {f['category']}</span>
                <span class="fatwa-org-tag">{f['org']}</span>
              </div>
              <div class="fatwa-card-title">{f['title']}</div>
              <div class="fatwa-card-snippet">{f['snippet']}</div>
            </div>
            """)
            if st.button("🔍 Telusuri Fatwa Ini", key=f"f_act_{f['category']}_{idx}", use_container_width=True):
                st.session_state["hasil_hybrid"] = [(f["doc_idx"], 1.0)]
                st.session_state["query_terakhir"] = f["title"]
                st.session_state["quick_query"] = ""
                st.rerun()


# =============================================================
# FOOTER
# =============================================================
render_html("""
<div class="site-footer-bar">
  <div style="font-size:1.6rem;margin-bottom:0.4rem;">🕌</div>
  <strong>Quranica Unimal</strong> &mdash; Sistem Penelusuran Fatwa Fikih Berbasis AI
  <br>
  <span style="font-size:0.75rem;opacity:0.8;">
    Universitas Malikussaleh &nbsp;·&nbsp; Hybrid Retrieval: BM25 + Indo S-BERT &nbsp;·&nbsp; Reciprocal Rank Fusion
  </span>
</div>
""")
