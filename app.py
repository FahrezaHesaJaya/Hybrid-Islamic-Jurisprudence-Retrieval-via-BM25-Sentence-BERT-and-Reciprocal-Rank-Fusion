# =============================================================
# app.py — Quranica Unimal
# Fikih Hybrid Search: BM25 + Indo S-BERT + Weighted RRF
# Premium UI — Clean Google Style with Quranica Theme
# =============================================================

import os, csv, re, pickle, html as html_lib, urllib.parse, requests, base64, random, textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH    = os.path.join(BASE_DIR, "Data_ubudiyah_final.csv")
PARQUET_PATH    = os.path.join(BASE_DIR, "Data_ubudiyah_final.parquet")
BM25_CACHE_PATH = os.path.join(BASE_DIR, "bm25_model.pkl")
EMB_CACHE_PATH  = os.path.join(BASE_DIR, "corpus_emb.npy")
LOG_SEARCH      = os.path.join(BASE_DIR, "log_pencarian.csv")
LOG_FEEDBACK    = os.path.join(BASE_DIR, "log_feedback.csv")
HERO_IMG_PATH   = os.path.join(BASE_DIR, "assets", "quranica_hero.jpg")

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Quranica — Pencarian Fikih Islam",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Helper for Raw HTML (Guarantees zero indentation) ─────────
def render_html(html_str: str):
    """
    Renders HTML safely in Streamlit by stripping all leading whitespace from every line.
    This strictly prevents Markdown from treating lines with 4+ spaces as <pre><code> blocks.
    """
    clean_lines = [line.lstrip() for line in html_str.splitlines()]
    clean_html = "\n".join(clean_lines).strip()
    st.markdown(clean_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# CSS — QURANICA WARM ISLAMIC GOOGLE THEME
# ═══════════════════════════════════════════════════════════════
render_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@600;700;800;900&family=Nunito:wght@400;600;700;800;900&display=swap');

/* ─── Global Base & Streamlit Overrides ─── */
html, body, [data-testid="stAppViewContainer"], .stApp {
background-color: #FAF7F0 !important;
font-family: 'Nunito', sans-serif !important;
color: #1A202C !important;
}

[data-testid="stHeader"] {
background: transparent !important;
}

[data-testid="stSidebar"], section[data-testid="stSidebar"] {
display: none !important;
}

div[data-testid="stToolbar"] {
display: none !important;
}

footer, #MainMenu {
display: none !important;
}

.main .block-container {
max-width: 1040px !important;
padding: 1rem 1.5rem 4rem 1.5rem !important;
margin: 0 auto !important;
}

/* ─── Top Nav Bar ─── */
.top-nav {
background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 100%);
border-radius: 16px;
padding: 0.75rem 1.5rem;
display: flex;
align-items: center;
justify-content: space-between;
box-shadow: 0 4px 18px rgba(27,67,50,0.18);
margin-bottom: 1.25rem;
}

.top-nav-brand {
display: flex;
align-items: center;
gap: 0.6rem;
color: #FFFFFF;
font-weight: 900;
font-size: 1.2rem;
letter-spacing: -0.01em;
}

.top-nav-brand .q-accent {
color: #F4B400;
font-size: 1.4rem;
}

.top-nav-info {
color: rgba(255,255,255,0.85);
font-size: 0.8rem;
font-weight: 700;
}

/* ─── Prayer Time Schedule Bar ─── */
.prayer-bar-wrap {
background: #FFFFFF;
border: 1.5px solid #EAE5D8;
border-radius: 18px;
padding: 0.85rem 1.25rem;
box-shadow: 0 3px 12px rgba(45,106,79,0.06);
margin-bottom: 2rem;
display: flex;
flex-wrap: wrap;
align-items: center;
justify-content: space-between;
gap: 0.75rem;
}

.prayer-bar-header {
display: flex;
align-items: center;
gap: 0.4rem;
font-size: 0.82rem;
font-weight: 800;
color: #2D6A4F;
text-transform: uppercase;
letter-spacing: 0.05em;
white-space: nowrap;
}

.prayer-cards-group {
display: flex;
align-items: center;
gap: 0.5rem;
flex-wrap: wrap;
justify-content: center;
}

.prayer-chip {
background: #FAF7F0;
border: 1.5px solid #EAE5D8;
border-radius: 12px;
padding: 0.4rem 0.85rem;
text-align: center;
min-width: 78px;
transition: all 0.2s ease;
position: relative;
}

.prayer-chip.active {
background: linear-gradient(135deg, #2D6A4F 0%, #1B4332 100%) !important;
border-color: #1B4332 !important;
box-shadow: 0 4px 12px rgba(45,106,79,0.28);
transform: translateY(-2px);
}

.prayer-chip.active .p-name,
.prayer-chip.active .p-time {
color: #FFFFFF !important;
}

.prayer-chip.active .p-badge {
display: block;
}

.p-badge {
display: none;
position: absolute;
top: -8px;
right: -6px;
background: #F4B400;
color: #1A202C;
font-size: 0.55rem;
font-weight: 800;
padding: 0.1rem 0.35rem;
border-radius: 10px;
box-shadow: 0 2px 5px rgba(0,0,0,0.15);
}

.p-name {
font-size: 0.68rem;
font-weight: 800;
text-transform: uppercase;
letter-spacing: 0.04em;
color: #64748B;
margin-bottom: 0.1rem;
}

.p-time {
font-size: 0.95rem;
font-weight: 800;
color: #1A202C;
line-height: 1.1;
}

.prayer-loc-date {
font-size: 0.75rem;
color: #64748B;
font-weight: 700;
white-space: nowrap;
}

/* ─── Hero Section ─── */
.hero-container {
width: 100% !important;
display: flex !important;
flex-direction: column !important;
align-items: center !important;
justify-content: center !important;
text-align: center !important;
margin: 0 auto !important;
padding: 0.5rem 0 1.25rem 0 !important;
}

.hero-banner-frame {
display: block !important;
margin: 0 auto 1.2rem auto !important;
max-width: 540px !important;
width: 100% !important;
text-align: center !important;
}

.hero-banner-image {
display: block !important;
margin: 0 auto !important;
width: 100% !important;
max-width: 540px !important;
height: auto !important;
border-radius: 20px !important;
border: none !important;
box-shadow: 0 10px 30px rgba(45,106,79,0.15), 0 2px 8px rgba(0,0,0,0.05) !important;
transition: transform 0.25s ease, box-shadow 0.25s ease !important;
}

.hero-banner-image:hover {
transform: translateY(-2px) scale(1.008) !important;
box-shadow: 0 16px 38px rgba(45,106,79,0.22) !important;
}

.brand-title {
font-family: 'Outfit', sans-serif !important;
font-size: 3.9rem;
font-weight: 900;
line-height: 1.1;
margin: 0.2rem auto 0.5rem auto !important;
text-align: center !important;
letter-spacing: -0.015em;
filter: drop-shadow(0 4px 10px rgba(0,0,0,0.06));
display: inline-block !important;
}

.brand-title span {
display: inline-block;
transition: transform 0.2s ease, filter 0.2s ease;
cursor: default;
}

.brand-title span:hover {
transform: translateY(-4px) scale(1.08);
filter: drop-shadow(0 6px 14px rgba(0,0,0,0.12));
}

.q-Q  { color: #F4B400; }
.q-u  { color: #4285F4; }
.q-r  { color: #EA4335; }
.q-a  { color: #34A853; }
.q-n  { color: #FBBC05; }
.q-i  { color: #0F9D58; }
.q-c  { color: #1A73E8; }
.q-aa { color: #EA4335; }

.subtitle-badge {
display: inline-flex !important;
align-items: center !important;
justify-content: center !important;
gap: 0.55rem;
flex-wrap: wrap;
background: #FFFFFF;
border: 1.5px solid #EAE5D8;
border-radius: 50px;
padding: 0.5rem 1.4rem;
box-shadow: 0 2px 8px rgba(0,0,0,0.03);
font-size: 0.88rem;
font-weight: 700;
color: #475569;
margin: 0.2rem auto 1.4rem auto !important;
text-align: center !important;
}

.subtitle-badge .sub-lead {
color: #1E293B;
font-weight: 800;
}

.subtitle-badge .sub-sep {
color: #CBD5E1;
font-weight: 900;
font-size: 0.9rem;
}

.subtitle-badge .sub-and {
color: #94A3B8;
font-weight: 600;
}

.subtitle-badge .org-nu {
color: #2D6A4F;
font-weight: 800;
background: #E8F5E9;
padding: 0.15rem 0.6rem;
border-radius: 20px;
}

.subtitle-badge .org-mu {
color: #1A73E8;
font-weight: 800;
background: #E8F0FE;
padding: 0.15rem 0.6rem;
border-radius: 20px;
}

.subtitle-badge .org-unimal {
color: #EA4335;
font-weight: 800;
}

/* Stats Bar */
.stats-pills-row {
display: flex;
justify-content: center;
gap: 0.75rem;
flex-wrap: wrap;
margin-bottom: 1.5rem;
}

.stat-chip {
background: #FFFFFF;
border: 1.5px solid #EAE5D8;
border-radius: 50px;
padding: 0.4rem 1.1rem;
display: flex;
align-items: center;
gap: 0.5rem;
font-size: 0.78rem;
font-weight: 800;
color: #2D6A4F;
box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}

/* ─── Google Search Box & Buttons ─── */
div[data-testid="stTextInput"] input {
background: #FFFFFF !important;
border: 2px solid #E2E8F0 !important;
border-radius: 50px !important;
color: #1A202C !important;
padding: 0.95rem 1.8rem !important;
font-size: 1.02rem !important;
font-family: 'Nunito', sans-serif !important;
box-shadow: 0 4px 18px rgba(0,0,0,0.06) !important;
transition: all 0.2s ease !important;
}

div[data-testid="stTextInput"] input:focus {
border-color: #2D6A4F !important;
box-shadow: 0 4px 24px rgba(45,106,79,0.18) !important;
outline: none !important;
}

div[data-testid="stTextInput"] input::placeholder {
color: #94A3B8 !important;
}

/* Primary search button */
div[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"],
div[data-testid="stFormSubmitButton"] button[kind="primary"] {
background: linear-gradient(135deg, #2D6A4F 0%, #1B4332 100%) !important;
color: #FFFFFF !important;
border: none !important;
border-radius: 50px !important;
font-weight: 800 !important;
font-size: 0.95rem !important;
padding: 0.7rem 1.6rem !important;
box-shadow: 0 4px 14px rgba(45,106,79,0.3) !important;
transition: transform 0.15s ease, box-shadow 0.15s ease !important;
font-family: 'Nunito', sans-serif !important;
}

div[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"]:hover,
div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
transform: translateY(-2px) !important;
box-shadow: 0 6px 20px rgba(45,106,79,0.4) !important;
}

/* Secondary (lucky) button */
div[data-testid="stFormSubmitButton"] button:not([kind="primary"]):not([kind="primaryFormSubmit"]) {
background: #FFFFFF !important;
color: #2D6A4F !important;
border: 2px solid #CBD5E1 !important;
border-radius: 50px !important;
font-weight: 800 !important;
font-size: 0.95rem !important;
padding: 0.7rem 1.6rem !important;
box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
transition: all 0.15s ease !important;
font-family: 'Nunito', sans-serif !important;
}

div[data-testid="stFormSubmitButton"] button:not([kind="primary"]):not([kind="primaryFormSubmit"]):hover {
background: #F8FAFC !important;
color: #1B4332 !important;
border-color: #2D6A4F !important;
transform: translateY(-2px) !important;
}

/* General buttons */
div[data-testid="stButton"] button {
background: #FFFFFF !important;
border: 1.5px solid #E2E8F0 !important;
color: #2D6A4F !important;
border-radius: 14px !important;
font-size: 0.85rem !important;
font-weight: 800 !important;
padding: 0.45rem 1rem !important;
transition: all 0.15s ease !important;
box-shadow: 0 2px 6px rgba(0,0,0,0.03) !important;
font-family: 'Nunito', sans-serif !important;
}

div[data-testid="stButton"] button:hover {
background: #2D6A4F !important;
color: #FFFFFF !important;
border-color: #2D6A4F !important;
transform: translateY(-2px) !important;
box-shadow: 0 4px 12px rgba(45,106,79,0.2) !important;
}

/* ─── Section Headers ─── */
.section-heading {
font-size: 1.25rem;
font-weight: 900;
color: #1A202C;
margin: 2.2rem 0 0.25rem 0;
display: flex;
align-items: center;
gap: 0.4rem;
}

.section-caption {
font-size: 0.82rem;
color: #64748B;
font-weight: 600;
margin-bottom: 0.75rem;
}

.section-accent-line {
width: 48px;
height: 3.5px;
border-radius: 4px;
margin-bottom: 1.25rem;
}

.line-emerald { background: linear-gradient(90deg, #2D6A4F, #52B788); }
.line-amber   { background: linear-gradient(90deg, #F4B400, #F59E0B); }
.line-blue    { background: linear-gradient(90deg, #4285F4, #1A73E8); }

/* ─── Trivia Card Box ─── */
.trivia-box {
background: #FFFFFF;
border-radius: 16px;
padding: 1.1rem 1.25rem;
border: 1.5px solid #EAE5D8;
border-top: 4px solid #F4B400;
box-shadow: 0 4px 14px rgba(0,0,0,0.04);
height: 100%;
display: flex;
flex-direction: column;
justify-content: space-between;
}

.trivia-box-q {
font-size: 0.88rem;
font-weight: 800;
color: #1A202C;
line-height: 1.4;
margin-bottom: 0.4rem;
}

.trivia-box-hint {
font-size: 0.78rem;
color: #64748B;
line-height: 1.5;
margin-bottom: 0.6rem;
}

/* ─── Fatwa Recommendation Card ─── */
.fatwa-box {
background: #FFFFFF;
border-radius: 16px;
padding: 1.2rem 1.25rem;
border: 1.5px solid #EAE5D8;
border-top: 4px solid #2D6A4F;
box-shadow: 0 4px 14px rgba(0,0,0,0.04);
transition: transform 0.2s ease, box-shadow 0.2s ease;
height: 100%;
}

.fatwa-box:hover {
transform: translateY(-3px);
box-shadow: 0 8px 24px rgba(45,106,79,0.1);
}

.fatwa-badge-row {
display: flex;
align-items: center;
justify-content: space-between;
margin-bottom: 0.6rem;
}

.fatwa-cat-tag {
background: #E8F5E9;
color: #2D6A4F;
font-size: 0.68rem;
font-weight: 800;
padding: 0.2rem 0.65rem;
border-radius: 50px;
text-transform: uppercase;
letter-spacing: 0.04em;
}

.fatwa-org-tag {
background: #FEF3C7;
color: #92400E;
font-size: 0.65rem;
font-weight: 800;
padding: 0.15rem 0.55rem;
border-radius: 50px;
border: 1px solid #FDE68A;
}

.fatwa-card-title {
font-size: 0.95rem;
font-weight: 800;
color: #1A202C;
line-height: 1.4;
margin-bottom: 0.45rem;
}

.fatwa-card-snippet {
font-size: 0.8rem;
color: #64748B;
line-height: 1.55;
margin-bottom: 0.5rem;
display: -webkit-box;
-webkit-line-clamp: 2;
-webkit-box-orient: vertical;
overflow: hidden;
}

/* ─── Search Results Cards ─── */
.result-box {
background: #FFFFFF;
border: 1.5px solid #EAE5D8;
border-top: 4px solid #2D6A4F;
border-radius: 18px;
padding: 1.4rem 1.6rem;
margin-bottom: 1.25rem;
box-shadow: 0 4px 16px rgba(0,0,0,0.05);
}

.result-rank-tag {
display: inline-flex;
align-items: center;
gap: 0.35rem;
background: #E8F5E9;
border: 1px solid #A7F3D0;
border-radius: 50px;
padding: 0.2rem 0.75rem;
font-size: 0.72rem;
font-weight: 800;
color: #1B4332;
text-transform: uppercase;
margin-bottom: 0.45rem;
}

.result-title-text {
font-size: 1.15rem;
font-weight: 900;
color: #1A202C;
margin: 0.2rem 0 0.5rem 0;
line-height: 1.35;
}

.result-meta-pills {
display: flex;
align-items: center;
gap: 0.5rem;
flex-wrap: wrap;
margin-bottom: 0.9rem;
}

.pill-kat {
background: #F1F5F9;
color: #475569;
font-size: 0.7rem;
font-weight: 800;
padding: 0.2rem 0.65rem;
border-radius: 50px;
}

.pill-score {
background: #FEF3C7;
color: #B45309;
font-size: 0.7rem;
font-weight: 800;
padding: 0.2rem 0.65rem;
border-radius: 50px;
border: 1px solid #FDE68A;
}

/* ─── Consensus / Khilafiyah Banner ─── */
.hukum-consensus-box {
display: flex;
align-items: center;
gap: 0.75rem;
border-radius: 12px;
padding: 0.75rem 1.1rem;
margin-top: 0.65rem;
font-size: 0.86rem;
line-height: 1.5;
}

.hukum-consensus-box.consensus-agree {
background: #F0FDF4;
border: 1.5px solid #BBF7D0;
border-left: 5px solid #10B981;
color: #14532D;
}

.hukum-consensus-box.consensus-differ {
background: #FFFBEB;
border: 1.5px solid #FDE68A;
border-left: 5px solid #F59E0B;
color: #78350F;
}

/* ─── Executive Fiqh Synthesis Box ─── */
.sintesis-box {
background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
border: 1.5px solid #CBD5E1;
border-top: 4.5px solid #0D9488;
border-radius: 18px;
padding: 1.35rem 1.6rem;
margin-bottom: 1.8rem;
box-shadow: 0 4px 18px rgba(45,106,79,0.06);
}

.sintesis-header {
display: flex;
align-items: center;
justify-content: space-between;
flex-wrap: wrap;
gap: 0.8rem;
margin-bottom: 0.85rem;
padding-bottom: 0.75rem;
border-bottom: 1px dashed #CBD5E1;
}

.sintesis-title {
display: flex;
align-items: center;
gap: 0.5rem;
font-family: 'Outfit', sans-serif;
font-size: 1.1rem;
font-weight: 900;
color: #0F172A;
letter-spacing: -0.01em;
}

.sintesis-stats-badge {
display: flex;
align-items: center;
gap: 0.5rem;
flex-wrap: wrap;
}

.badge-agree-pill {
background: #DCFCE7;
color: #166534;
border: 1px solid #86EFAC;
padding: 0.25rem 0.8rem;
border-radius: 50px;
font-size: 0.76rem;
font-weight: 800;
}

.badge-differ-pill {
background: #FEF3C7;
color: #92400E;
border: 1px solid #FCD34D;
padding: 0.25rem 0.8rem;
border-radius: 50px;
font-size: 0.76rem;
font-weight: 800;
}

.sintesis-badge-agree {
background: #DCFCE7;
color: #166534;
border: 1px solid #86EFAC;
padding: 0.2rem 0.65rem;
border-radius: 50px;
font-size: 0.72rem;
font-weight: 800;
white-space: nowrap;
}

.sintesis-badge-differ {
background: #FEF3C7;
color: #92400E;
border: 1px solid #FCD34D;
padding: 0.2rem 0.65rem;
border-radius: 50px;
font-size: 0.72rem;
font-weight: 800;
white-space: nowrap;
}

.sintesis-overview-lead {
font-size: 0.88rem;
color: #475569;
line-height: 1.55;
margin-bottom: 1.1rem;
}

.sintesis-overview-lead {
font-size: 0.9rem;
color: #334155;
line-height: 1.6;
margin-bottom: 1.15rem;
}

.sintesis-dual-box {
display: grid;
grid-template-columns: 1fr 1fr;
gap: 1rem;
margin-bottom: 1.15rem;
}

@media (max-width: 768px) {
.sintesis-dual-box {
grid-template-columns: 1fr;
}
}

.sintesis-dual-panel {
background: #FFFFFF;
border: 1.5px solid #E2E8F0;
border-radius: 12px;
padding: 1.05rem 1.2rem;
display: flex;
flex-direction: column;
gap: 0.4rem;
box-shadow: 0 2px 8px rgba(0,0,0,0.02);
}

.sintesis-dual-panel.nu {
border-left: 4.5px solid #2D6A4F;
background: linear-gradient(180deg, #FFFFFF 0%, #F4FBF7 100%);
}

.sintesis-dual-panel.mu {
border-left: 4.5px solid #1A73E8;
background: linear-gradient(180deg, #FFFFFF 0%, #F4F8FD 100%);
}

.sintesis-dual-head {
font-size: 0.84rem;
font-weight: 900;
text-transform: uppercase;
letter-spacing: 0.04em;
margin-bottom: 0.25rem;
display: flex;
align-items: center;
gap: 0.4rem;
}

.sintesis-dual-head.nu { color: #2D6A4F; }
.sintesis-dual-head.mu { color: #1A73E8; }

.sintesis-dual-body {
font-size: 0.88rem;
color: #1E293B;
line-height: 1.65;
}

.sintesis-poin-block {
background: #FAF7F0;
border: 1.5px solid #EAE5D8;
border-radius: 14px;
padding: 1.05rem 1.2rem;
margin-bottom: 1.15rem;
}

.sintesis-poin-title {
font-family: 'Outfit', sans-serif;
font-size: 0.92rem;
font-weight: 800;
color: #1E293B;
margin-bottom: 0.75rem;
display: flex;
align-items: center;
gap: 0.45rem;
}

.sintesis-poin-list {
display: flex;
flex-direction: column;
gap: 0.65rem;
}

.sintesis-poin-item {
display: flex;
align-items: flex-start;
gap: 0.75rem;
background: #FFFFFF;
border: 1px solid #E2E8F0;
border-radius: 10px;
padding: 0.75rem 0.95rem;
font-size: 0.85rem;
line-height: 1.5;
transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.sintesis-poin-item:hover {
transform: translateY(-1px);
box-shadow: 0 4px 10px rgba(0,0,0,0.04);
}

.sintesis-poin-badge {
background: #1B4332;
color: #FFFFFF;
font-size: 0.7rem;
font-weight: 900;
border-radius: 6px;
padding: 0.2rem 0.5rem;
flex-shrink: 0;
margin-top: 0.1rem;
letter-spacing: 0.02em;
}

.sintesis-footer {
font-size: 0.88rem;
color: #0F766E;
background: #F0FDFA;
border: 1.5px solid #CCFBF1;
border-radius: 12px;
padding: 0.9rem 1.25rem;
line-height: 1.6;
}

/* Comparative Answer Cards */
.jawaban-card {
background: #FAF7F0;
border-radius: 14px;
padding: 1.1rem 1.25rem;
height: 100%;
border: 1px solid #EAE5D8;
border-left: 4.5px solid #2D6A4F;
}

.jawaban-card.mu {
border-left-color: #1A73E8;
}

.jawaban-header {
font-size: 0.78rem;
font-weight: 900;
text-transform: uppercase;
letter-spacing: 0.05em;
margin-bottom: 0.5rem;
display: flex;
align-items: center;
justify-content: space-between;
flex-wrap: wrap;
gap: 0.35rem;
}

.jawaban-header.nu { color: #2D6A4F; }
.jawaban-header.mu { color: #1A73E8; }

.jawaban-body {
font-size: 0.88rem;
line-height: 1.7;
color: #334155;
}

.badge-hukum {
display: inline-block;
font-size: 0.68rem;
font-weight: 800;
border-radius: 20px;
padding: 0.15rem 0.6rem;
}

mark.hl {
background: #FDE68A;
color: #92400E;
border-radius: 3px;
padding: 0 3px;
font-weight: 800;
}

/* Feature grid */
.feature-box-grid {
display: grid;
grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
gap: 0.85rem;
margin-bottom: 1.5rem;
}

.feature-mini-card {
background: #FFFFFF;
border: 1.5px solid #EAE5D8;
border-radius: 16px;
padding: 1.1rem 1rem;
text-align: center;
box-shadow: 0 2px 8px rgba(0,0,0,0.03);
transition: transform 0.2s ease;
}

.feature-mini-card:hover {
transform: translateY(-2px);
box-shadow: 0 6px 16px rgba(45,106,79,0.08);
}

.feature-mini-icon { font-size: 1.8rem; margin-bottom: 0.35rem; }
.feature-mini-title { font-size: 0.85rem; font-weight: 800; color: #1A202C; margin-bottom: 0.2rem; }
.feature-mini-desc { font-size: 0.72rem; color: #64748B; line-height: 1.45; }

/* Footer */
.site-footer-bar {
background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 100%);
color: rgba(255,255,255,0.85);
border-radius: 18px;
text-align: center;
padding: 2rem 1.5rem;
margin-top: 3rem;
font-size: 0.82rem;
box-shadow: 0 4px 18px rgba(27,67,50,0.15);
}

.site-footer-bar strong { color: #F4B400; }

/* ─── Print Friendly Styling ─── */
@media print {
  [data-testid="stSidebar"],
  header,
  footer,
  .stDeployButton,
  .stChatFloatingInputContainer,
  button,
  .unmatched-feedback-box,
  .stExpander,
  .feature-box-grid,
  .section-accent-line {
    display: none !important;
  }
  .main .block-container {
    max-width: 100% !important;
    padding: 1rem !important;
    margin: 0 !important;
  }
  body, .stApp {
    background: #FFFFFF !important;
    color: #111827 !important;
    font-size: 11pt !important;
  }
  .result-box, .jawaban-card {
    border: 1.5px solid #CBD5E1 !important;
    box-shadow: none !important;
    page-break-inside: avoid;
    background: #FFFFFF !important;
    margin-bottom: 1.5rem !important;
  }
  .result-rank-tag {
    background: #2D6A4F !important;
    color: #FFFFFF !important;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
}
</style>
""")


# =============================================================
# HELPER FUNCTIONS & IMAGE LOADER
# =============================================================
@st.cache_data(show_spinner=False)
def get_hero_image_b64():
    """Loads and caches hero banner image as base64 string."""
    if os.path.exists(HERO_IMG_PATH):
        try:
            with open(HERO_IMG_PATH, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            pass
    return ""


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prayer_times(city="Lhokseumawe", country="ID"):
    """Fetches real prayer times from Aladhan API or returns defaults."""
    try:
        today = datetime.now()
        url = (f"https://api.aladhan.com/v1/timingsByCity/{today.day}-{today.month}-{today.year}"
               f"?city={city}&country={country}&method=11")
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            t = r.json()["data"]["timings"]
            return {
                "Subuh":   t.get("Fajr",    "05:12"),
                "Dzuhur":  t.get("Dhuhr",   "12:24"),
                "Ashar":   t.get("Asr",     "15:36"),
                "Maghrib": t.get("Maghrib", "18:21"),
                "Isya":    t.get("Isha",    "19:33"),
            }
    except Exception:
        pass
    return {"Subuh": "05:12", "Dzuhur": "12:24", "Ashar": "15:36", "Maghrib": "18:21", "Isya": "19:33"}


def get_next_prayer(times):
    now_min = datetime.now().hour * 60 + datetime.now().minute
    for name in ["Subuh", "Dzuhur", "Ashar", "Maghrib", "Isya"]:
        try:
            h, m = map(int, times[name].split(":"))
            if now_min < h * 60 + m:
                return name
        except Exception:
            pass
    return "Subuh"


# =============================================================
# DATASET & ML / NLP ENGINE
# =============================================================
@st.cache_resource(show_spinner=False)
def load_sastrawi():
    if not SASTRAWI_AVAILABLE:
        return None, None
    return StemmerFactory().create_stemmer(), StopWordRemoverFactory().create_stop_word_remover()


@st.cache_resource(show_spinner="🕌 Memuat model AI Quranica Multilingual...")
def load_sbert():
    import os
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    use_cuda = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 6
    dev = "cuda" if use_cuda else "cpu"
    try:
        return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=dev)
    except Exception:
        try:
            return SentenceTransformer("firqaaa/indo-sentence-bert-base", device=dev)
        except Exception:
            return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device=dev)


SINONIM_FIQIH = {
    # ── Shalat & Sujud ──
    "sholat": ["shalat", "salat", "sembahyang", "fardhu"],
    "shalat": ["sholat", "salat", "sembahyang", "fardhu"],
    "salat":  ["shalat", "sholat", "sembahyang"],
    "jamak":  ["qashar", "safar", "musafir", "taqdim", "takhir"],
    "qashar": ["jamak", "safar", "musafir"],
    "sujud":  ["sujud sahwi", "tilawah", "syukur"],
    "rakaat": ["rokaat", "tasyahud", "tahiyat"],
    "masbuk": ["makmum masbuk", "tertinggal", "makmum"],
    "jamaah": ["berjamaah", "jemaah", "imam"],

    # ── Thaharah & Wudhu ──
    "wudhu":       ["wudlu", "wudu", "berwudhu", "thaharah", "bersuci"],
    "wudlu":       ["wudhu", "wudu", "thaharah", "bersuci"],
    "wudu":        ["wudhu", "wudlu", "bersuci"],
    "thaharah":    ["taharah", "toharoh", "bersuci", "wudhu", "tayamum"],
    "taharah":     ["thaharah", "bersuci", "wudhu"],
    "bersentuhan": ["sentuh", "sentuhan kulit", "batal wudhu", "menyentuh", "kulit", "ajnabiyyah"],
    "sentuh":      ["bersentuhan", "kulit", "menyentuh", "batal wudhu"],
    "berhubungan": ["persetubuhan", "jima", "hubungan biologis"],
    "suami":       ["suami istri", "pasangan", "istri"],
    "istri":       ["suami istri", "pasangan", "suami", "isteri"],
    "tayamum":     ["tayammum", "debu", "tanpa air", "rukun tayamum", "tanah"],
    "tayammum":    ["tayamum", "debu", "tanpa air", "rukun tayamum", "tanah"],
    "debu":        ["tayamum", "tayammum", "tanah suci"],
    "najis":       ["kotoran", "hadats", "mughallazhah", "mutawassithah", "mukhaffafah"],
    "hadas":       ["hadats", "hadas kecil", "hadas besar", "junub"],
    "hadats":      ["hadas", "junub"],
    "junub":       ["mandi wajib", "mandi besar", "hadas besar"],
    "teh":         ["air teh", "air muqayyad", "air berubah warna", "macam macam air untuk bersuci"],
    "kopi":        ["air kopi", "air muqayyad", "air berubah warna", "macam macam air untuk bersuci"],
    "galon":       ["air kemasan", "air mineral", "botol"],
    "kemasan":     ["galon", "botol", "air mineral"],

    # ── Puasa ──
    "puasa":    ["shaum", "shiyam"],
    "shaum":    ["puasa", "shiyam"],
    "ramadhan": ["ramadan", "bulan ramadhan", "puasa ramadhan"],
    "ramadan":  ["ramadhan", "bulan ramadhan", "puasa ramadhan"],
    "inhaler":  ["asma", "semprotan", "tetes mata", "obat"],
    "tetes":    ["tetes mata", "obat tetes", "inhaler"],
    "fidyah":   ["qadha", "fidyat", "bayar fidyah"],
    "qadha":    ["mengganti puasa", "fidyah"],

    # ── Thaharah Janabah & Hadas Besar ──
    "mani":     ["sperma", "janabah", "junub", "mandi wajib", "mandi besar", "hadas besar"],
    "sperma":   ["mani", "janabah", "junub", "mandi wajib"],
    "janabah":  ["junub", "mandi wajib", "hadas besar", "keluar mani"],
    "junub":    ["janabah", "mandi wajib", "mandi besar", "hadas besar"],

    # ── Zakat & Infaq ──
    "zakat":    ["zakah", "nishab", "haul", "muzakki"],
    "fitrah":   ["zakat fitrah", "beras", "makanan pokok", "uang"],
    "mal":      ["zakat mal", "zakat harta", "emas perak", "tabungan"],
    "emas":     ["zakat emas", "nisab emas", "perhiasan simpanan"],
    "perak":    ["zakat perak", "nisab perak"],
    "qris":     ["transfer", "dompet digital", "online", "bank"],
    "uang":     ["tunai", "nilai makanan", "harga beras"],

    # ── Haji & Umrah ──
    "haji":     ["manasik", "tamattu", "ifrad", "qiran", "wukuf", "tawaf"],
    "umrah":    ["umroh", "tamattu", "haji tamattu", "ihram"],
    "umroh":    ["umrah", "tamattu", "haji tamattu"],
    "tamattu":  ["umrah sebelum haji", "dam", "haji tamattu"],
    "dam":      ["denda haji", "kambing", "fidyah haji", "tamattu"],
    "tawaf":    ["thawaf", "ifadah", "wada", "qudum"],

    # ── Istilah Hukum ──
    "sunnah":   ["sunnat", "mustahab", "dianjurkan"],
    "wajib":    ["fardhu", "fardh", "harus", "rukun"],
    "fardhu":   ["wajib", "fardh"],
    "boleh":    ["mubah", "diperbolehkan", "sah", "halal", "rukhshah"],
    "haram":    ["diharamkan", "terlarang", "dosa"],
    "makruh":   ["dimakruhkan", "makrooh"],
}

SLANG_DICT = {
    # ── Ketiadaan / Negasi ──
    r"\bgd\b": "tidak ada",
    r"\bgaada\b": "tidak ada",
    r"\bga\s+ada\b": "tidak ada",
    r"\bgak\s+ada\b": "tidak ada",
    r"\bnggak\s+ada\b": "tidak ada",
    r"\bkaga\s+ada\b": "tidak ada",
    r"\bgk\b": "tidak",
    r"\bga\b": "tidak",
    r"\bgak\b": "tidak",
    r"\bnggak\b": "tidak",
    r"\bkaga\b": "tidak",
    r"\bg\b": "tidak",
    r"\byh\b": "",
    r"\byg\b": "yang",
    r"\bsih\b": "",
    r"\bdong\b": "",
    r"\btuh\b": "",
    r"\bnih\b": "",

    # ── Kata Tanya & Interogatif ──
    r"\bapakh\b": "apakah",
    r"\bapkh\b": "apakah",
    r"\bapkah\b": "apakah",
    r"\bgmna\b": "bagaimana",
    r"\bgmn\b": "bagaimana",
    r"\bgimana\b": "bagaimana",
    r"\bbgmn\b": "bagaimana",
    r"\bkek mana\b": "bagaimana",
    r"\bkayak mana\b": "bagaimana",
    r"\bcranya\b": "caranya",
    r"\bknp\b": "kenapa",
    r"\bngapa\b": "kenapa",

    # ── Waktu & Status ──
    r"\bpas\b": "ketika",
    r"\bktika\b": "ketika",
    r"\bktk\b": "ketika",
    r"\bwkt\b": "waktu",
    r"\bwktu\b": "waktu",
    r"\bsdh\b": "sudah",
    r"\bsudh\b": "sudah",
    r"\budh\b": "sudah",
    r"\budhh\b": "sudah",
    r"\budah\b": "sudah",
    r"\bblm\b": "belum",
    r"\bblom\b": "belum",
    r"\bblum\b": "belum",
    r"\bstlh\b": "setelah",
    r"\bstl\b": "setelah",
    r"\bsesdh\b": "setelah",
    r"\babis\b": "setelah",
    r"\bhbis\b": "setelah",
    r"\bsblm\b": "sebelum",
    r"\bsblmnya\b": "sebelum",
    r"\bbru\b": "baru",
    r"\btd\b": "tadi",
    r"\btrs\b": "terus",
    r"\bskrg\b": "sekarang",
    r"\bskrng\b": "sekarang",
    r"\btpi\b": "tapi",
    r"\btp\b": "tapi",
    r"\bjg\b": "juga",
    r"\bjga\b": "juga",
    r"\bsm\b": "sama",
    r"\bsma\b": "sama",
    r"\bdgn\b": "dengan",
    r"\bdg\b": "dengan",
    r"\bklo\b": "kalau",
    r"\bkalo\b": "kalau",
    r"\bklu\b": "kalau",
    r"\bkl\b": "kalau",
    r"\bbwt\b": "buat",
    r"\butk\b": "untuk",
    r"\bkrn\b": "karena",
    r"\bkarna\b": "karena",
    r"\bak\b": "saya",
    r"\bsy\b": "saya",
    r"\bkudu\b": "harus wajib",
    r"\bwjib\b": "wajib",

    # ── Kelaikan & Hukum ──
    r"\bblh\b": "boleh",
    r"\bbsa\b": "bisa",
    r"\bbs\b": "bisa",
    r"\bbal\b": "batal",
    r"\bbtl\b": "batal",
    r"\bbatl\b": "batal",
    r"\bbtal\b": "batal",
    r"\bbatalin\b": "membatalkan",
    r"\bmebatalakn\b": "membatalkan",
    r"\bmbatalkan\b": "membatalkan",

    # ── Darah & Donor ──
    r"\bdrh\b": "darah",
    r"\bdrah\b": "darah",
    r"\bdraah\b": "darah",
    r"\bdara\b": "darah",
    r"\bdnor\b": "donor",
    r"\bdnr\b": "donor",

    # ── Wudhu & Thaharah ──
    r"\bwudlu\b": "wudhu",
    r"\bwudu\b": "wudhu",
    r"\bwdhu\b": "wudhu",
    r"\bwdu\b": "wudhu",
    r"\bwuduu\b": "wudhu",
    r"\bbwudhu\b": "wudhu",
    r"\bwudhuan\b": "wudhu",
    r"\baer\b": "air",
    r"\bayr\b": "air",
    r"\bdbu\b": "debu",
    r"\btyamum\b": "tayamum",
    r"\btymum\b": "tayamum",
    r"\bnjs\b": "najis",
    r"\bnajs\b": "najis",
    r"\bhds\b": "hadas",
    r"\bjnbah\b": "janabah",
    r"\bjnb\b": "junub",
    r"\bmndi\b": "mandi",
    r"\bmnd\b": "mandi",
    r"\bmny\b": "mani",
    r"\bsprma\b": "mani",
    r"\bmdzi\b": "madzi",
    r"\bcebok\b": "istinja buang air",
    r"\bcbok\b": "istinja",
    r"\bompol\b": "air kencing",
    r"\bkencg\b": "kencing",
    r"\bkcng\b": "kencing",
    r"\bbyi\b": "bayi",
    r"\bbju\b": "baju",
    r"\bpkian\b": "pakaian",
    r"\bclna\b": "celana",
    r"\bcln\b": "celana",
    r"\btisu\b": "batu tisu",
    r"\bkna\b": "kena",
    r"\bpke\b": "pakai",
    r"\bpk\b": "pakai",

    # ── Shalat & Gerakan ──
    r"\bsholat\b": "shalat",
    r"\bsolat\b": "shalat",
    r"\bshlt\b": "shalat",
    r"\bslt\b": "shalat",
    r"\bshlat\b": "shalat",
    r"\brokaat\b": "rakaat",
    r"\brkat\b": "rakaat",
    r"\brkaat\b": "rakaat",
    r"\bsujd\b": "sujud",
    r"\bsjd\b": "sujud",
    r"\bsahwy\b": "sahwi",
    r"\bsahwii\b": "sahwi",
    r"\blpa\b": "lupa",
    r"\bgabungin\b": "jamak",
    r"\bjmak\b": "jamak",
    r"\bqasar\b": "qashar",
    r"\bqashor\b": "qashar",
    r"\bpswt\b": "pesawat",
    r"\bkrta\b": "kereta",
    r"\btrwih\b": "tarawih",
    r"\bkntut\b": "kentut",

    # ── Puasa & Medis ──
    r"\bposo\b": "puasa",
    r"\bpuaso\b": "puasa",
    r"\bpso\b": "puasa",
    r"\bmmpi\b": "mimpi",
    r"\bbsh\b": "basah",
    r"\bmta\b": "mata",
    r"\bhri\b": "hari",
    r"\btts\b": "tetes",
    r"\bmntah\b": "muntah",
    r"\btlan\b": "menelan",
    r"\bnelan\b": "menelan",
    r"\bldah\b": "ludah",
    r"\bmulu\b": "terus menerus",
    r"\bskt\b": "sakit",
    r"\bobt\b": "obat",
    r"\binheler\b": "inhaler",

    # ── Zakat & Haji ──
    r"\bbyar\b": "bayar",
    r"\bzkt\b": "zakat",
    r"\bsdkh\b": "sedekah",
    r"\bhji\b": "haji",
    r"\bumroh\b": "umrah",
    r"\bumrh\b": "umrah",
    r"\bdnda\b": "denda dam",
    r"\bsndiri\b": "sendiri",
    r"\bsndirian\b": "sendirian",
}

COMMON_ID_WORDS = {
    'saat', 'ketika', 'waktu', 'pas', 'kala', 'sewaktu', 'sedang', 'masih',
    'sudah', 'telah', 'belum', 'habis', 'setelah', 'sesudah', 'sebelum',
    'sekarang', 'nanti', 'kemarin', 'besok', 'lusa', 'hari', 'malam', 'siang',
    'pagi', 'sore', 'apakah', 'apa', 'siapa', 'kapan', 'mana', 'bagaimana', 'gimana',
    'kenapa', 'mengapa', 'berapa', 'tolong', 'coba', 'bisa', 'boleh', 'dapat',
    'harus', 'wajib', 'tidak', 'bukan', 'tanpa', 'perlu', 'mungkin', 'pasti',
    'sah', 'batal', 'kalau', 'kalo', 'jika', 'bila', 'apabila', 'atau', 'dan',
    'serta', 'sama', 'dengan', 'karena', 'sebab', 'maka', 'sehingga', 'oleh',
    'pada', 'dalam', 'untuk', 'bagi', 'dari', 'ke', 'di', 'ini', 'itu', 'tersebut',
    'saya', 'aku', 'kami', 'kita', 'kamu', 'anda', 'dia', 'mereka', 'orang',
    'seseorang', 'diri', 'sendiri', 'istri', 'suami', 'anak', 'bayi', 'ibu', 'ayah',
    'baru', 'tadi', 'lagi', 'keluar', 'masuk', 'kena', 'terkena', 'ada',
    'hanya', 'saja', 'lebih', 'kurang', 'sangat', 'amat', 'juga', 'pun',
    'lain', 'ikut', 'buat', 'pakai', 'punya', 'tahu', 'mau', 'ingin'
}

FIQH_CANONICAL_TERMS = [
    'wudhu', 'shalat', 'puasa', 'zakat', 'haji', 'thaharah', 'tayamum', 'sujud', 'sahwi', 
    'rakaat', 'ruku', 'masbuk', 'makmum', 'imam', 'iftitah', 'fatihah', 'ramadhan', 'sahur', 
    'iftar', 'imsak', 'fidyah', 'kafarat', 'donor', 'darah', 'inhaler', 'tetes', 'mata', 
    'muntah', 'mani', 'madzi', 'wadi', 'junub', 'janabah', 'hadas', 'najis', 'istinja', 
    'kencing', 'istihadhah', 'nifas', 'haid', 'mutlak', 'mustamal', 'debu', 'khuf', 
    'jamak', 'qashar', 'tarawih', 'witir', 'tahajud', 'dhuha', 'jumat', 'khutbah', 
    'adzan', 'iqamah', 'fitrah', 'nishab', 'haul', 'muzakki', 'mustahik', 'sedekah', 
    'amil', 'qris', 'umrah', 'ihram', 'tawaf', 'wukuf', 'arafah', 'mina', 'dam', 
    'tahalul', 'miqat', 'jumrah', 'batal', 'membatalkan', 'rukun', 'syarat', 'sah', 
    'wajib', 'hukum', 'sunnah', 'makruh', 'haram', 'mubah', 'bacaan', 'niat', 'bekam'
]

INTENT_EXPANSION = [
    (r"\bwudhu\s+(tanah|debu|pasir)\b", "tayamum ketika tidak menemukan air debu tanah suci pengganti wudhu"),
    (r"\b(tidak ada|tanpa|ketiadaan|sulit|kehabisan)\s+air\b", "tayamum ketika tidak menemukan air debu pengganti wudhu bersuci"),
    (r"\b(tersentuh|sentuh)\s+(istri|suami)\b", "bersentuhan kulit suami istri ajnabiyyah"),
    (r"\b(botol|aqua|galon|kemasan)\b", "air kemasan botol air mutlak darurat"),
    (r"\blupa\s+rakaat\b", "lupa jumlah rakaat sujud sahwi shalat"),
    (r"\b(pesawat|penerbangan)\b", "shalat di pesawat safar kendaraan darurat"),
    (r"\b(qris|dana|transfer|dompet digital)\b", "membayar zakat fitrah menggunakan metode qris transfer dompet digital"),
    (r"\b(telat|tertinggal)\s+(sholat|imam)\b", "makmum masbuk tertinggal rakaat"),
    (r"\b(keramas|berendam)\s+puasa\b", "mandi ketika puasa"),
    (r"\bsafar\s+jauh\b", "shalat jamak qashar musafir bepergian jauh"),
    (r"\bkentut\b", "kentut dan wudhu buang angin was was"),
    (r"\b(saya|orang|pasien)\s+sakit\b", "shalat orang sakit rukhsah duduk berbaring keringanan kewajiban"),
    (r"\b(mimpi\s+basah|ihtilam).*puasa|puasa.*(mimpi\s+basah|ihtilam)\b", "mimpi basah ketika puasa batal puasa ihtilam keluar mani siang hari"),
    (r"\b(keluar\s+mani|sperma|ejakulasi|junub|janabah)\b", "hadas besar cara menyucikannya mandi wajib janabah keluar mani shalat bersuci thaharah"),
    (r"\b(tetes\s+mata|inhaler)\b", "menggunakan obat tetes mata ketika puasa membatalkan puasa"),
    (r"\b(krl|bus|kereta)\b", "tata cara shalat saat bepergian menggunakan krl atau bus antar kota"),
    (r"\b(puasa\s+ramadhan|puasa\s+ramadan|hukum\s+puasa\s+ramadhan)\b", "hukum puasa ramadan kewajiban fardhu ain rukun islam puasa ramadhan"),
    (r"\b(air\s+teh|air\s+kopi|air\s+sirup|air\s+kelapa|air\s+berwarna|air\s+sabun)\b", "air yang berubah warna karena benda lain air teh muqayyad macam macam air untuk bersuci"),
    (r"\b(cebok|istinja|tisu)\b", "buang air kecil dan wudhu buang air besar dan wudhu istinja bersuci"),
    (r"\b(donor\s+darah|darah|bekam|luka\s+berdarah|mimisan).*wudhu\b|\bwudhu.*(donor\s+darah|darah|bekam|luka\s+berdarah|mimisan)\b", "darah keluar setelah wudhu donor darah membatalkan wudhu thaharah bersuci"),
    (r"\b(baju|pakaian).*(najis|ompol|kencing).*sh[ao]lat\b|\bsh[ao]lat.*(baju|pakaian).*(najis|ompol|kencing)\b", "shalat ketika pakaian terkena najis pakaian terkena najis sah batal shalat"),
]

def normalize_slang(text: str) -> str:
    cleaned = text.lower()
    for pattern, replacement in SLANG_DICT.items():
        cleaned = re.sub(pattern, replacement, cleaned)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    tokens = cleaned.split()
    corrected = []
    import difflib
    for t in tokens:
        # Kata umum bahasa Indonesia jangan pernah diubah oleh korektor fikih
        if t in COMMON_ID_WORDS:
            corrected.append(t)
            continue
        if len(t) >= 4 and t not in FIQH_CANONICAL_TERMS:
            matches = difflib.get_close_matches(t, FIQH_CANONICAL_TERMS, n=1, cutoff=0.82)
            # Batasi perbedaan panjang maksimal 1 huruf agar tidak korupsi kata (misal 'saat' jadi 'syarat')
            if matches and abs(len(t) - len(matches[0])) <= 1:
                corrected.append(matches[0])
                continue
        corrected.append(t)
    return " ".join(corrected)

def expand_query(query: str) -> str:
    norm = normalize_slang(query)
    extra = []
    is_tayamum_intent = False
    # 1. Intent expansion
    for pattern, exp in INTENT_EXPANSION:
        if re.search(pattern, norm):
            extra.append(exp)
            if "tayamum" in exp:
                is_tayamum_intent = True
    # 2. Synonym expansion
    words = re.findall(r'\w+', norm)
    for w in words:
        # Jika query bermaksud tayamum, jangan banjiri dengan sinonim wudhu air
        if is_tayamum_intent and w in ["wudhu", "wudlu", "wudu"]:
            continue
        if w in SINONIM_FIQIH:
            extra.extend(SINONIM_FIQIH[w][:2])
    return (norm + " " + " ".join(extra)).strip() if extra else norm



def preprocess_text(text: str, stemmer, stopword) -> list:
    text = str(text).lower()
    text = re.sub(r'\[.*?\]:', '', text)
    if stopword:
        text = stopword.remove(text)
    if stemmer:
        text = stemmer.stem(text)
    return text.split()


def find_col(df, candidates):
    lmap = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lmap:
            return lmap[c.lower()]
    return None


@st.cache_resource(show_spinner="⚙️ Memuat basis data & model Quranica...")
def build_indexes(dataset_path: str):
    # 1. Fast Parquet / CSV Loading
    if os.path.exists(PARQUET_PATH):
        df = pd.read_parquet(PARQUET_PATH)
    else:
        df = pd.read_csv(dataset_path, encoding="utf-8-sig")
        try:
            df.to_parquet(PARQUET_PATH, index=False)
        except Exception:
            pass

    col_kat    = find_col(df, ["Kategori", "Category"])
    col_top    = find_col(df, ["Topik", "Judul", "Title"])
    col_nu     = find_col(df, ["Jawaban_NU", "NU", "Jawaban NU"])
    col_mu     = find_col(df, ["Jawaban_MU", "MU", "Jawaban_Muhammadiyah"])
    col_url_nu = find_col(df, ["URL_NU", "url_nu"])
    col_url_mu = find_col(df, ["URL_MU", "url_mu"])

    stemmer, stopword = load_sastrawi()

    # 2. Fast BM25 Model Loading (via bm25_model.pkl)
    bm25_model = None
    if os.path.exists(BM25_CACHE_PATH):
        try:
            with open(BM25_CACHE_PATH, "rb") as f:
                bm25_model = pickle.load(f)
        except Exception:
            bm25_model = None

    if bm25_model is None:
        bm25_corpus = []
        for _, row in df.iterrows():
            top = str(row[col_top]).strip() if col_top and pd.notna(row[col_top]) else ""
            j_nu = str(row[col_nu]).strip() if col_nu and pd.notna(row[col_nu]) else ""
            j_mu = str(row[col_mu]).strip() if col_mu and pd.notna(row[col_mu]) else ""
            bm25_corpus.append(f"{top} {top} {top} {j_nu[:350]} {j_mu[:350]}")
        tok_corpus = [preprocess_text(d, stemmer, stopword) for d in bm25_corpus]
        bm25_model = BM25Okapi(tok_corpus)
        try:
            with open(BM25_CACHE_PATH, "wb") as f:
                pickle.dump(bm25_model, f)
        except Exception:
            pass

    # 3. Fast S-BERT Embeddings Loading (via corpus_emb.npy)
    sbert_model = load_sbert()
    corpus_emb = None
    if os.path.exists(EMB_CACHE_PATH):
        try:
            corpus_emb = np.load(EMB_CACHE_PATH)
        except Exception:
            corpus_emb = None

    if corpus_emb is None:
        sbert_corpus = []
        for _, row in df.iterrows():
            top = str(row[col_top]).strip() if col_top and pd.notna(row[col_top]) else ""
            j_nu = str(row[col_nu]).strip() if col_nu and pd.notna(row[col_nu]) else ""
            j_mu = str(row[col_mu]).strip() if col_mu and pd.notna(row[col_mu]) else ""
            sbert_corpus.append(f"{top}. {j_nu[:500]} {j_mu[:500]}".strip())
        corpus_emb = sbert_model.encode(sbert_corpus, convert_to_numpy=True, show_progress_bar=False, batch_size=64)
        try:
            np.save(EMB_CACHE_PATH, corpus_emb)
        except Exception:
            pass

    cols = {
        "kat": col_kat, "top": col_top,
        "nu": col_nu, "mu": col_mu,
        "url_nu": col_url_nu, "url_mu": col_url_mu
    }
    return df, bm25_model, corpus_emb, cols, None


def search_bm25(query, bm25_model, stemmer, stopword, top_k=20):
    exp  = expand_query(query)
    tok  = preprocess_text(exp, stemmer, stopword)
    sc   = bm25_model.get_scores(tok)
    lim  = min(top_k, len(sc))
    rank = np.argsort(sc)[::-1][:lim]
    return [{"idx_pandas": int(i), "skor_bm25": float(sc[i]), "rank_bm25": r+1}
            for r, i in enumerate(rank) if sc[i] > 0]


def search_sbert(query, sbert_model, corpus_emb, top_k=20):
    import torch
    q_emb = sbert_model.encode(query, convert_to_tensor=True, show_progress_bar=False)
    if isinstance(corpus_emb, np.ndarray):
        c_emb = torch.from_numpy(corpus_emb).to(q_emb.device)
    else:
        c_emb = corpus_emb
    sc   = util.cos_sim(q_emb, c_emb)[0].cpu().numpy()
    lim  = min(top_k, len(sc))
    rank = np.argsort(sc)[::-1][:lim]
    return [{"idx_pandas": int(i), "skor_sbert": float(sc[i]), "rank_sbert": r+1}
            for r, i in enumerate(rank)]


def reciprocal_rank_fusion(bm25_res, sbert_res, k=60, w_bm25=0.45, w_sbert=0.55):
    scores = {}
    for item in bm25_res:
        idx = item["idx_pandas"]
        scores[idx] = scores.get(idx, 0.0) + w_bm25 / (k + item["rank_bm25"])
    for item in sbert_res:
        idx = item["idx_pandas"]
        scores[idx] = scores.get(idx, 0.0) + w_sbert / (k + item["rank_sbert"])
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


CATEGORY_KEYWORDS = {
    "THAHARAH": [
        "wudhu", "wudlu", "wudu", "bersuci", "tayamum", "tayammum", "hadats", "hadas",
        "najis", "istinja", "junub", "janabah", "mani", "sperma", "mandi", "haid", "nifas",
        "sentuh", "bersentuhan", "kulit", "batal", "cebok", "khuf", "istihadlah", "istihadhah",
        "madzi", "wadi", "darah", "kencing", "kotoran", "air", "debu"
    ],
    "SHALAT": [
        "shalat", "sholat", "salat", "sembahyang", "rakaat", "sujud", "ruku", "masbuk",
        "makmum", "imam", "fardhu", "tarawih", "witir", "tahajud", "dhuha", "jamak",
        "qashar", "khutbah", "jumat", "adzan", "iqamah", "qobliyah", "badiyah", "takbiratul ihram"
    ],
    "PUASA": [
        "puasa", "shaum", "shiyam", "ramadhan", "ramadan", "sahur", "iftar", "fidyah",
        "kafarat", "inhaler", "donor", "suntik", "imsak", "berbuka", "berkumur", "menelan", "tertelan"
    ],
    "ZAKAT": [
        "zakat", "zakah", "fitrah", "mal", "nishab", "haul", "muzakki", "mustahiq",
        "infaq", "sedekah", "amil", "qris", "beras"
    ],
    "HAJI": [
        "haji", "umrah", "umroh", "ihram", "tawaf", "thawaf", "sai", "wukuf", "arafah",
        "mina", "tamattu", "ifrad", "qiran", "dam", "tahalul", "miqat", "jumrah"
    ]
}

def clean_topic_subject(t: str) -> str:
    s = str(t).lower().strip()
    s = re.sub(r'[\?\.!]+$', '', s).strip()
    for prefix in [
        'pengertian dan hukum ', 'pengertian ', 'definisi ', 'makna ',
        'hukum dan tata cara ', 'hukum ', 'tata cara ', 'syarat wajib dan sah ',
        'syarat sah ', 'syarat wajib ', 'rukun ', 'dalil kewajiban ', 'keutamaan ',
        'kedudukan ', 'kewajiban ', 'larangan '
    ]:
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
            break
    return s

def detect_query_category(query: str, return_strict: bool = False):
    q_low = query.lower()
    # Explicit override: Takbiratul ihram adalah rukun SHALAT murni (bukan Haji)
    if "takbiratul ihram" in q_low:
        return ("SHALAT", True) if return_strict else "SHALAT"

    # Puasa overrides (misal: mimpi basah saat puasa, tetes mata saat puasa, menelan air saat wudhu)
    if "puasa" in q_low and any(k in q_low for k in ["mimpi basah", "ihtilam", "tetes mata", "inhaler", "menelan", "berkumur"]):
        return ("PUASA", True) if return_strict else "PUASA"

    # Thaharah specific fluids & purifications
    if any(k in q_low for k in ["keluar mani", "mani", "sperma", "janabah", "junub", "mandi wajib", "hadas besar", "air teh", "air kopi", "air sirup", "air sabun", "air musta"]):
        return ("THAHARAH", True) if return_strict else "THAHARAH"

    # Darah / Bekam dalam konteks Thaharah & Wudhu
    if any(k in q_low for k in ["darah", "bekam", "mimisan"]) and any(w in q_low for w in ["wudhu", "thaharah", "bersuci"]):
        return ("THAHARAH", True) if return_strict else "THAHARAH"

    words = re.findall(r'\w+', q_low)
    scores = {}
    for cat, kws in CATEGORY_KEYWORDS.items():
        score = sum(1 for w in words if w in kws)
        if score > 0:
            scores[cat] = score

    if not scores:
        return (None, False) if return_strict else None

    sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_cat, top_score = sorted_cats[0]

    # Resolusi multi-kategori / pertanyaan silang bab ibadah
    is_strict = True
    if len(sorted_cats) > 1:
        cats_in_query = {c for c, _ in sorted_cats}
        if "SHALAT" in cats_in_query and "THAHARAH" in cats_in_query:
            top_cat = "SHALAT"
        elif "PUASA" in cats_in_query and "THAHARAH" in cats_in_query:
            # HANYA pilih PUASA jika kueri menyebutkan puasa/ramadhan/sahur/imsak/iftar
            if any(w in words for w in ["puasa", "shaum", "shiyam", "ramadhan", "ramadan", "sahur", "imsak", "iftar", "buka"]):
                top_cat = "PUASA"
            else:
                top_cat = "THAHARAH"
        elif "HAJI" in cats_in_query and "SHALAT" in cats_in_query:
            top_cat = "HAJI"
        is_strict = False
    else:
        # Jika hanya ada kata umum seperti 'air' tanpa kata kunci thaharah lain, jangan strict
        if top_cat == "THAHARAH" and "air" in words and len(words) <= 3:
            is_strict = False

    return (top_cat, is_strict) if return_strict else top_cat


def hybrid_search(query, bm25_model, sbert_model, corpus_emb,
                  stemmer, stopword, df=None, cols=None,
                  top_k=5, w_bm25=0.50, w_sbert=0.50, return_details=False):
    import torch
    exp  = expand_query(query)
    q_low = exp.lower()
    tok  = preprocess_text(exp, stemmer, stopword)
    bm_scores = np.array(bm25_model.get_scores(tok))

    # Dense semantic embedding scores (menggunakan query ternormalisasi agar slang & typo teratasi)
    q_emb = sbert_model.encode(exp, convert_to_tensor=True, show_progress_bar=False)
    if isinstance(corpus_emb, np.ndarray):
        c_emb = torch.from_numpy(corpus_emb).to(q_emb.device)
    else:
        c_emb = corpus_emb
    dense_scores = util.cos_sim(q_emb, c_emb)[0].cpu().numpy()

    # Min-Max Normalization of lexical BM25
    bm_max = bm_scores.max() if bm_scores.max() > 0 else 1.0
    norm_bm = np.clip(bm_scores / bm_max, 0, 1)

    # Per-Query Min-Max Normalization of Dense S-BERT (menghilangkan baseline 0.70 noise)
    d_min, d_max = dense_scores.min(), dense_scores.max()
    norm_dense = (dense_scores - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(dense_scores)

    # Specificity-Aware Subject & Topic Bonus (Presisi Tinggi Fiqih Ubudiyah)
    n_docs = len(bm_scores)
    topic_bonus = np.zeros(n_docs)
    q_stems = set(tok)
    if "ramadhan" in q_stems or "ramadan" in q_stems:
        q_stems.add("ramadhan")
        q_stems.add("ramadan")

    if q_stems and df is not None and cols and cols.get("top"):
        col_top = cols["top"]
        for i, r in df.iterrows():
            t_val = str(r[col_top]) if pd.notna(r[col_top]) else ""
            clean_subj = clean_topic_subject(t_val)
            top_stems = set(preprocess_text(t_val, stemmer, stopword))
            clean_stems = set(preprocess_text(clean_subj, stemmer, stopword))
            if "ramadhan" in top_stems or "ramadan" in top_stems:
                top_stems.add("ramadhan")
                top_stems.add("ramadan")
            
            # 1. Full clean subject phrase match in normalized query
            if len(clean_subj) >= 4 and clean_subj in q_low:
                topic_bonus[i] += 0.25
            
            # 2. Clean subject stem coverage (dokumen spesifik mendapatkan bobot lebih proporsional)
            if clean_stems:
                match_clean = len(clean_stems.intersection(q_stems))
                if match_clean == len(clean_stems) and len(clean_stems) >= 2:
                    topic_bonus[i] += 0.20 + (len(clean_stems) * 0.05)
                elif match_clean > 0:
                    topic_bonus[i] += (match_clean / len(clean_stems)) * 0.15
            
            # 3. Overlap kata topik penuh
            overlap = len(q_stems.intersection(top_stems))
            if overlap > 0:
                topic_bonus[i] += min(overlap, 4) * 0.03

        # Skala bonus seimbang (0.50x) agar tidak menutupi keunggulan leksikal BM25 & semantik S-BERT
        topic_bonus *= 0.50

    # Soft Category Gating (Non-Destructive Smart Filter)
    cat_adj = np.zeros(n_docs)
    q_cat, is_strict = detect_query_category(exp, return_strict=True)
    if q_cat and df is not None and cols and cols.get("kat"):
        col_k = cols["kat"]
        for i, r in df.iterrows():
            doc_cat = str(r[col_k]).strip().upper() if pd.notna(r[col_k]) else ""
            if q_cat in doc_cat:
                cat_adj[i] = 0.20
            else:
                cat_adj[i] = -0.25 if is_strict else -0.05

    final_scores = (w_bm25 * norm_bm) + (w_sbert * norm_dense) + topic_bonus + cat_adj
    best_indices = np.argsort(final_scores)[::-1][:top_k]
    hy_res = [(int(i), float(final_scores[i])) for i in best_indices]

    if return_details:
        bm_best = np.argsort(bm_scores)[::-1][:top_k]
        bm_res = [(int(i), float(bm_scores[i])) for i in bm_best]
        sb_best = np.argsort(dense_scores)[::-1][:top_k]
        sb_res = [(int(i), float(dense_scores[i])) for i in sb_best]
        return hy_res, bm_res, sb_res
    return hy_res


# =============================================================
# ALGORITMA KLASIFIKASI STATUS HUKUM & KONSENSUS
# =============================================================
def deteksi_hukum(text):
    """
    Context-Aware Rule-Based Classifier dengan Negation Scope Resolution.
    Mendeteksi status hukum Islam dengan membedakan negasi (misal 'tidak membatalkan' -> Boleh)
    dan memprioritaskan kalimat penentu hukum utama.
    Returns: (label_badge, warna_hex, kategori_singkat)
    """
    if not text or not isinstance(text, str):
        return "ℹ️ Mubah / Boleh", "#22c55e", "Boleh / Sah"

    t = text.lower()

    # 1. Negasi pembatal (contoh: "tidak membatalkan wudhu/puasa", "bukan pembatal") -> Boleh/Sah
    if re.search(r"\b(tidak|bukan)\s+(membatalkan|merusak|menggugurkan)\b", t):
        if re.search(r"\b(wajib|fardhu)\b", t[:250]):
            return "☑️ Wajib", "#10b981", "Wajib"
        return "✅ Boleh / Sah", "#22c55e", "Boleh / Sah"

    # 2. Negasi keabsahan / Batal (contoh: "tidak sah", "tidak boleh", "batal")
    if re.search(r"\b(tidak|bukan)\s+(sah|boleh|diperbolehkan|dibenarkan)\b", t) or re.search(r"\b(batal|membatalkan)\b", t[:350]):
        return "❌ Tidak Sah / Batal", "#ef4444", "Tidak Sah / Batal"

    # 3. Larangan mutlak / Haram (pastikan tidak ada negasi "tidak haram")
    if re.search(r"\b(haram|diharamkan|terlarang)\b", t):
        if not re.search(r"\b(tidak|bukan)\s+(haram|diharamkan)\b", t):
            return "🚫 Haram", "#dc2626", "Haram"

    # 4. Keringanan syariat (Rukhshah) -> Boleh
    if re.search(r"\b(rukhshah|kemudahan|keringanan)\b", t[:350]):
        return "✅ Boleh (Rukhshah)", "#14b8a6", "Boleh (Rukhshah)"

    # 5. Kewajiban (Wajib / Fardhu / Rukun)
    if re.search(r"\b(wajib|fardhu|harus|kewajiban|rukun)\b", t[:350]):
        if not re.search(r"\b(tidak|bukan)\s+(wajib|fardhu|harus)\b", t[:350]):
            return "☑️ Wajib", "#10b981", "Wajib"

    # 6. Anjuran (Sunnah / Mustahab)
    if re.search(r"\b(sunnah|mustahab|dianjurkan|afdhal|keutamaan)\b", t[:350]):
        if not re.search(r"\b(tidak|bukan)\s+(sunnah|dianjurkan)\b", t[:350]):
            return "✨ Sunnah", "#3b82f6", "Sunnah"

    # 7. Makruh
    if re.search(r"\b(makruh|dimakruhkan|kurang disukai)\b", t[:350]):
        if not re.search(r"\b(tidak|bukan)\s+(makruh|dimakruhkan)\b", t[:350]):
            return "⚠️ Makruh", "#f59e0b", "Makruh"

    # 8. Mubah / Sah / Boleh
    if re.search(r"\b(boleh|diperbolehkan|sah|mubah|halal)\b", t[:350]):
        return "✅ Boleh / Sah", "#22c55e", "Boleh / Sah"

    # Fallback scanning
    if re.search(r"\b(wajib|fardhu)\b", t): return "☑️ Wajib", "#10b981", "Wajib"
    if re.search(r"\bsunnah\b", t): return "✨ Sunnah", "#3b82f6", "Sunnah"
    if re.search(r"\b(boleh|sah|mubah)\b", t): return "✅ Boleh", "#22c55e", "Boleh / Sah"
    if re.search(r"\bmakruh\b", t): return "⚠️ Makruh", "#f59e0b", "Makruh"
    if re.search(r"\b(haram)\b", t): return "🚫 Haram", "#dc2626", "Haram"

    return "ℹ️ Mubah / Boleh", "#22c55e", "Boleh / Sah"


def ringkasan_konsensus_hukum(label_nu, cat_nu, label_mu, cat_mu):
    """
    Menghasilkan HTML banner komparasi hukum antara NU dan Muhammadiyah.
    """
    # Normalkan kategori singkat untuk komparasi kesepakatan
    norm_nu = cat_nu.lower().replace(" (rukhshah)", "").replace(" / sah", "").replace(" / batal", "")
    norm_mu = cat_mu.lower().replace(" (rukhshah)", "").replace(" / sah", "").replace(" / batal", "")

    if norm_nu == norm_mu:
        return f"""
        <div class="hukum-consensus-box consensus-agree">
          <span style="font-size:1.3rem;">🤝</span>
          <div>
            <strong>Kesimpulan Komparasi (Sepakat / Ittifaq):</strong><br>
            Kedua ormas (Nahdlatul Ulama & Muhammadiyah) bersepakat menetapkan hukum:
            <span style="font-weight:800;text-decoration:underline;">{cat_nu}</span>.
          </div>
        </div>
        """
    else:
        return f"""
        <div class="hukum-consensus-box consensus-differ">
          <span style="font-size:1.3rem;">⚖️</span>
          <div>
            <strong>Kesimpulan Komparasi (Perbedaan / Khilafiyah):</strong><br>
            Terdapat perbedaan rincian hukum: Nahdlatul Ulama memandang <strong>{cat_nu}</strong>,
            sedangkan Muhammadiyah memandang <strong>{cat_mu}</strong>.
          </div>
        </div>
        """


def ekstrak_intisari_fatwa(text, max_len=240):
    """
    Ekstraksi kalimat kunci representatif (salient sentence extraction)
    yang memuat ketetapan hukum resmi dari teks fatwa tanpa LLM.
    """
    if not text or not isinstance(text, str):
        return ""
    # Bersihkan markdown formatting berlebih
    clean_text = re.sub(r'\[.*?\]:', '', text)
    clean_text = re.sub(r'[*#_]', '', clean_text)
    
    # Pecah kalimat berdasarkan tanda titik atau baris baru
    raw_sentences = [s.strip() for s in re.split(r'[\.\n]+', clean_text) if len(s.strip()) > 25]
    
    # Cari kalimat yang memuat kata kunci keputusan hukum / fatwa
    decision_pattern = re.compile(
        r'\b(hukum|boleh|sah|wajib|sunnah|makruh|haram|tidak sah|batal|kemudahan|rukhshah|fatwa|bahtsul|tarjih|menetapkan|memfatwakan|dianjurkan|disyariatkan)\b',
        re.IGNORECASE
    )
    for s in raw_sentences:
        if decision_pattern.search(s):
            compact_s = re.sub(r'\s+', ' ', s)
            return compact_s[:max_len] + ('…' if len(compact_s) > max_len else '.')

    # Fallback ke kalimat pertama yang padat
    return (raw_sentences[0][:max_len] + '…') if raw_sentences else ""


def buat_sintesis_eksekutif_top4(top_docs, q_last):
    """
    Menghasilkan Rangkuman & Sintesis Eksekutif Fikih terpadu (single unified synthesis)
    yang menggabungkan 4 fatwa teratas menjadi satu kesatuan komparatif:
    1. Header & Rasio Konsensus (Sepakat vs Khilafiyah).
    2. Tinjauan Analisis Terpadu.
    3. Dual-Panel Komparasi Garis Besar Mazhab (🟢 Nahdlatul Ulama vs 🔵 Muhammadiyah).
    4. Poin-Poin Ketetapan Hukum 4 Fatwa Terkait (4 concise items).
    5. Harmonisasi Fikih & Panduan Praktis Ibadah.
    """
    if not top_docs:
        return ""
    
    total_docs = len(top_docs)
    count_agree = sum(1 for d in top_docs if d.get("is_agree", False))
    count_khilaf = total_docs - count_agree
    
    # Overview & Guidance logic
    if count_khilaf == 0:
        stat_badge = f'<span class="badge-agree-pill">🤝 {count_agree} Fatwa Sepakat (Ittifaq)</span>'
        harmoni_title = "Ketetapan Terpadu (Ittifaq Mutlak)"
        harmoni_text = (
            f"Seluruh fatwa teratas ({total_docs} fatwa terkait) menunjukkan kesepakatan mutlak antara Nahdlatul Ulama dan Majelis Tarjih Muhammadiyah. "
            "Umat Islam dapat mengamalkannya dengan penuh ketenangan tanpa keraguan."
        )
    elif count_agree == 0:
        stat_badge = f'<span class="badge-differ-pill">⚖️ {count_khilaf} Fatwa Khilafiyah</span>'
        harmoni_title = "Kaidah Menghadapi Khilafiyah"
        harmoni_text = (
            "Fatwa-fatwa terkait memiliki variasi penafsiran (<em>khilafiyah ijtihadiyah</em>). "
            "Mengacu pada kaidah ushul fikih <strong>'Al-khuruju minal khilaf mustahab'</strong> (keluar dari perselisihan hukum adalah dianjurkan), "
            "umat disarankan mengambil langkah kehati-hatian (<em>ihtiyath</em>) atau konsisten mengikuti ketetapan ormas yang diyakini."
        )
    else:
        stat_badge = (
            f'<span class="badge-agree-pill">🤝 {count_agree} Sepakat</span> '
            f'<span class="badge-differ-pill">⚖️ {count_khilaf} Khilafiyah</span>'
        )
        harmoni_title = "Harmonisasi & Solusi Praktis Ibadah"
        harmoni_text = (
            f"Dari <strong>{total_docs} fatwa teratas</strong> yang teridentifikasi, terdapat {count_agree} fatwa berstatus sepakat (<em>ittifaq</em>) dan {count_khilaf} fatwa dengan rincian <em>khilafiyah</em>. "
            "Perbedaan dalam cabang fikih (<em>furu'iyyah</em>) merupakan rahmat dan kelapangan syariat (<em>rukhshah</em>). Kaum muslimin dipersilakan mengamalkan sesuai keyakinan atau arahan ulama setempat."
        )

    # Sintesis Naratif NU & MU
    d0 = top_docs[0]
    cat_nu_main = d0.get("cat_nu", "Boleh")
    cat_mu_main = d0.get("cat_mu", "Boleh")
    inti_nu_main = d0.get("inti_nu", "")
    inti_mu_main = d0.get("inti_mu", "")
    
    # Rangkuman tambahan dari fatwa pendukung jika ada
    extra_nu = f" Pada perkara terkait {top_docs[1]['topik'].lower()}: {top_docs[1]['inti_nu']}" if total_docs > 1 and top_docs[1].get('inti_nu') else ""
    extra_mu = f" Pada perkara terkait {top_docs[1]['topik'].lower()}: {top_docs[1]['inti_mu']}" if total_docs > 1 and top_docs[1].get('inti_mu') else ""

    narasi_nu = f"Berdasarkan ketetapan Bahtsul Masail (mazhab Syafi'i), NU menetapkan hukum utama <strong>{cat_nu_main}</strong>. {inti_nu_main}{extra_nu}"
    narasi_mu = f"Berdasarkan Manhaj Tarjih (Al-Qur'an &amp; Sunnah maqashid), Muhammadiyah menetapkan hukum utama <strong>{cat_mu_main}</strong>. {inti_mu_main}{extra_mu}"

    # Poin-poin 4 fatwa
    poin_items_html = ""
    for d in top_docs:
        rank = d.get("rank", 1)
        topik = html_lib.escape(str(d.get("topik", "")))
        cat_nu = html_lib.escape(str(d.get("cat_nu", "")))
        cat_mu = html_lib.escape(str(d.get("cat_mu", "")))
        inti_nu = html_lib.escape(str(d.get("inti_nu", "")))
        inti_mu = html_lib.escape(str(d.get("inti_mu", "")))

        if d.get("is_agree", False):
            chip = f'<span class="sintesis-badge-agree">🤝 Sepakat: {cat_nu}</span>'
        else:
            chip = f'<span class="sintesis-badge-differ">⚖️ Khilafiyah: NU ({cat_nu}) vs MU ({cat_mu})</span>'

        poin_items_html += f"""
        <div class="sintesis-poin-item">
          <span class="sintesis-poin-badge">#{rank}</span>
          <div style="flex:1;">
            <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.4rem;margin-bottom:0.25rem;">
              <span style="font-weight:800;color:#1A202C;font-size:0.9rem;">{topik}</span>
              {chip}
            </div>
            <div style="font-size:0.83rem;color:#475569;line-height:1.5;">
              <span style="color:#2D6A4F;font-weight:700;">🟢 NU:</span> {inti_nu} &nbsp;&bull;&nbsp; 
              <span style="color:#1A73E8;font-weight:700;">🔵 MU:</span> {inti_mu}
            </div>
          </div>
        </div>
        """

    return f"""
    <div class="sintesis-box">
      <div class="sintesis-header">
        <div class="sintesis-title">
          <span>📋</span> Rangkuman &amp; Sintesis Terpadu 4 Fatwa Fikih
        </div>
        <div class="sintesis-stats-badge">
          {stat_badge}
        </div>
      </div>
      
      <div class="sintesis-overview-lead">
        Sintesis komparatif AI merangkum ketetapan hukum dari <strong>{total_docs} fatwa fikih paling relevan</strong> untuk persoalan: 
        &ldquo;<em>{html_lib.escape(q_last)}</em>&rdquo;. Membandingkan garis besar ijtihad antara <strong>Nahdlatul Ulama</strong> dan <strong>Muhammadiyah</strong>:
      </div>

      <div class="sintesis-dual-box">
        <div class="sintesis-dual-panel nu">
          <div class="sintesis-dual-head nu">
            <span>🟢</span> Garis Besar Sikap Nahdlatul Ulama (NU)
          </div>
          <div class="sintesis-dual-body">
            {narasi_nu}
          </div>
        </div>
        <div class="sintesis-dual-panel mu">
          <div class="sintesis-dual-head mu">
            <span>🔵</span> Garis Besar Sikap Majelis Tarjih (MU)
          </div>
          <div class="sintesis-dual-body">
            {narasi_mu}
          </div>
        </div>
      </div>

      <div class="sintesis-poin-block">
        <div class="sintesis-poin-title">
          <span>📌</span> Rincian Ketetapan 4 Fatwa Terkait:
        </div>
        <div class="sintesis-poin-list">
          {poin_items_html}
        </div>
      </div>

      <div class="sintesis-footer">
        💡 <strong>{harmoni_title}:</strong> {harmoni_text}
      </div>
    </div>
    """


def highlight(text, query):
    esc  = html_lib.escape(text)
    kata = sorted(set(k for k in re.findall(r"\w+", query.lower()) if len(k) > 2), key=len, reverse=True)
    for k in kata:
        esc = re.compile(rf"(?<!\w)({re.escape(k)}\w*)", re.IGNORECASE).sub(r'<mark class="hl">\1</mark>', esc)
    return esc


def log_pencarian(query: str, topik_hasil: list):
    is_new = not os.path.exists(LOG_SEARCH)
    try:
        with open(LOG_SEARCH, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(["timestamp", "query", "topik"])
            for t in topik_hasil:
                w.writerow([datetime.now().isoformat(), query, t])
    except Exception:
        pass


def log_feedback(query: str, topik: str, nilai: str):
    is_new = not os.path.exists(LOG_FEEDBACK)
    try:
        with open(LOG_FEEDBACK, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(["timestamp", "query", "topik", "nilai"])
            w.writerow([datetime.now().isoformat(), query, topik, nilai])
    except Exception:
        pass


# =============================================================
# DATA DEFINITIONS (TRIVIA & RECS)
# =============================================================
TRIVIA_FIKIH = [
    {
        "doc_idx": 84,
        "title": "Tidur dan wudhu",
        "q": "Apakah tidur membatalkan wudhu jika duduk tegak?",
        "hint": "Menurut mayoritas ulama Syafi'iyah, tidur dengan pantat rapat di lantai tidak membatalkan wudhu.",
        "query": "tidur membatalkan wudhu posisi duduk tegak"
    },
    {
        "doc_idx": 347,
        "title": "Menggunakan obat tetes mata ketika puasa",
        "q": "Apakah obat tetes mata atau inhaler membatalkan puasa?",
        "hint": "Fatwa kontemporer Bahtsul Masail & Tarjih menyatakan tetes mata tidak membatalkan puasa karena rongga mata bukan jalur makanan.",
        "query": "obat tetes mata inhaler membatalkan puasa"
    },
    {
        "doc_idx": 285,
        "title": "Tata cara shalat saat bepergian menggunakan KRL atau bus antar kota",
        "q": "Bagaimana cara shalat di kendaraan umum (KRL / bus antar kota)?",
        "hint": "Boleh shalat di kursi kendaraan menghadap ke arah laju kendaraan jika tidak memungkinkan berdiri atau menghadap kiblat.",
        "query": "tata cara shalat di kendaraan bergerak safar"
    },
    {
        "doc_idx": 417,
        "title": "Membayar zakat fitrah menggunakan metode QRIS atau transfer dompet digital",
        "q": "Bolehkah membayar zakat fitrah via transfer bank / QRIS?",
        "hint": "NU dan Muhammadiyah membolehkan pembayaran via QRIS/transfer kepada lembaga amil terpercaya sebagai wakil zakat.",
        "query": "zakat fitrah uang tunai beras madzhab"
    }
]

FATWA_RECS = [
    {
        "doc_idx": 151,
        "title": "Lupa Membaca Al-Fatihah",
        "snippet": "Penjelasan hukum shalat bagi makmum atau imam yang lupa membaca Al-Fatihah serta ketentuan sujud sahwi...",
        "category": "sholat", "icon": "🤲",
        "org": "NU & MU", "query": "lupa membaca al fatihah"
    },
    {
        "doc_idx": 28,
        "title": "Hukum Tayamum Ketika Tidak Menemukan Air",
        "snippet": "Tata cara dan rukun bersuci dengan debu suci sebagai pengganti wudhu saat ketiadaan air atau kondisi sakit...",
        "category": "thaharah", "icon": "💧",
        "org": "NU & MU", "query": "hukum tayamum ketika tidak menemukan air"
    },
    {
        "doc_idx": 367,
        "title": "Puasa Bagi Perempuan Hamil",
        "snippet": "Ketentuan rukhshah berbuka puasa bagi wanita hamil atau menyusui beserta kewajiban qadha dan pembayaran fidyah...",
        "category": "puasa", "icon": "🌙",
        "org": "NU & MU", "query": "puasa bagi perempuan hamil"
    },
    {
        "doc_idx": 417,
        "title": "Zakat Fitrah via QRIS & Transfer Digital",
        "snippet": "Keabsahan fiqih kontemporer pembayaran zakat fitrah dan mal melalui sistem perbankan modern dan dompet digital...",
        "category": "zakat", "icon": "💳",
        "org": "NU & MU", "query": "membayar zakat fitrah menggunakan qris transfer dompet digital"
    },
    {
        "doc_idx": 255,
        "title": "Ketentuan Shalat Jamak Bagi Musafir",
        "snippet": "Ketetapan menggabungkan dua shalat fardhu dalam satu waktu baik taqdim maupun ta'khir bagi musafir...",
        "category": "sholat", "icon": "🚗",
        "org": "NU & MU", "query": "shalat jamak"
    },
    {
        "doc_idx": 3,
        "title": "Perbedaan Mendasar Antara Hadas & Najis",
        "snippet": "Pengertian hadats kecil/besar serta klasifikasi najis mukhaffafah, mutawassithah, dan mughallazhah beserta cara menyucikannya...",
        "category": "thaharah", "icon": "🚿",
        "org": "NU & MU", "query": "perbedaan mendasar antara hadas dan najis"
    },
    {
        "doc_idx": 260,
        "title": "Hukum Jamak dan Qashar Shalat",
        "snippet": "Keringanan (rukhshah) meringkas dan menggabungkan shalat dalam perjalanan beserta syarat jarak tempuh safar...",
        "category": "sholat", "icon": "📖",
        "org": "NU & MU", "query": "jamak dan qashar"
    },
    {
        "doc_idx": 84,
        "title": "Hukum Tidur dan Pembatal Wudhu",
        "snippet": "Kondisi tidur yang membatalkan wudhu dan posisi duduk tetap di lantai yang tidak membatalkan kesucian...",
        "category": "thaharah", "icon": "📚",
        "org": "NU & MU", "query": "tidur dan wudhu"
    },
    {
        "doc_idx": 416,
        "title": "Zakat Fitrah dengan Makanan Pokok",
        "snippet": "Takaran wajib zakat fitrah berupa beras atau gandum per jiwa serta perbandingan dengan konversi nilai uang tunai...",
        "category": "zakat", "icon": "🌾",
        "org": "NU & MU", "query": "zakat fitrah dengan makanan pokok"
    },
    {
        "doc_idx": 347,
        "title": "Menggunakan Obat Tetes Mata Saat Puasa",
        "snippet": "Pandangan fatwa NU dan Majelis Tarjih Muhammadiyah mengenai penyerapan obat tetes mata di siang hari Ramadhan...",
        "category": "puasa", "icon": "💊",
        "org": "NU & MU", "query": "menggunakan obat tetes mata ketika puasa"
    },
    {
        "doc_idx": 495,
        "title": "Ketentuan dan Syarat Tawaf Ifadah",
        "snippet": "Rukun haji yang wajib dilaksanakan setelah wukuf di Padang Arafah untuk menyempurnakan keabsahan ibadah haji...",
        "category": "haji", "icon": "🕋",
        "org": "NU & MU", "query": "tawaf ifadah"
    },
    {
        "doc_idx": 345,
        "title": "Keramas dan Mandi Saat Berpuasa",
        "snippet": "Hukum membasahi kepala, berkeramas, atau mandi berendam di siang hari Ramadhan untuk menyegarkan badan...",
        "category": "puasa", "icon": "🛁",
        "org": "NU & MU", "query": "keramas ketika puasa"
    },
]

QUICK_TOPICS = [
    ("🤲 Sholat",    "syarat rukun sah shalat"),
    ("🚿 Wudhu",     "tata cara dan syarat wudhu"),
    ("🌙 Puasa",     "hukum puasa ramadhan"),
    ("💰 Zakat",     "nisab zakat penghasilan"),
    ("💧 Thaharah",  "bersuci dari najis"),
    ("🕋 Haji",      "rukun dan syarat sah haji"),
]


# =============================================================
# INITIALIZE STATE
# =============================================================
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
