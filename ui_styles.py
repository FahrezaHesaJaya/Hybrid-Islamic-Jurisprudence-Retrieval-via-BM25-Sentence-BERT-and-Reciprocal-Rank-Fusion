# =============================================================
# ui_styles.py — Quranica UI Design System & Islamic Theme
# Custom CSS, Typography, and HTML Components
# =============================================================

import os
import base64
import streamlit as st

def render_html(html_str: str):
    """
    Renders HTML safely in Streamlit by stripping all leading whitespace from every line.
    This strictly prevents Markdown from treating lines with 4+ spaces as <pre><code> blocks.
    """
    clean_lines = [line.lstrip() for line in html_str.splitlines()]
    clean_html = "\n".join(clean_lines).strip()
    st.markdown(clean_html, unsafe_allow_html=True)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_HERO_PATH = os.path.join(BASE_DIR, "assets", "quranica_hero.jpg")

@st.cache_data(show_spinner=False)
def get_hero_image_b64(hero_img_path: str = ""):
    """Loads and caches hero banner image as base64 string."""
    target_path = hero_img_path if hero_img_path else DEFAULT_HERO_PATH
    if target_path and os.path.exists(target_path):
        try:
            with open(target_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            pass
    return ""


def apply_custom_styles():
    """Applies the complete Quranica Warm Islamic Google Theme CSS."""
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
</style>
""")
