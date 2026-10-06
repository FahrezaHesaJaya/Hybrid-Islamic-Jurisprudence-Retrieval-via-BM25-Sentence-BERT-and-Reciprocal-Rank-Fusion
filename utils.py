# =============================================================
# utils.py — Utilities, Prayer Times, Trivia & Data Definitions
# =============================================================

import os
import csv
import re
import requests
import html as html_lib
from datetime import datetime
import streamlit as st

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
LOG_SEARCH   = os.path.join(BASE_DIR, "log_pencarian.csv")
LOG_FEEDBACK = os.path.join(BASE_DIR, "log_feedback.csv")

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




HARI_ID = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
BULAN_ID = ["", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
            "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
