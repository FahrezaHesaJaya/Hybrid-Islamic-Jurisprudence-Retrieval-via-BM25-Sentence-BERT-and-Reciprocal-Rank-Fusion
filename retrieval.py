# =============================================================
# retrieval.py — Information Retrieval Core Engine
# BM25 + Indo Sentence-BERT + Intent Expansion + Reciprocal Rank Fusion
# =============================================================

import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH    = os.path.join(BASE_DIR, "Data_ubudiyah_final.csv")
PARQUET_PATH    = os.path.join(BASE_DIR, "Data_ubudiyah_final.parquet")
BM25_CACHE_PATH = os.path.join(BASE_DIR, "bm25_model.pkl")
EMB_CACHE_PATH  = os.path.join(BASE_DIR, "corpus_emb.npy")

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
    "mimisan":     ["darah", "darah keluar", "najis darah", "batal wudhu"],
    "darah":       ["darah keluar", "luka berdarah", "mimisan", "najis"],
    "luka":        ["darah mengalir", "najis darah", "perban"],
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
    (r"\b(mimisan|darah|luka\s+berdarah).*sh[ao]lat\b|\bsh[ao]lat.*(mimisan|darah|luka\s+berdarah)\b", "darah keluar ketika shalat batal wudhu ketika shalat pakaian terkena darah najis darah"),
    (r"\b(mimisan|hidung\s+berdarah)\b", "darah keluar setelah wudhu membatalkan wudhu pakaian terkena darah najis darah"),
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
        "madzi", "wadi", "darah", "mimisan", "luka", "bekam", "kencing", "kotoran", "air", "debu"
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

