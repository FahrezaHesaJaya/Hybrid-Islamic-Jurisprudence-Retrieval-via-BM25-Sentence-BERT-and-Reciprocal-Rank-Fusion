# =============================================================
# classifier.py — Fiqh Legal Status Classifier & Synthesis
# Rule-Based Context-Aware Hukum Classifier & Cross-Madzhab Consensus
# =============================================================

import re
import html as html_lib

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

