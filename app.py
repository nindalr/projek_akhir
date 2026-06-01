import os
import time
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import simpy
import random

# Import modul simulasi lokal
from queue_simulation import run_queue_simulation, get_queue_statistics, compute_mmc_analytical
from gold_price_simulation import load_historical_gold_price, calculate_gbm_parameters, run_monte_carlo_forecast, get_forecast_statistics
from combined_simulation import run_combined_simulation

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Simulasi Layanan Buyback Toko Emas",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# INISIALISASI SESSION STATE TEMA (sebelum CSS dirender)
# ============================================================
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "light"

# Tentukan variabel CSS berdasarkan tema aktif
is_dark = st.session_state.theme_mode == "dark"

# ============================================================
# PALET WARNA — SESUAI SPESIFIKASI
# ============================================================
# LIGHT MODE
LT_TOPBAR     = "#E8DDD2"   # Header paling atas
LT_BG         = "#F4EAE1"   # Background utama
LT_CARD       = "#FFFFFF"   # Card / Kontainer
LT_SIDEBAR    = "#EDE0D4"   # Sidebar (turunan cream)
LT_TEXT_BIG   = "#2C1E11"   # Teks judul besar & angka
LT_TEXT_SM    = "#614B35"   # Label kecil & deskripsi
LT_ACCENT     = "#8B5A2B"   # Aksen / Highlight
LT_GOLD       = "#B38F4D"   # Garis emas / border top
LT_BORDER     = "#D2C4B7"   # Border

# DARK MODE
DK_TOPBAR     = "#140D0A"   # Header paling atas (menyatu dg bg)
DK_BG         = "#140D0A"   # Background utama
DK_CARD       = "#1F1510"   # Card / Kontainer
DK_SIDEBAR    = "#1A100C"   # Sidebar (sedikit di atas bg)
DK_TEXT_BIG   = "#E6D7CB"   # Teks judul besar & angka
DK_TEXT_SM    = "#A38A75"   # Label kecil & deskripsi
DK_ACCENT     = "#C69C6D"   # Warm Gold Highlight
DK_GOLD       = "#C69C6D"   # Garis gold / border top
DK_BORDER     = "#2E1F17"   # Border gelap

# Resolusi palet aktif
P = {
    "topbar"   : DK_TOPBAR  if is_dark else LT_TOPBAR,
    "bg"       : DK_BG      if is_dark else LT_BG,
    "card"     : DK_CARD    if is_dark else LT_CARD,
    "sidebar"  : DK_SIDEBAR if is_dark else LT_SIDEBAR,
    "text_big" : DK_TEXT_BIG if is_dark else LT_TEXT_BIG,
    "text_sm"  : DK_TEXT_SM  if is_dark else LT_TEXT_SM,
    "accent"   : DK_ACCENT  if is_dark else LT_ACCENT,
    "gold"     : DK_GOLD    if is_dark else LT_GOLD,
    "border"   : DK_BORDER  if is_dark else LT_BORDER,
    "sidebar_hover": "#241710" if is_dark else "#E3D3C5",
    "tab_hover_bg" : "#241710" if is_dark else LT_BG,
    "info_bg"  : "#1F1510"  if is_dark else "#FFFFFF",
    "shadow1"  : "rgba(0,0,0,0.25)"       if is_dark else "rgba(44,30,17,0.07)",
    "shadow2"  : "rgba(0,0,0,0.40)"       if is_dark else "rgba(139,90,43,0.12)",
    "plot_bg"  : "rgba(20,13,10,0.5)"     if is_dark else "rgba(244,234,225,0.5)",
}

# Injeksi CSS Tema Lengkap
st.markdown(f"""
<style>
    /* Font Vintage: Playfair Display (serif elegan) + Lato (body) */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Lato:wght@300;400;700&display=swap');

    /* ============================================================
       TOP BAR — Header paling atas (Streamlit toolbar)
       Target: [data-testid="stHeader"] dan turunannya
    ============================================================ */
    [data-testid="stHeader"],
    [data-testid="stHeader"] > div,
    header[data-testid="stHeader"] {{
        background-color: {P['topbar']} !important;
        border-bottom: 1px solid {P['border']} !important;
    }}
    /* Ikon-ikon di toolbar atas (hamburger, deploy, dll) */
    [data-testid="stHeader"] button,
    [data-testid="stHeader"] svg,
    [data-testid="stHeader"] span {{
        color: {P['text_sm']} !important;
        fill: {P['text_sm']} !important;
    }}
    /* Toolbar Streamlit Cloud / status bar */
    [data-testid="stToolbar"],
    [data-testid="stStatusWidget"] {{
        background-color: {P['topbar']} !important;
    }}
    [data-testid="stStatusWidget"] span,
    [data-testid="stStatusWidget"] svg {{
        color: {P['text_sm']} !important;
        fill: {P['text_sm']} !important;
    }}
    /* Decoration bar tipis di bawah top bar */
    [data-testid="stDecoration"] {{
        background-image: linear-gradient(90deg, {P['gold']}, {P['accent']}) !important;
        height: 2px !important;
    }}

    /* ============================================================
       GLOBAL BACKGROUND & TEKS
    ============================================================ */
    html, body {{
        background-color: {P['bg']} !important;
        color: {P['text_big']} !important;
    }}
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > section,
    .main,
    .block-container {{
        background-color: {P['bg']} !important;
        font-family: 'Lato', sans-serif !important;
        color: {P['text_big']} !important;
    }}
    /* Pastikan semua teks default ikut tema */
    p, span, div, li, td, th {{
        color: {P['text_big']};
    }}

    /* ============================================================
       SIDEBAR
    ============================================================ */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div {{
        background-color: {P['sidebar']} !important;
        border-right: 1px solid {P['border']} !important;
    }}
    section[data-testid="stSidebar"] * {{
        color: {P['text_sm']} !important;
    }}
    section[data-testid="stSidebar"] h3 {{
        color: {P['accent']} !important;
        font-family: 'Playfair Display', serif !important;
    }}
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stSlider p {{
        color: {P['text_sm']} !important;
        font-size: 13px !important;
    }}
    /* Slider track & thumb */
    section[data-testid="stSidebar"] [data-baseweb="slider"] [data-testid="stSliderThumbValue"],
    section[data-testid="stSidebar"] .stSlider [aria-valuenow] {{
        color: {P['accent']} !important;
    }}
    /* Sidebar scrollbar */
    section[data-testid="stSidebar"]::-webkit-scrollbar-thumb {{
        background: {P['border']} !important;
    }}

    /* ============================================================
       METRIC CARDS (Custom HTML div)
    ============================================================ */
    .metric-card {{
        background: {P['card']};
        border: 1px solid {P['border']};
        border-top: 3px solid {P['gold']};
        border-radius: 6px;
        padding: 22px 20px;
        text-align: center;
        box-shadow: 0 2px 10px {P['shadow1']};
        transition: all 0.3s ease;
        margin-bottom: 12px;
        position: relative;
    }}
    .metric-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 20px {P['shadow2']};
        border-top-color: {P['accent']};
    }}
    .metric-title {{
        font-family: 'Lato', sans-serif;
        font-size: 10px;
        font-weight: 700;
        color: {P['text_sm']} !important;
        text-transform: uppercase;
        letter-spacing: 2.5px;
        margin-bottom: 10px;
    }}
    .metric-value {{
        font-family: 'Playfair Display', serif;
        font-size: 26px;
        font-weight: 700;
        color: {P['text_big']} !important;
        margin-bottom: 6px;
    }}
    .metric-sub {{
        font-size: 11px;
        color: {P['text_sm']} !important;
        font-family: 'Lato', sans-serif;
    }}

    /* ============================================================
       ALERT / STATUS COLORS
    ============================================================ */
    .alert-green  {{ color: {'#4CAF73' if is_dark else '#3A7D44'}; font-weight: 700; }}
    .alert-yellow {{ color: {P['accent']}; font-weight: 700; }}
    .alert-red    {{ color: {'#E05C4B' if is_dark else '#A63223'}; font-weight: 700; }}

    /* ============================================================
       TITLE BANNER
    ============================================================ */
    .title-banner {{
        background: {P['card']};
        padding: 36px 32px;
        border-radius: 6px;
        text-align: center;
        border: 1px solid {P['border']};
        border-top: 4px solid {P['gold']};
        margin-bottom: 30px;
        box-shadow: 0 2px 16px {P['shadow1']};
        position: relative;
    }}
    .title-banner::before {{
        content: '';
        position: absolute;
        top: 8px; left: 8px; right: 8px; bottom: 8px;
        border: 1px solid {P['border']};
        opacity: 0.5;
        pointer-events: none;
        border-radius: 3px;
    }}
    .title-banner h1 {{
        font-family: 'Playfair Display', serif !important;
        color: {P['accent']} !important;
        font-weight: 700;
        font-size: 28px;
        letter-spacing: 0.5px;
        margin: 0;
    }}
    .title-banner p {{
        font-family: 'Lato', sans-serif;
        color: {P['text_sm']} !important;
        font-size: 13px;
        margin-top: 10px;
        margin-bottom: 0;
        letter-spacing: 0.8px;
        font-style: italic;
    }}

    /* ============================================================
       HEADINGS
    ============================================================ */
    h1, h2, h3, h4 {{
        font-family: 'Playfair Display', serif !important;
        color: {P['accent']} !important;
    }}
    /* Subheader Streamlit */
    [data-testid="stHeading"],
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {{
        color: {P['accent']} !important;
    }}

    /* ============================================================
       DIVIDER / HR
    ============================================================ */
    hr {{
        border: none !important;
        border-top: 1px solid {P['border']} !important;
        margin: 20px 0 !important;
    }}
    [data-testid="stDivider"] {{
        border-color: {P['border']} !important;
    }}
    [data-testid="stDivider"] > hr {{
        border-color: {P['border']} !important;
    }}

    /* ============================================================
       TABS — Tab list, tab item, tab panel
    ============================================================ */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        background-color: transparent !important;
        border-bottom: 2px solid {P['border']} !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 44px;
        background-color: {P['card']} !important;
        border-radius: 4px 4px 0 0;
        color: {P['text_sm']} !important;
        border: 1px solid {P['border']} !important;
        border-bottom: none !important;
        padding: 10px 20px;
        font-family: 'Lato', sans-serif;
        font-weight: 700;
        letter-spacing: 0.5px;
        font-size: 12px;
        text-transform: uppercase;
        transition: all 0.2s ease;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {P['accent']} !important;
        background-color: {P['tab_hover_bg']} !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: {'#140D0A' if is_dark else '#FFFFFF'} !important;
        background-color: {P['accent']} !important;
        border-color: {P['accent']} !important;
        border-bottom-color: {P['accent']} !important;
    }}
    /* Tab panel background */
    .stTabs [data-baseweb="tab-panel"] {{
        background-color: {P['bg']} !important;
        padding-top: 16px;
    }}

    /* ============================================================
       STREAMLIT NATIVE METRIC CONTAINERS
    ============================================================ */
    [data-testid="metric-container"] {{
        background: {P['card']} !important;
        border: 1px solid {P['border']} !important;
        border-top: 3px solid {P['gold']} !important;
        border-radius: 6px !important;
        padding: 16px !important;
        box-shadow: 0 2px 8px {P['shadow1']} !important;
    }}
    [data-testid="metric-container"] label,
    [data-testid="stMetricLabel"] p {{
        color: {P['text_sm']} !important;
        font-family: 'Lato', sans-serif !important;
        font-size: 11px !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
    }}
    [data-testid="metric-container"] [data-testid="stMetricValue"] {{
        color: {P['text_big']} !important;
        font-family: 'Playfair Display', serif !important;
        font-size: 22px !important;
    }}
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {{
        color: {P['text_sm']} !important;
    }}

    /* ============================================================
       BUTTONS
    ============================================================ */
    .stButton > button {{
        background: {P['accent']} !important;
        color: {'#140D0A' if is_dark else '#FFFFFF'} !important;
        border: 1px solid {P['accent']} !important;
        border-radius: 4px !important;
        font-family: 'Lato', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        font-size: 12px !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 2px 8px {P['shadow1']} !important;
    }}
    .stButton > button:hover {{
        filter: brightness(0.88) !important;
        box-shadow: 0 4px 14px {P['shadow2']} !important;
    }}

    /* ============================================================
       INFO / WARNING / SUCCESS / ERROR BOXES
    ============================================================ */
    [data-testid="stNotification"],
    .stAlert, div[data-baseweb="notification"] {{
        border-radius: 4px !important;
        font-family: 'Lato', sans-serif !important;
        background-color: {P['card']} !important;
        border-left-color: {P['gold']} !important;
        color: {P['text_big']} !important;
    }}
    div[data-baseweb="notification"] p,
    div[data-baseweb="notification"] span {{
        color: {P['text_big']} !important;
    }}

    /* ============================================================
       SELECTBOX / DROPDOWN / INPUT FIELDS
    ============================================================ */
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div {{
        background-color: {P['card']} !important;
        border-color: {P['border']} !important;
        color: {P['text_big']} !important;
    }}
    [data-baseweb="menu"],
    [data-baseweb="popover"] {{
        background-color: {P['card']} !important;
        border: 1px solid {P['border']} !important;
    }}
    [data-baseweb="menu"] li span,
    [data-baseweb="menu"] li div {{
        color: {P['text_big']} !important;
    }}

    /* ============================================================
       SPINNER
    ============================================================ */
    [data-testid="stSpinner"] p {{
        color: {P['accent']} !important;
    }}

    /* ============================================================
       SCROLLBAR GLOBAL
    ============================================================ */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: {P['bg']}; }}
    ::-webkit-scrollbar-thumb {{
        background: {P['border']};
        border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{ background: {P['accent']}; }}

    /* ============================================================
       DIVIDER
    ============================================================ */
    [data-testid="stDivider"] {{
        border-color: {P['border']} !important;
    }}

    /* ============================================================
       FOOTER
    ============================================================ */
    footer {{ visibility: hidden; }}

</style>
""", unsafe_allow_html=True)

# Path Data Emas (Telah disesuaikan dengan berkas baru)
CSV_PATH = "Data_Historis_GAU_IDR.csv"

# Inisialisasi State Aplikasi
if "sim_ran" not in st.session_state:
    st.session_state.sim_ran = False
    st.session_state.combined_results = None
    st.session_state.queue_df = None

# ============================================================
# VARIABEL CHART PLOTLY — SINKRON DENGAN TEMA AKTIF
# ============================================================
CHART_PLOT_BG  = P["plot_bg"]
CHART_GRID     = P["border"]
CHART_ACCENT   = P["accent"]
CHART_GOLD     = P["gold"]
CHART_TEXT_SM  = P["text_sm"]
CHART_TEMPLATE = "plotly_dark" if is_dark else "plotly_white"

# Header Banner Aplikasi (Tanpa Emotikon - Sangat Minimalis & Bersih)
st.markdown("""
<div class="title-banner">
    <h1>Analisis Sistem Transaksi & Optimalisasi Layanan Buyback Toko Emas</h1>
    <p>Model Antrean Kejadian Diskrit M/M/c Terintegrasi dengan Proyeksi Harga Monte Carlo Berbasis Data Historis Kontrak Berjangka</p>
</div>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR PARAMETERS -----------------
# --- TOMBOL TOGGLE TEMA ---
icon = "☀️ Light Mode" if is_dark else "🌑 Dark Mode"
if st.sidebar.button(icon, use_container_width=True):
    st.session_state.theme_mode = "light" if is_dark else "dark"
    st.rerun()

st.sidebar.markdown(f"<h3 style='color:{P['accent']}; text-align:center; font-weight:600; letter-spacing:0.5px; font-family:Playfair Display,serif;'>Panel Pengaturan</h3>", unsafe_allow_html=True)
st.sidebar.divider()

# Parameter 1: Sistem Antrean
st.sidebar.markdown(f"<p style='color:{P['accent']}; font-weight:600; margin-bottom:5px;'>1. Konfigurasi Antrean (M/M/c)</p>", unsafe_allow_html=True)
lambda_base = st.sidebar.slider(
    "Laju Kedatangan Pelanggan (λ / jam)",
    min_value=5.0,
    max_value=40.0,
    value=15.0,
    step=1.0,
    help="Rata-rata kedatangan pelanggan dasar per jam."
)

avg_service_time_min = st.sidebar.slider(
    "Rata-rata Waktu Pelayanan (menit)",
    min_value=2.0,
    max_value=20.0,
    value=8.0,
    step=0.5,
    help="Rata-rata waktu pelayanan teller per transaksi pelanggan."
)

c_servers = st.sidebar.slider(
    "Jumlah Teller / Staf (c)",
    min_value=1,
    max_value=5,
    value=3,
    step=1,
    help="Jumlah teller yang melayani antrean secara paralel."
)

st.sidebar.divider()

# Parameter 2: Simulasi Monte Carlo
st.sidebar.markdown(f"<p style='color:{P['accent']}; font-weight:600; margin-bottom:5px;'>2. Pergerakan Harga Emas</p>", unsafe_allow_html=True)
volatility_scale = st.sidebar.slider(
    "Skala Volatilitas Pasar (σ scale)",
    min_value=0.5,
    max_value=3.0,
    value=1.0,
    step=0.1,
    help="Pengali volatilitas historis emas. Skala > 1.0 mensimulasikan pergerakan harga bergejolak."
)

st.sidebar.divider()

# Parameter 3: Dinamika Bisnis
st.sidebar.markdown(f"<p style='color:{P['accent']}; font-weight:600; margin-bottom:5px;'>3. Dinamika Finansial & Bisnis</p>", unsafe_allow_html=True)
sensitivity_alpha = st.sidebar.slider(
    "Sensitivitas Pelanggan (α)",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.5,
    help="Seberapa kuat kenaikan harga emas mendorong peningkatan volume kedatangan pelanggan untuk buyback."
)

spread_pct = st.sidebar.slider(
    "Margin / Spread Toko (%)",
    min_value=1.0,
    max_value=10.0,
    value=3.0,
    step=0.5,
    help="Persentase di bawah harga pasar harian tempat toko membeli emas dari nasabah."
) / 100.0

avg_gram_per_trans = st.sidebar.slider(
    "Rata-rata Emas per Transaksi (gram)",
    min_value=1.0,
    max_value=50.0,
    value=10.0,
    step=0.5,
    help="Rata-rata berat emas yang dijual oleh pelanggan per transaksi."
)

# Tombol Eksekusi Utama (Gaya Minimalis)
btn_run = st.sidebar.button("Jalankan Simulasi", use_container_width=True)

# ----------------- PROSES LOGIKA SIMULASI -----------------
# Muat data awal untuk parameter historis
try:
    df_gold_hist = load_historical_gold_price(CSV_PATH)
    mu_daily, sigma_daily, last_price_idr = calculate_gbm_parameters(df_gold_hist)
except Exception as e:
    st.error(f"Gagal memuat dataset emas: {e}")
    st.info("Pastikan file 'Data_Historis_GAU_IDR.csv' berada di direktori yang sama dengan app.py")
    st.stop()

# Peringatan ketidakstabilan sistem antrean teoretis M/M/c
utilitas_teoretis = (lambda_base / 60.0) / (c_servers * (1.0 / avg_service_time_min))
if utilitas_teoretis >= 1.0:
    st.warning(f"Peringatan Sistem Tidak Stabil: Utilitas teller teoretis mencapai {utilitas_teoretis*100:.1f}%. Antrean akan mengular tanpa batas. Pertimbangkan untuk menambah teller atau mempercepat waktu pelayanan.")

if btn_run or not st.session_state.sim_ran:
    with st.spinner("Memproses kalkulasi antrean diskrit dan 1,000 skenario harga Monte Carlo..."):
        # 1. Jalankan Simulasi Gabungan Bisnis (1000 Skenario, 30 Hari)
        results = run_combined_simulation(
            historical_csv_path=CSV_PATH,
            lambda_base=lambda_base,
            avg_service_time_min=avg_service_time_min,
            c_servers=c_servers,
            volatility_scale=volatility_scale,
            sensitivity_alpha=sensitivity_alpha,
            spread_pct=spread_pct,
            avg_gram_per_trans=avg_gram_per_trans,
            std_gram_per_trans=avg_gram_per_trans * 0.3,
            scenarios=1000,
            days=30,
            random_seed=42
        )
        
        # 2. Jalankan satu simulasi antrean detail hari operasional menggunakan SimPy untuk live chart & log
        df_queue = run_queue_simulation(
            lambda_hour=lambda_base,
            avg_service_time_min=avg_service_time_min,
            c=c_servers,
            duration_hours=8.0,
            random_seed=42
        )
        
        # Simpan ke session state
        st.session_state.combined_results = results
        st.session_state.queue_df = df_queue
        st.session_state.sim_ran = True

# Ambil hasil dari session state
results = st.session_state.combined_results
df_queue = st.session_state.queue_df

# ----------------- DASHBOARD METRIK (KPI CARDS) -----------------
# Hitung metrik agregat
avg_wait_time = results["avg_waiting_times"].mean()
total_grams = results["grams_bought"].mean()
total_cash = results["cash_outflow"].mean()
avg_profit = results["profits"].mean()
p5_profit = np.percentile(results["profits"], 5)
prob_loss = np.mean(results["profits"] < 0) * 100

# Format Status Kepuasan Pelanggan berdasarkan waktu tunggu
if avg_wait_time < 3.0:
    satisfaction = "SANGAT PUAS"
    satisfaction_class = "alert-green"
elif avg_wait_time < 8.0:
    satisfaction = "NORMAL"
    satisfaction_class = "alert-green"
elif avg_wait_time < 15.0:
    satisfaction = "KURANG PUAS"
    satisfaction_class = "alert-yellow"
else:
    satisfaction = "KRITIS / OVERLOAD"
    satisfaction_class = "alert-red"

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Kepuasan & Waktu Tunggu</div>
        <div class="metric-value">{avg_wait_time:.2f} Menit</div>
        <div class="metric-sub">Status Pelayanan: <span class="{satisfaction_class}">{satisfaction}</span></div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Emas Dibeli</div>
        <div class="metric-value">{total_grams:,.0f} Gram</div>
        <div class="metric-sub">Setara dengan <b>{total_grams/1000.0:.2f} Kilogram</b></div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Modal Transaksi (Kas)</div>
        <div class="metric-value">Rp {total_cash/1e9:.2f} Milyar</div>
        <div class="metric-sub">Rata-rata modal terserap 30 hari</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    loss_class = "alert-green" if p5_profit >= 0 else "alert-red"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Rata-rata Profit Bersih</div>
        <div class="metric-value">Rp {avg_profit/1e6:,.0f} Juta</div>
        <div class="metric-sub">Risiko Kerugian (P5): <span class="{loss_class}"><b>Rp {p5_profit/1e6:,.0f} Juta</b></span></div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ----------------- MULTI-TAB DISPLAY UNTUK VISUALISASI PREMIUM -----------------
tab1, tab2, tab3 = st.tabs([
    "Analisis Antrean & Kapasitas Layanan",
    "Proyeksi Nilai Aset & Pergerakan Emas",
    "Kinerja Finansial & Evaluasi Risiko"
])

# ----------------- TAB 1: OPERASIONAL & ANTREAN -----------------
with tab1:
    col_t1_1, col_t1_2 = st.columns([6, 5])
    
    with col_t1_1:
        st.subheader("Perkembangan Panjang Antrean Pelanggan")
        st.write("Representasi visual perubahan panjang antrean selama satu hari operasional (9:00 - 17:00).")
        
        # Buat visualisasi antrean bergerak menggunakan data run SimPy
        events = []
        for index, row in df_queue.iterrows():
            events.append((row["Arrival_Time"], 1))      # Pelanggan tiba -> antrean bertambah
            events.append((row["Service_Start"], -1))   # Pelanggan dilayani -> antrean berkurang
            
        events = sorted(events, key=lambda x: x[0])
        
        times = [0.0]
        q_len = [0]
        curr_q = 0
        for t_event, change in events:
            curr_q = max(0, curr_q + change)
            times.append(t_event)
            q_len.append(curr_q)
            
        df_q_history = pd.DataFrame({"Waktu": times, "Panjang Antrean": q_len})
        
        # Filter sampel 50 baris untuk kelancaran plot gerak
        sample_indices = np.linspace(0, len(df_q_history)-1, 50, dtype=int)
        df_q_sample = df_q_history.iloc[sample_indices].reset_index(drop=True)
        
        # Area kosong untuk animasi chart
        chart_holder = st.empty()
        
        # Tombol Replay Animasi
        btn_replay = st.button("Putar Ulang Animasi Antrean")
        
        if btn_replay or "animated" not in st.session_state:
            st.session_state.animated = True
            for frame in range(2, len(df_q_sample) + 1):
                df_frame = df_q_sample.iloc[:frame]
                
                fig_anim = go.Figure()
                fig_anim.add_trace(go.Scatter(
                    x=df_frame["Waktu"],
                    y=df_frame["Panjang Antrean"],
                    mode="lines",
                    line=dict(color=CHART_ACCENT, width=2.5),
                    fill="tozeroy",
                    fillcolor=f"rgba({int(CHART_ACCENT[1:3],16)},{int(CHART_ACCENT[3:5],16)},{int(CHART_ACCENT[5:7],16)},0.06)",
                    name="Panjang Antrean"
                ))
                
                fig_anim.update_layout(
                    xaxis_title="Waktu Operasional (Menit dari pukul 09:00)",
                    yaxis_title="Pelanggan Mengantre (Orang)",
                    xaxis=dict(range=[0, 480], gridcolor=CHART_GRID),
                    yaxis=dict(range=[0, max(df_q_history["Panjang Antrean"]) + 2], gridcolor=CHART_GRID),
                    template=CHART_TEMPLATE,
                    plot_bgcolor=CHART_PLOT_BG,
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=360,
                    margin=dict(l=40, r=40, t=30, b=40)
                )
                chart_holder.plotly_chart(fig_anim, use_container_width=True)
                time.sleep(0.04) # Delay mikro-animasi
        else:
            # Tampilkan langsung chart penuh tanpa animasi
            fig_anim = go.Figure()
            fig_anim.add_trace(go.Scatter(
                x=df_q_sample["Waktu"],
                y=df_q_sample["Panjang Antrean"],
                mode="lines",
                line=dict(color=CHART_ACCENT, width=2.5),
                fill="tozeroy",
                fillcolor=f"rgba({int(CHART_ACCENT[1:3],16)},{int(CHART_ACCENT[3:5],16)},{int(CHART_ACCENT[5:7],16)},0.06)"
            ))
            fig_anim.update_layout(
                xaxis_title="Waktu Operasional (Menit dari pukul 09:00)",
                yaxis_title="Pelanggan Mengantre (Orang)",
                xaxis=dict(range=[0, 480], gridcolor=CHART_GRID),
                yaxis=dict(range=[0, max(df_q_history["Panjang Antrean"]) + 2], gridcolor=CHART_GRID),
                template=CHART_TEMPLATE,
                plot_bgcolor=CHART_PLOT_BG,
                paper_bgcolor="rgba(0,0,0,0)",
                height=360,
                margin=dict(l=40, r=40, t=30, b=40)
            )
            chart_holder.plotly_chart(fig_anim, use_container_width=True)
            
    with col_t1_2:
        st.subheader("Distribusi Waktu Tunggu Pelanggan")
        st.write("Visualisasi sebaran waktu tunggu nasabah riil dibanding perhitungan analitis M/M/c teoretis.")
        
        # Hitung statistik antrean
        stats_q = get_queue_statistics(df_queue, lambda_base, avg_service_time_min, c_servers)
        sim_avg_wait = stats_q["sim_avg_waiting_time"]
        analytical_wq = stats_q["analytical_wq"]
        
        # Buat Histogram Plotly dengan warna Champagne Gold
        fig_hist = px.histogram(
            df_queue,
            x="Waiting_Time",
            nbins=12,
            color_discrete_sequence=[CHART_GOLD],
            opacity=0.75
        )
        
        # Tambahkan garis penanda rata-rata
        fig_hist.add_vline(x=sim_avg_wait, line_dash="dash", line_color=CHART_GOLD, line_width=2,
                           annotation_text=f"Rata-rata simulasi: {sim_avg_wait:.2f} m", annotation_position="top right")
        
        if stats_q["stable"] and analytical_wq != float('inf'):
            fig_hist.add_vline(x=analytical_wq, line_dash="dot", line_color=CHART_TEXT_SM, line_width=2,
                               annotation_text=f"Analitis M/M/c: {analytical_wq:.2f} m", annotation_position="top left")
            
        fig_hist.update_layout(
            xaxis_title="Waktu Tunggu Pelanggan (Menit)",
            yaxis_title="Jumlah Pelanggan (Orang)",
            template=CHART_TEMPLATE,
            plot_bgcolor=CHART_PLOT_BG,
            paper_bgcolor="rgba(0,0,0,0)",
            height=360,
            margin=dict(l=40, r=40, t=30, b=40)
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Validasi Hasil Antrean Teoretis vs Praktis
    st.markdown("#### Validasi Model Layanan Antrean")
    col_v1, col_v2, col_v3 = st.columns(3)
    
    with col_v1:
        st.metric(
            label="Rata-rata Waktu Tunggu", 
            value=f"{sim_avg_wait:.2f} Menit",
            delta=f"{sim_avg_wait - analytical_wq:.2f} m vs Teoretis" if stats_q["stable"] else "Teoretis Tdk Stabil"
        )
        
    with col_v2:
        val_teoretis_txt = f"{analytical_wq:.2f} Menit" if stats_q["stable"] else "Tidak Terhingga"
        st.metric(
            label="Waktu Tunggu Analitis (M/M/c)", 
            value=val_teoretis_txt
        )
        
    with col_v3:
        util_sim = (df_queue["Service_Duration"].sum() / (c_servers * 480.0)) * 100 if len(df_queue) > 0 else 0.0
        util_analytical = stats_q["analytical_rho"] * 100
        st.metric(
            label="Tingkat Kesibukan Staf (Utilitas)", 
            value=f"{util_sim:.1f}% (Simulasi)",
            delta=f"{util_sim - util_analytical:.1f}% vs Teoretis"
        )

# ----------------- TAB 2: PROYEKSI HARGE EMAS MONTE CARLO -----------------
with tab2:
    st.subheader("Proyeksi Pergerakan Harga Emas Monte Carlo (Fan Chart)")
    st.write("Analisis lintasan simulasi pergerakan harga emas untuk 30 hari ke depan dalam mata uang **Rupiah per Gram**.")
    
    # Ambil matriks harga dari hasil simulasi (IDR/gram)
    price_matrix_idr = results["price_matrix_idr"]
    days_forecast = price_matrix_idr.shape[1] - 1
    scenarios_sim = price_matrix_idr.shape[0]
    
    # Hitung statistik percentiles
    stats_price = get_forecast_statistics(price_matrix_idr)
    time_steps = np.arange(0, days_forecast + 1)
    
    # Buat Plotly Fan Chart
    fig_mc = go.Figure()
    
    # 1. Tambahkan Area Batas CI 90% (p5 s.d p95) - Transparan Champagne Gold
    fig_mc.add_trace(go.Scatter(
        x=np.concatenate([time_steps, time_steps[::-1]]),
        y=np.concatenate([stats_price["p95"], stats_price["p5"][::-1]]),
        fill="toself",
        fillcolor=f"rgba({int(CHART_ACCENT[1:3],16)},{int(CHART_ACCENT[3:5],16)},{int(CHART_ACCENT[5:7],16)},0.06)",
        line=dict(color="rgba(255, 255, 255, 0)"),
        hoverinfo="skip",
        showlegend=True,
        name="Interval Kepercayaan 90% (P5 s.d P95)"
    ))
    
    # 2. Tambahkan Area Batas CI 50% (p25 s.d p75) - Emas Lebih Gelap
    fig_mc.add_trace(go.Scatter(
        x=np.concatenate([time_steps, time_steps[::-1]]),
        y=np.concatenate([stats_price["p75"], stats_price["p25"][::-1]]),
        fill="toself",
        fillcolor="rgba(139,90,43,0.1)",
        line=dict(color="rgba(255, 255, 255, 0)"),
        hoverinfo="skip",
        showlegend=True,
        name="Interval Kepercayaan 50% (P25 s.d P75)"
    ))
    
    # 3. Tampilkan Lintasan Sampel Acak (5 scenario pertama)
    for i in range(min(5, scenarios_sim)):
        fig_mc.add_trace(go.Scatter(
            x=time_steps,
            y=price_matrix_idr[i, :],
            mode="lines",
            line=dict(width=1, color="rgba(255, 255, 255, 0.2)"),
            showlegend=False if i > 0 else True,
            name="Sampel Lintasan Simulasi"
        ))
        
    # 4. Tambahkan Garis Median (Median/p50) - Emas Champagne Solid
    fig_mc.add_trace(go.Scatter(
        x=time_steps,
        y=stats_price["p50"],
        mode="lines",
        line=dict(color=CHART_ACCENT, width=3),
        name="Garis Median Proyeksi (P50)"
    ))
    
    fig_mc.update_layout(
        xaxis_title="Hari Proyeksi",
        yaxis_title="Harga Emas (Rupiah / Gram)",
        xaxis=dict(tickmode="linear", tick0=0, dtick=2, gridcolor=CHART_GRID),
        yaxis=dict(gridcolor=CHART_GRID, tickformat=",.0f"),
        template=CHART_TEMPLATE,
        plot_bgcolor=CHART_PLOT_BG,
        paper_bgcolor="rgba(0,0,0,0)",
        height=450,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.92)")
    )
    
    st.plotly_chart(fig_mc, use_container_width=True)

    # Indikator Parameter Historis
    st.markdown("#### Parameter Historis Berdasarkan Model")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.metric("Harga Terakhir Emas (Rupiah/Gram)", f"Rp {last_price_idr:,.0f} / gram")
    with col_p2:
        st.metric("Estimasi Drift Harian (μ)", f"{mu_daily*100:.5f}% / hari")
    with col_p3:
        st.metric("Volatilitas Harian (σ)", f"{sigma_daily*100:.5f}% / hari")

# ----------------- TAB 3: KINERJA KEUANGAN & MANAJEMEN RISIKO -----------------
with tab3:
    col_t3_1, col_t3_2 = st.columns([6, 5])
    
    with col_t3_1:
        st.subheader("Distribusi Profit Bersih 30 Hari (Box/Violin)")
        st.write("Menampilkan spektrum keuntungan atau potensi kerugian bersih dari 1,000 skenario transaksi.")
        
        # Buat Violin & Box Plot gabungan bertema Emas Champagne
        fig_violin = go.Figure()
        
        fig_violin.add_trace(go.Violin(
            y=results["profits"],
            box_visible=True,
            meanline_visible=True,
            fillcolor="rgba(139,90,43,0.1)",
            line=dict(color="#B38F4D", width=2),
            marker=dict(color="#B38F4D"),
            name="Profit Bersih (Rp)"
        ))
        
        fig_violin.update_layout(
            yaxis_title="Profit Bersih (Rupiah)",
            yaxis=dict(gridcolor=CHART_GRID, tickformat=",.0f"),
            template=CHART_TEMPLATE,
            plot_bgcolor=CHART_PLOT_BG,
            paper_bgcolor="rgba(0,0,0,0)",
            height=380,
            margin=dict(l=40, r=40, t=20, b=40)
        )
        st.plotly_chart(fig_violin, use_container_width=True)
        
    with col_t3_2:
        st.subheader("Evaluasi Risiko Finansial (Value at Risk - VaR)")
        st.write("Evaluasi paparan risiko keuangan modal akibat fluktuasi pasar emas pasca-buyback.")
        
        # Hitung statistik profit
        profits = results["profits"]
        p5 = np.percentile(profits, 5)
        p50 = np.percentile(profits, 50)
        p95 = np.percentile(profits, 95)
        
        # UI Tampilan Laporan Risiko
        st.markdown(f"""
        <div style="background-color: {P['card']}; border-radius: 12px; padding: 20px; border-left: 4px solid {P['gold']}; border: 1px solid {P['border']};">
            <p style="margin:0; font-size:13.5px; color:{P['text_big']}; line-height:1.6;">
            Berdasarkan simulasi Monte Carlo, probabilitas toko mengalami kerugian akibat depresiasi harga emas di akhir bulan adalah sebesar <b>{prob_loss:.1f}%</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.markdown(f"""
        * **Estimasi Realistis Profit (P50 - Median)**: `Rp {p50:,.0f}` 
          *(Keuntungan paling moderat yang diharapkan).*
        * **Potensi Keuntungan Tinggi (P95)**: `Rp {p95:,.0f}` 
          *(Jika pasar berada dalam tren kenaikan harga yang agresif).*
        * **Value at Risk Kas Toko (P5)**: `Rp {p5:,.0f}` 
          *(Tingkat keyakinan 95% menunjukkan risiko penurunan/kerugian maksimal tidak akan melebihi angka ini).*
        """)
        
        # Analisis kapasitas teller untuk optimalisasi layanan
        utilitas = (lambda_base / 60.0) / (c_servers * (1.0 / avg_service_time_min))
        st.subheader("Evaluasi Efisiensi Layanan")
        if utilitas < 0.5:
            st.info(f"Kapasitas Layanan Longgar (Utilitas {utilitas*100:.1f}%): Jumlah staf melebihi kebutuhan. Disarankan mengurangi jumlah teller aktif untuk efisiensi biaya operasional.")
        elif utilitas <= 0.85:
            st.success(f"Kapasitas Layanan Optimal (Utilitas {utilitas*100:.1f}%): Keseimbangan ideal antara efisiensi kerja staf dan waktu antrean nasabah. Konfigurasi sangat direkomendasikan.")
        else:
            st.warning(f"Kapasitas Layanan Kritis (Utilitas {utilitas*100:.1f}%): Antrean menumpuk akibat keterbatasan staf. Sangat disarankan menambah teller untuk memangkas waktu tunggu nasabah.")

# Footer Presentasi Kelompok (Sangat Bersih)
st.divider()
st.markdown(f"<p style='text-align: center; color: {P['text_sm']}; font-size: 12px; font-weight: 500;'>Simulasi Sistem Transaksi & Optimalisasi Layanan Buyback Toko Emas - 2026</p>", unsafe_allow_html=True)
