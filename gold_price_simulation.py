import os
import pandas as pd
import numpy as np

def load_historical_gold_price(csv_path="Data_Historis_GAU_IDR.csv"):
    """
    Membaca data historis harga emas GAU dalam IDR/gram dan melakukan pembersihan data.
    Format CSV: Tanggal, Terakhir, Pembukaan, Tertinggi, Terendah, Vol., Perubahan%
    Angka menggunakan format Indonesia (titik = pemisah ribuan, koma = desimal).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File data historis tidak ditemukan di path: {csv_path}")

    # Baca CSV — kolom sudah ada headernya di baris pertama
    df = pd.read_csv(csv_path)

    # Standarisasi nama kolom (hapus spasi/BOM tersembunyi)
    df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=False).str.replace('"', '')

    # Ambil hanya kolom Tanggal dan Terakhir (harga penutupan)
    df = df[["Tanggal", "Terakhir"]].copy()
    df.columns = ["Date", "Close"]

    # Bersihkan nilai string: hapus tanda petik, titik ribuan, ganti koma desimal → titik
    df["Close"] = (
        df["Close"]
        .astype(str)
        .str.replace('"', '', regex=False)
        .str.replace('.', '', regex=False)   # hapus pemisah ribuan
        .str.replace(',', '.', regex=False)  # koma desimal → titik
    )
    df["Date"] = df["Date"].astype(str).str.replace('"', '', regex=False).str.strip()

    # Konversi ke numerik dan tanggal
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    # Hapus baris dengan nilai kosong dan urutkan kronologis
    df = df.dropna().reset_index(drop=True)
    df = df.sort_values(by="Date").reset_index(drop=True)

    return df

def calculate_gbm_parameters(df):
    """
    Menghitung parameter µ dan σ harian dari log return data historis (IDR/gram).
    """
    df = df.copy()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    returns = df["Log_Return"].dropna()

    mu_daily = returns.mean()
    sigma_daily = returns.std()

    return mu_daily, sigma_daily, df["Close"].iloc[-1]

def run_monte_carlo_forecast(start_price, mu_daily, sigma_daily, days=30, scenarios=1000,
                              volatility_scale=1.0, random_seed=None):
    """
    Menjalankan simulasi Monte Carlo untuk harga emas 30 hari ke depan.
    Harga dalam IDR/gram (tidak ada konversi USD).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    sigma_sim = sigma_daily * volatility_scale

    simulation_matrix = np.zeros((scenarios, days + 1))
    simulation_matrix[:, 0] = start_price

    drift = mu_daily - 0.5 * (sigma_sim ** 2)

    for t in range(1, days + 1):
        Z = np.random.standard_normal(scenarios)
        simulation_matrix[:, t] = simulation_matrix[:, t - 1] * np.exp(drift + sigma_sim * Z)

    return simulation_matrix

def get_forecast_statistics(simulation_matrix):
    """
    Menghitung interval kepercayaan (P5, P25, P50, P75, P95) dari hasil simulasi.
    """
    percentiles = {
        "p5":  np.percentile(simulation_matrix, 5,  axis=0),
        "p25": np.percentile(simulation_matrix, 25, axis=0),
        "p50": np.percentile(simulation_matrix, 50, axis=0),
        "p75": np.percentile(simulation_matrix, 75, axis=0),
        "p95": np.percentile(simulation_matrix, 95, axis=0),
    }
    return percentiles

if __name__ == "__main__":
    print("=" * 60)
    print("DEMO SIMULASI HARGA EMAS MONTE CARLO (STANDALONE)")
    print("=" * 60)

    csv_path = "Data_Historis_GAU_IDR.csv"

    try:
        df_gold = load_historical_gold_price(csv_path)
        print(f"Dataset berhasil dimuat!")
        print(f"  - Rentang Tanggal : {df_gold['Date'].min().strftime('%Y-%m-%d')} s.d {df_gold['Date'].max().strftime('%Y-%m-%d')}")
        print(f"  - Jumlah Data     : {len(df_gold)} baris")
        print(f"  - Harga Terakhir  : Rp {df_gold['Close'].iloc[-1]:,.0f} / gram")

        mu, sigma, last_price = calculate_gbm_parameters(df_gold)
        print(f"\nParameter Terestimasi (Harian):")
        print(f"  - Drift (mu)       : {mu:.6f}")
        print(f"  - Volatilitas (sig): {sigma:.6f}")
        print("-" * 60)

        scenarios_count = 1000
        forecast_days = 30
        sim_matrix = run_monte_carlo_forecast(
            start_price=last_price,
            mu_daily=mu,
            sigma_daily=sigma,
            days=forecast_days,
            scenarios=scenarios_count,
            volatility_scale=1.0,
            random_seed=42
        )

        stats = get_forecast_statistics(sim_matrix)

        print(f"Proyeksi Simulasi Monte Carlo ({scenarios_count} Skenario, {forecast_days} Hari Ke Depan):")
        print(f"  - Harga Hari ke-0  : Rp {sim_matrix[0, 0]:,.0f} / gram")
        print(f"  - Hasil Hari ke-30 (Prediksi):")
        print(f"    * Persentil  5  (Batas Bawah 90% CI) : Rp {stats['p5'][-1]:,.0f}")
        print(f"    * Persentil 50  (Median)             : Rp {stats['p50'][-1]:,.0f}")
        print(f"    * Persentil 95  (Batas Atas 90% CI)  : Rp {stats['p95'][-1]:,.0f}")
        print("-" * 60)

        print("Sampel Harga 5 Skenario Acak (Hari ke 0, 10, 20, 30):")
        for i in range(5):
            s = sim_matrix[i, [0, 10, 20, 30]]
            print(f"  Skenario {i+1:2d} : D0: Rp {s[0]:,.0f} -> D10: Rp {s[1]:,.0f} -> D20: Rp {s[2]:,.0f} -> D30: Rp {s[3]:,.0f}")
        print("=" * 60)

    except Exception as e:
        print(f"Error saat menjalankan simulasi: {e}")
