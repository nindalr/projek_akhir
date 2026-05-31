import numpy as np
import pandas as pd
from gold_price_simulation import run_monte_carlo_forecast, load_historical_gold_price, calculate_gbm_parameters

# Data historis sudah dalam IDR/gram — tidak perlu konversi USD

def simulate_fast_queue_day(lambda_hour, avg_service_time_min, c, duration_hours=8.0):
    """
    Simulasi antrean cepat (M/M/c) dalam pure NumPy.
    Sangat efisien untuk dijalankan ribuan kali dalam simulasi Monte Carlo.
    """
    duration_min = duration_hours * 60.0
    lam = lambda_hour / 60.0
    mu = 1.0 / avg_service_time_min

    expected_cust = int(lam * duration_min + 3 * np.sqrt(lam * duration_min))
    expected_cust = max(10, expected_cust)

    inter_arrivals = np.random.exponential(1.0 / lam, expected_cust)
    arrival_times = np.cumsum(inter_arrivals)
    arrival_times = arrival_times[arrival_times <= duration_min]
    num_cust = len(arrival_times)

    if num_cust == 0:
        return 0, 0.0, 0.0

    service_durations = np.random.exponential(avg_service_time_min, num_cust)

    teller_free_time = np.zeros(c)
    waiting_times = np.zeros(num_cust)

    for i in range(num_cust):
        arr = arrival_times[i]
        idx = np.argmin(teller_free_time)

        if teller_free_time[idx] <= arr:
            waiting_times[i] = 0.0
            teller_free_time[idx] = arr + service_durations[i]
        else:
            waiting_times[i] = teller_free_time[idx] - arr
            teller_free_time[idx] += service_durations[i]

    avg_wait = np.mean(waiting_times)
    max_wait = np.max(waiting_times)

    return num_cust, avg_wait, max_wait

def run_combined_simulation(
    historical_csv_path,
    lambda_base,
    avg_service_time_min,
    c_servers,
    volatility_scale=1.0,
    sensitivity_alpha=2.0,
    spread_pct=0.03,
    avg_gram_per_trans=10.0,
    std_gram_per_trans=3.0,
    scenarios=1000,
    days=30,
    random_seed=42
):
    """
    Simulasi gabungan: Monte Carlo harga emas IDR/gram + simulasi antrean dinamis.
    Data historis sudah dalam IDR/gram, sehingga tidak diperlukan konversi USD.
    """
    np.random.seed(random_seed)

    # 1. Muat data historis (sudah IDR/gram) dan hitung parameter GBM
    df_gold = load_historical_gold_price(historical_csv_path)
    mu_daily, sigma_daily, last_price = calculate_gbm_parameters(df_gold)

    # 2. Jalankan Monte Carlo — harga sudah dalam IDR/gram
    price_matrix_idr = run_monte_carlo_forecast(
        start_price=last_price,
        mu_daily=mu_daily,
        sigma_daily=sigma_daily,
        days=days,
        scenarios=scenarios,
        volatility_scale=volatility_scale,
        random_seed=random_seed
    )

    # 3. Inisialisasi matriks hasil
    scenario_profits         = np.zeros(scenarios)
    scenario_grams_bought    = np.zeros(scenarios)
    scenario_cash_outflow    = np.zeros(scenarios)
    scenario_avg_waiting_time = np.zeros(scenarios)
    scenario_total_customers = np.zeros(scenarios)

    for s in range(scenarios):
        total_grams      = 0.0
        total_cash_spent = 0.0
        total_wait_time  = 0.0
        total_cust       = 0

        for t in range(1, days + 1):
            price_today     = price_matrix_idr[s, t]
            price_yesterday = price_matrix_idr[s, t - 1]

            daily_return    = (price_today - price_yesterday) / price_yesterday
            effective_return = max(0.0, daily_return)
            lambda_t = lambda_base * (1.0 + sensitivity_alpha * effective_return)

            num_cust, avg_wait, _ = simulate_fast_queue_day(
                lambda_hour=lambda_t,
                avg_service_time_min=avg_service_time_min,
                c=c_servers,
                duration_hours=8.0
            )

            if num_cust > 0:
                grams_sold = np.random.normal(avg_gram_per_trans, std_gram_per_trans, num_cust)
                grams_sold = np.clip(grams_sold, 1.0, None)

                day_grams           = np.sum(grams_sold)
                buyback_price_gram  = price_today * (1.0 - spread_pct)
                cash_spent          = day_grams * buyback_price_gram

                total_grams      += day_grams
                total_cash_spent += cash_spent
                total_wait_time  += avg_wait * num_cust
                total_cust       += num_cust

        final_price_day30 = price_matrix_idr[s, days]
        asset_value_day30 = total_grams * final_price_day30
        net_profit        = asset_value_day30 - total_cash_spent

        scenario_profits[s]          = net_profit
        scenario_grams_bought[s]     = total_grams
        scenario_cash_outflow[s]     = total_cash_spent
        scenario_avg_waiting_time[s] = total_wait_time / total_cust if total_cust > 0 else 0.0
        scenario_total_customers[s]  = total_cust

    return {
        "price_matrix_idr":  price_matrix_idr,
        "profits":           scenario_profits,
        "grams_bought":      scenario_grams_bought,
        "cash_outflow":      scenario_cash_outflow,
        "avg_waiting_times": scenario_avg_waiting_time,
        "total_customers":   scenario_total_customers,
        "last_price":        last_price,
        "mu_daily":          mu_daily,
        "sigma_daily":       sigma_daily,
    }

if __name__ == "__main__":
    print("=" * 60)
    print("DEMO SIMULASI GABUNGAN TOKO EMAS (STANDALONE)")
    print("=" * 60)

    csv_path = "Data_Historis_GAU_IDR.csv"

    try:
        results = run_combined_simulation(
            historical_csv_path=csv_path,
            lambda_base=15.0,
            avg_service_time_min=8.0,
            c_servers=3,
            volatility_scale=1.0,
            sensitivity_alpha=5.0,
            spread_pct=0.03,
            scenarios=500,
            days=30,
            random_seed=42
        )

        print("Hasil Simulasi Finansial & Operasional Toko Emas (30 Hari Proyeksi):")
        print(f"  - Harga Awal Emas             : Rp {results['last_price']:,.0f} / gram")
        print(f"  - Rata-rata Pelanggan         : {results['total_customers'].mean():.1f} orang")
        print(f"  - Rata-rata Waktu Tunggu      : {results['avg_waiting_times'].mean():.2f} menit")
        print(f"  - Rata-rata Emas Dibeli       : {results['grams_bought'].mean():.2f} gram")
        print(f"  - Rata-rata Modal Kas Keluar  : Rp {results['cash_outflow'].mean():,.0f}")
        print(f"  - Rata-rata Estimasi Profit   : Rp {results['profits'].mean():,.0f}")

        p5  = np.percentile(results["profits"], 5)
        p95 = np.percentile(results["profits"], 95)

        print("-" * 60)
        print("Analisis Risiko (VaR):")
        print(f"  - Skenario Terburuk P5  : Rp {p5:,.0f}  ({'KERUGIAN' if p5 < 0 else 'KEUNTUNGAN MINIMAL'})")
        print(f"  - Skenario Terbaik  P95 : Rp {p95:,.0f}")
        print(f"  - Prob. Rugi            : {np.mean(results['profits'] < 0) * 100:.1f}%")
        print("=" * 60)

    except Exception as e:
        print(f"Error: {e}")
