import random
import math
import pandas as pd
import numpy as np
import simpy

def compute_mmc_analytical(lambda_hour, avg_service_time_min, c):
    """
    Menghitung metrik antrean teoretis M/M/c.
    lambda_hour: Kedatangan pelanggan per jam
    avg_service_time_min: Rata-rata waktu pelayanan dalam menit
    c: Jumlah teller/staf
    """
    # Ubah ke satuan per menit
    lam = lambda_hour / 60.0
    mu = 1.0 / avg_service_time_min
    
    # Traffic intensity (utilitas)
    rho = lam / (c * mu)
    
    if rho >= 1.0:
        return {
            "stable": False,
            "rho": rho,
            "P0": None,
            "Pq": None,
            "Wq_min": float('inf'),
            "W_min": float('inf')
        }
    
    # Hitung P0 (probabilitas 0 pelanggan di sistem)
    sum_terms = 0.0
    for n in range(c):
        sum_terms += ((lam / mu) ** n) / math.factorial(n)
        
    prob_queue_term = ((lam / mu) ** c) / (math.factorial(c) * (1.0 - rho))
    P0 = 1.0 / (sum_terms + prob_queue_term)
    
    # Hitung Pq (probabilitas mengantre / Erlang C)
    Pq = prob_queue_term * P0
    
    # Rata-rata waktu tunggu di antrean (Wq) dalam menit
    Wq = Pq / (c * mu - lam)
    
    # Rata-rata waktu total di sistem (W) dalam menit
    W = Wq + (1.0 / mu)
    
    return {
        "stable": True,
        "rho": rho,
        "P0": P0,
        "Pq": Pq,
        "Wq_min": Wq,
        "W_min": W
    }

def run_queue_simulation(lambda_hour, avg_service_time_min, c, duration_hours=8.0, random_seed=None):
    """
    Menjalankan simulasi antrean SimPy untuk satu hari operasional.
    lambda_hour: tingkat kedatangan pelanggan per jam
    avg_service_time_min: rata-rata waktu pelayanan per pelanggan (menit)
    c: jumlah teller
    duration_hours: durasi simulasi (jam)
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        
    env = simpy.Environment()
    teller = simpy.Resource(env, capacity=c)
    
    # List untuk menyimpan log pelanggan
    customer_logs = []
    
    # Parameter laju (per menit)
    lam = lambda_hour / 60.0
    mu = 1.0 / avg_service_time_min
    
    def customer_process(env, name, teller):
        arrival_time = env.now
        
        # Request teller
        with teller.request() as req:
            yield req
            start_service = env.now
            waiting_time = start_service - arrival_time
            
            # Waktu pelayanan berdistribusi eksponensial
            service_duration = random.expovariate(mu)
            yield env.timeout(service_duration)
            
            end_service = env.now
            
            customer_logs.append({
                "Customer_ID": name,
                "Arrival_Time": arrival_time,          # Menit ke-
                "Service_Start": start_service,        # Menit ke-
                "Service_End": end_service,            # Menit ke-
                "Waiting_Time": waiting_time,          # Menit
                "Service_Duration": service_duration   # Menit
            })
            
    def customer_generator(env, teller):
        i = 1
        while True:
            # Waktu antar-kedatangan berdistribusi eksponensial (Poisson process)
            inter_arrival = random.expovariate(lam)
            yield env.timeout(inter_arrival)
            
            env.process(customer_process(env, f"Cust_{i}", teller))
            i += 1
            
    # Mulai generator kedatangan pelanggan
    env.process(customer_generator(env, teller))
    
    # Jalankan simulasi (konversi jam ke menit)
    sim_duration_minutes = duration_hours * 60.0
    env.run(until=sim_duration_minutes)
    
    # Buat DataFrame dari hasil log
    if len(customer_logs) > 0:
        df = pd.DataFrame(customer_logs)
        df = df.sort_values(by="Arrival_Time").reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=["Customer_ID", "Arrival_Time", "Service_Start", "Service_End", "Waiting_Time", "Service_Duration"])
        
    return df

def get_queue_statistics(df, lambda_hour, avg_service_time_min, c):
    """
    Menghitung statistik ringkasan hasil simulasi dan membandingkannya dengan analitis.
    """
    analytical = compute_mmc_analytical(lambda_hour, avg_service_time_min, c)
    
    if len(df) == 0:
        return {
            "total_customers": 0,
            "sim_avg_waiting_time": 0.0,
            "sim_max_waiting_time": 0.0,
            "sim_avg_service_duration": 0.0,
            "analytical_wq": analytical["Wq_min"] if analytical["stable"] else float('inf'),
            "analytical_rho": analytical["rho"],
            "stable": analytical["stable"]
        }
        
    stats = {
        "total_customers": len(df),
        "sim_avg_waiting_time": df["Waiting_Time"].mean(),
        "sim_max_waiting_time": df["Waiting_Time"].max(),
        "sim_avg_service_duration": df["Service_Duration"].mean(),
        "analytical_wq": analytical["Wq_min"] if analytical["stable"] else float('inf'),
        "analytical_rho": analytical["rho"],
        "stable": analytical["stable"]
    }
    
    return stats

if __name__ == "__main__":
    print("="*60)
    print("DEMO SIMULASI ANTREAN TOKO EMAS (STANDALONE)")
    print("="*60)
    
    # Parameter default
    lambda_val = 15.0      # 15 pelanggan per jam
    avg_service = 10.0     # Rata-rata pelayanan 10 menit
    c_servers = 3          # 3 teller/staf
    duration = 8.0         # 8 jam kerja
    
    print(f"Parameter Input:")
    print(f"  - Laju Kedatangan (lambda)    : {lambda_val} pelanggan/jam")
    print(f"  - Rata-rata Pelayanan (1/mu)  : {avg_service} menit/pelanggan")
    print(f"  - Jumlah Teller (c)           : {c_servers}")
    print(f"  - Durasi Simulasi             : {duration} jam")
    print("-" * 60)
    
    # Jalankan simulasi
    df_result = run_queue_simulation(lambda_val, avg_service, c_servers, duration_hours=duration, random_seed=42)
    
    # Hitung statistik
    results = get_queue_statistics(df_result, lambda_val, avg_service, c_servers)
    
    print("Hasil Simulasi:")
    print(f"  - Total Pelanggan Dilayani    : {results['total_customers']}")
    print(f"  - Rata-rata Waktu Tunggu      : {results['sim_avg_waiting_time']:.4f} menit")
    print(f"  - Waktu Tunggu Maksimum       : {results['sim_max_waiting_time']:.4f} menit")
    print(f"  - Rata-rata Waktu Pelayanan   : {results['sim_avg_service_duration']:.4f} menit")
    print("-" * 60)
    
    print("Validasi Teoretis M/M/c Analitis:")
    print(f"  - Utilitas Staf (rho)         : {results['analytical_rho']*100:.2f}%")
    print(f"  - Status Sistem               : {'STABIL' if results['stable'] else 'TIDAK STABIL'}")
    if results['stable']:
        print(f"  - Rata-rata Wq Teoretis       : {results['analytical_wq']:.4f} menit")
        selisih = abs(results['sim_avg_waiting_time'] - results['analytical_wq'])
        persen_error = (selisih / results['analytical_wq']) * 100 if results['analytical_wq'] > 0 else 0
        print(f"  - Selisih (Error)             : {selisih:.4f} menit ({persen_error:.2f}%)")
    else:
        print("  - PERINGATAN: Sistem tidak stabil! Utilitas >= 100%. Antrean akan terus bertambah.")
    print("="*60)
    
    # Cetak 5 sampel pertama data pelanggan
    print("\nSampel Data Pelanggan (5 Pertama):")
    print(df_result.head().to_string(index=False))
    print("="*60)
