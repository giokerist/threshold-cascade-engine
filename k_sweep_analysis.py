import json
import subprocess
import os
import tempfile
import matplotlib.pyplot as plt
import numpy as np

# Configuration
BASE_CONFIG = "cascade_engine/config_stochastic_er.json"
K_VALUES = [2, 5, 10, 20, 40, 80, 150, 300]
OUTPUT_BASE = "results_k_sweep"

results = []

print(f"{'k-Value':<10} | {'Spearman Rho':<15} | {'RMSE':<10}")
print("-" * 40)

for k in K_VALUES:
    with open(BASE_CONFIG, 'r') as f:
        config = json.load(f)
    
    config['stochastic_k'] = k
    tmp_fd, temp_config_path = tempfile.mkstemp(suffix=f"_k{k}.json", prefix="cascade_")
    with os.fdopen(tmp_fd, 'w') as f:
        json.dump(config, f)

    output_dir = os.path.join(OUTPUT_BASE, f"k_{k}")
    cmd = ["python3", "-m", "cascade_engine.runner", temp_config_path, "--output-dir", output_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  [WARNING] subprocess failed for k={k} (exit {proc.returncode}):")
        for line in proc.stderr.strip().splitlines()[-5:]:
            print(f"    {line}")
    
    summary_path = os.path.join(output_dir, "summary.json")
    try:
        with open(summary_path, 'r') as f:
            summary = json.load(f)
            
            # Match your exact JSON keys
            rho = summary.get('spearman_rho')
            rmse = summary.get('rmse_det_vs_stochastic')
            
            # Handle NaN/None values for low k (spearman undefined when all cascade sizes equal)
            if rho is None or (isinstance(rho, float) and np.isnan(rho)):
                rho = 0.0
            if rmse is None or (isinstance(rmse, float) and np.isnan(rmse)):
                rmse = float('nan')
                
            results.append((k, rho, rmse))
            print(f"{k:<10} | {rho:<15.4f} | {rmse:<10.4f}")
    except Exception as e:
        print(f"Error for k={k}: {e}")
    
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)

# Plotting
if results:
    ks, rhos, rmses = zip(*results)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(ks, rhos, marker='o', color='blue')
    plt.title('Convergence: Spearman $\\rho \\to 1.0$')
    plt.xlabel('Steepness (k)')
    plt.ylabel('Rank Correlation')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(ks, rmses, marker='s', color='red')
    plt.title('Error: RMSE $\\to 0$')
    plt.xlabel('Steepness (k)')
    plt.ylabel('RMSE')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("k_sweep_results.png")
    plt.show()
