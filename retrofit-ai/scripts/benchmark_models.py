import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
from scripts.extract_real_features import run_real_feature_extraction
from src.models.benchmark import benchmark_models_on_real_data

def run_real_benchmark(csv_path=r"C:\Users\rakes\Downloads\mlx90614_dataset_converted.csv", config_path="config.yaml"):
    print("=========================================================")
    print("   RETROFIT REAL DATA MODEL SELECTION BENCHMARK         ")
    print("=========================================================")
    
    real_features_csv = "data/features/real_features.csv"
    if not os.path.exists(real_features_csv):
        df_features = run_real_feature_extraction(csv_path, config_path, real_features_csv)
    else:
        df_features = pd.read_csv(real_features_csv)
        
    df_benchmark = benchmark_models_on_real_data(df_features, top_n=30)
    
    print("\n--- REAL DATA MODEL BENCHMARK RESULTS ---")
    print(df_benchmark.to_string(index=False))
    
    out_csv = "data/models/real_model_benchmark.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_benchmark.to_csv(out_csv, index=False)
    print(f"\n[BENCHMARK] Saved real model evaluation benchmark to {out_csv}")
    
    # Generate HTML Report
    html_out = "data/reports/real_model_benchmark.html"
    os.makedirs(os.path.dirname(html_out), exist_ok=True)
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>RetroFit Real Model Selection Benchmark</title>
    <style>
        body {{ font-family: sans-serif; background-color: #0f172a; color: #f8fafc; padding: 20px; }}
        h1 {{ color: #3b82f6; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ border: 1px solid #334155; padding: 10px; text-align: left; }}
        th {{ background-color: #1e293b; color: #94a3b8; }}
        tr:nth-child(even) {{ background-color: #1e293b; }}
    </style>
</head>
<body>
    <h1>RetroFit Real Data Model Benchmark Report</h1>
    <p>Evaluated on real held-out session dataset (<code>mlx90614_dataset_converted.csv</code>) using strict leak-free temporal split.</p>
    {df_benchmark.to_html(index=False, classes='table')}
</body>
</html>"""
    with open(html_out, 'w') as f:
        f.write(html_content)
    print(f"[BENCHMARK HTML] Generated HTML Benchmark Report -> {html_out}")
    
    print("\n=========================================================")
    print("         REAL MODEL BENCHMARK COMPLETE SUCCESS!          ")
    print("=========================================================")

if __name__ == "__main__":
    run_real_benchmark()
