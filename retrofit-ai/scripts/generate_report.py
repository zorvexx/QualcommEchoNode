import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import argparse
from fpdf import FPDF
from src.inference.inference import RetroFitInferencePipeline

def generate_pdf_report(inference_output, output_pdf_path="data/reports/Machine_Intelligence_Report.pdf"):
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "RetroFit Machine Intelligence Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 8, "Self-Learning Machine Behavioral Identity & Edge AI Summary", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)
    
    # Section 1: Executive Summary
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "1. Executive Status & Identity", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    
    pdf.cell(0, 6, f"Machine ID: {inference_output.get('machine_id', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Operating State: State {inference_output.get('state', 0)}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"System Status: {inference_output.get('status', 'NORMAL')}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Behavioral Similarity: {inference_output.get('similarity', 100.0)}%", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Behavioral Drift: {inference_output.get('behavior_drift', 0.0)}%", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Confidence Metric: {inference_output.get('confidence', 95.0)}%", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # Section 2: Sensor Modality Attribution
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "2. Modality Contribution Breakdown", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    
    mod = inference_output.get('modality_contribution', {})
    for k, v in mod.items():
        pdf.cell(0, 6, f" - {k.capitalize()}: {v * 100:.1f}% contribution", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # Section 3: Feature Contributors
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "3. Primary Feature Deviations", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    
    top_f = inference_output.get('top_features', [])
    pdf.cell(0, 6, f"Top Deviating Features: {', '.join(top_f)}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # Section 4: Machine Personality
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "4. Machine Personality & Behavior Profile", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    
    pers = inference_output.get('machine_personality', {})
    for k, v in pers.items():
        pdf.cell(0, 6, f" - {k.replace('_', ' ').capitalize()}: {v}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # Section 5: Recommendations
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "5. Engineering Recommendations", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    
    if inference_output.get('behavior_drift', 0.0) > 20.0:
        rec = "ATTENTION: Behavioral drift detected. Inspect mechanical alignment and lubrication."
    else:
        rec = "NORMAL: Machine is operating within learned healthy identity baseline."
        
    pdf.multi_cell(0, 6, rec)
    
    pdf.output(output_pdf_path)
    print(f"[REPORT] Successfully generated PDF Machine Intelligence Report -> {output_pdf_path}")
    return output_pdf_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="data/models")
    args = parser.parse_args()
    
    import pandas as pd
    from src.features.extractor import extract_features_from_dataframe
    import yaml
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    df_raw = pd.read_csv("data/raw/demo_machine.csv")
    df_feat = extract_features_from_dataframe(df_raw, config)
    
    pipeline = RetroFitInferencePipeline(models_dir=args.models)
    output = pipeline.predict_window(df_feat.tail(1))
    
    generate_pdf_report(output)
