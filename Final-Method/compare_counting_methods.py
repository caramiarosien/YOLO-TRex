#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_counting_methods.py
===========================
Central script for merging and statistical evaluation of counting data:
1. Loads YOLO results (16Nov & 17Nov), Koger reference data (Excel), and brightness data.
2. Creates and saves the consolidated master comparison CSV to:
   /Users/cara/Desktop/BA/Method_YOLO/Final-Output/master_counting_comparison.csv
3. Calculates pooled error and correlation metrics (Bias, MAE, RMSE, rel. error %, Pearson, Spearman)
   for YOLO vs. Manual and Koger vs. Manual – both overall and per brightness category.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# ===========================================================================
# ⚙️ CONFIGURATION — File Paths
# ===========================================================================
YOLO_16NOV = "/Users/cara/Desktop/BA/Method_YOLO/Final-Output/counting-16Nov-summary.csv"
YOLO_17NOV = "/Users/cara/Desktop/BA/Method_YOLO/Final-Output/counting-17Nov-summary.csv"
KOGER_EXCEL = "/Users/cara/Desktop/BA/Koger-Method/bat-counting-error-falloff.xlsx"
BRIGHTNESS_16NOV = "/Users/cara/Desktop/BA/Data/2019_brightness/video_brightness-16Nov19.csv"
BRIGHTNESS_17NOV = "/Users/cara/Desktop/BA/Data/2019_brightness/video_brightness-17Nov19.csv"

MASTER_OUTPUT_CSV = "/Users/cara/Desktop/BA/Method_YOLO/Final-Output/master_counting_comparison.csv"


# ---------------------------------------------------------------------------
# 1. Load and Merge Data (Create Master CSV)
# ---------------------------------------------------------------------------
def build_master_dataframe() -> pd.DataFrame:
    """Loads all data sources and merges them into a master table."""
    print("[1/3] Loading data sources...")

    # A) YOLO data
    df_y16 = pd.read_csv(YOLO_16NOV)
    df_y17 = pd.read_csv(YOLO_17NOV)
    df_yolo = pd.concat([df_y16, df_y17], ignore_index=True)
    df_yolo['norm_id'] = df_yolo['Video name'].astype(str).str.replace('.mp4', '', regex=False)

    # B) Koger & Manual Counts (Excel)
    df_koger = pd.read_excel(KOGER_EXCEL, header=1)
    df_koger['norm_id'] = df_koger['video-clip-name'].astype(str).str.replace('.mp4', '', regex=False)

    # C) Brightness data
    df_b16 = pd.read_csv(BRIGHTNESS_16NOV)
    df_b17 = pd.read_csv(BRIGHTNESS_17NOV)
    df_bright = pd.concat([df_b16, df_b17], ignore_index=True)
    df_bright['norm_id'] = df_bright['Videoname'].astype(str).str.replace('.mp4', '', regex=False)

    # Merge via norm_id
    df_merged = pd.merge(
        df_koger,
        df_yolo[['norm_id', 'Counting forward', 'Counting backward', 'Nett count']],
        on='norm_id',
        how='inner'
    )
    df_merged = pd.merge(
        df_merged,
        df_bright[['norm_id', 'Location', 'Helligkeit', 'Kategorie']],
        on='norm_id',
        how='inner'
    )

    # Create target structure with exact column names
    df_final = pd.DataFrame()
    df_final['Snippet-ID'] = df_merged['norm_id']
    df_final['Day'] = df_merged['date-folder']
    df_final['Location'] = df_merged['Location']
    df_final['Manual Count'] = df_merged['total-bats']
    df_final['Manual Checker'] = df_merged['name-of-checker']

    df_final['Koger forward +1'] = df_merged['new-method-count-going']
    df_final['Koger backwards -1'] = df_merged['new-method-count-coming']
    df_final['Koger Nett Count'] = df_merged['total-bats-new-method']
    df_final['Koger Error'] = df_final['Koger Nett Count'] - df_final['Manual Count']

    df_final['YOLO Forward +1'] = df_merged['Counting forward']
    df_final['YOLO backwards -1'] = df_merged['Counting backward']
    df_final['YOLO Nett Count'] = df_merged['Nett count']
    df_final['YOLO Error'] = df_final['YOLO Nett Count'] - df_final['Manual Count']

    df_final['Brightness'] = df_merged['Helligkeit']
    df_final['Brightness-Category'] = df_merged['Kategorie']
    df_final['Setting'] = ''
    df_final['Exclusion Status'] = ''
    df_final['Notes'] = df_merged['notes'].fillna('')

    # Save Master CSV
    out_path = Path(MASTER_OUTPUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(out_path, index=False)
    print(f"[2/3] Master CSV ({len(df_final)} rows) successfully saved to:\n      {out_path}")

    return df_final


# ---------------------------------------------------------------------------
# 2. Calculate Error and Correlation Metrics
# ---------------------------------------------------------------------------
def calculate_metrics_for_pair(manual: np.ndarray, auto: np.ndarray, method_name: str, category: str = "ALL") -> dict:
    """Calculates error and correlation metrics for a method compared to manual counts."""
    diff = auto - manual
    bias = float(np.mean(diff))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    valid_mask = manual > 0
    rel_error_pct = float(np.mean((diff[valid_mask] / manual[valid_mask]) * 100.0)) if np.any(valid_mask) else np.nan

    pearson_r, pearson_p = stats.pearsonr(manual, auto)
    spearman_rho, spearman_p = stats.spearmanr(manual, auto)

    return {
        "Category": category,
        "Method": method_name,
        "n_obs": len(manual),
        "bias": bias,
        "mae": mae,
        "rmse": rmse,
        "mean_relative_error_pct": rel_error_pct,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p)
    }


# ---------------------------------------------------------------------------
# 3. Compute Metrics Across All Categories
# ---------------------------------------------------------------------------
def calculate_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates metrics for YOLO vs. Manual and Koger vs. Manual (overall & by brightness category)."""
    results = []
    categories = ["ALL"] + sorted(df["Brightness-Category"].dropna().unique().tolist())

    for cat in categories:
        sub_df = df if cat == "ALL" else df[df["Brightness-Category"] == cat]
        if len(sub_df) < 2:
            continue

        manual = sub_df["Manual Count"].values
        yolo = sub_df["YOLO Nett Count"].values
        koger = sub_df["Koger Nett Count"].values

        results.append(calculate_metrics_for_pair(manual, yolo, "YOLO + T-ReX", cat))
        results.append(calculate_metrics_for_pair(manual, koger, "Koger (Method)", cat))

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 4. Print Results to Console
# ---------------------------------------------------------------------------
def print_comparison_results(df_res: pd.DataFrame):
    """Outputs the calculated metrics clearly to the console."""
    print("\n[3/3] CALCULATED METRICS (MANUAL vs. YOLO vs. KOGER)")
    print("=" * 85)

    for cat, group in df_res.groupby("Category", sort=False):
        print(f"\n--- Category: {cat} (n = {group['n_obs'].iloc[0]}) ---")
        for _, row in group.iterrows():
            print(f" Method: {row['Method']:<18}")
            print(f"   Bias (Mean Error)         : {row['bias']:>8.2f}")
            print(f"   MAE (Mean Abs. Error)     : {row['mae']:>8.2f}")
            print(f"   RMSE (Root Mean Sq Err)   : {row['rmse']:>8.2f}")
            print(f"   Mean Rel. Error (%)       : {row['mean_relative_error_pct']:>8.2f}%")
            print(f"   Pearson Correlation (r)   : {row['pearson_r']:>8.4f} (p = {row['pearson_p']:.2e})")
            print(f"   Spearman Correlation (rho): {row['spearman_rho']:>8.4f} (p = {row['spearman_p']:.2e})")
            print("-" * 55)
    print("=" * 85 + "\n")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main():
    # 1. Merge and save master data
    df_master = build_master_dataframe()

    # 2. Calculate metrics
    df_metrics = calculate_all_metrics(df_master)

    # 3. Output results
    print_comparison_results(df_metrics)


if __name__ == "__main__":
    main()
