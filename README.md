# customer-behavior-analysis
 End-to-end customer analytics project — EDA, data cleaning, and machine learning models to predict churn, purchases, and spending behavior. Built with Python, Pandas, and Scikit-learn.
# Customer Behavior Analysis — Phase 1 (T1–T5)
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12

Generic Pandas pipeline that handles Tasks T1–T5 from the project task
breakdown, on **any** customer CSV (no column names are hardcoded — the
script uses keyword heuristics to classify columns).

| Task | What it does | Output files |
|---|---|---|
| T1 | Dataset Structure Exploration | `T1_structure_summary.json`, `T1_sample_rows.csv` |
| T2 | Variable Type Identification (data dictionary) | `T2_data_dictionary.csv` |
| T3 | Missing Value Profiling | `T3_missing_value_report.csv`, `T3_missing_value_heatmap.png`, `T3_missing_value_barplot.png` |
| T4 | Duplicate & Inconsistency Detection | `T4_duplicate_rows.csv`, `T4_inconsistency_report.csv` |
| T5 | Data Cleaning & Preprocessing | `T5_cleaning_log.txt`, `cleaned_customer_data.csv` |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python customer_eda_phase1.py --input path/to/your_customer_data.csv --outdir outputs
```

- `--input` — path to your raw customer CSV file (required)
- `--outdir` — folder where all reports/cleaned data are written (default: `outputs`)

## How column classification works

Each column is heuristically tagged as **Demographic**, **Transaction
History**, **Customer Activity**, **Date/Time**, or **Identifier** based on
keyword matches in its name (e.g. `income`, `purchase`, `login`, `id`).
Semantic data type (Numeric / Categorical / Boolean / Datetime / Text) is
inferred from the values themselves, independent of column naming.

## Cleaning logic applied in T5

- Strips whitespace and standardizes casing on text/categorical columns
- Drops exact duplicate rows
- Fills missing numeric values with the column median
- Fills missing categorical values with the column mode
- Clips unexpected negative values in transaction/activity numeric columns to 0
- Leaves missing datetime values as `NaT`, flagged in the cleaning log for manual review

## Next steps

Phase 2 (feature engineering + predictive modeling) will build on
`cleaned_customer_data.csv` produced here.
"""
Customer Behavior Analysis - Phase 1
Tasks T1-T5: Structure Exploration, Variable Typing, Missing Value Profiling,
Duplicate/Inconsistency Detection, and Data Cleaning & Preprocessing.

Usage:
    python customer_eda_phase1.py --input path/to/customer_data.csv --outdir outputs

Designed to be generic -- works on any customer dataset (demographic, transaction,
or activity columns), using heuristics to classify variables and flag issues.
No dataset-specific column names are hardcoded, so this runs as-is once you
point it at your CSV.
"""

import argparse
import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEMOGRAPHIC_HINTS = [
    "age", "gender", "sex", "income", "occupation", "education",
    "marital", "city", "state", "country", "region", "zipcode", "zip",
    "dob", "birth", "nationality",
]
TRANSACTION_HINTS = [
    "purchase", "order", "amount", "price", "revenue", "spend", "spent",
    "transaction", "invoice", "payment", "quantity", "qty", "discount",
    "total", "cost", "sales",
]
ACTIVITY_HINTS = [
    "login", "visit", "session", "click", "engagement", "activity",
    "last_seen", "last_active", "frequency", "recency", "usage",
    "app_open", "page_view", "churn", "subscription", "tenure",
]
ID_HINTS = ["id", "uuid", "customer_no", "index"]
DATE_HINTS = ["date", "time", "timestamp", "created", "updated", "joined"]


def classify_column(col_name: str) -> str:
    """Heuristically classify a column into a business category based on its name."""
    name = col_name.lower()
    if any(h in name for h in ID_HINTS):
        return "Identifier"
    if any(h in name for h in DATE_HINTS):
        return "Date/Time"
    if any(h in name for h in DEMOGRAPHIC_HINTS):
        return "Demographic"
    if any(h in name for h in TRANSACTION_HINTS):
        return "Transaction History"
    if any(h in name for h in ACTIVITY_HINTS):
        return "Customer Activity"
    return "Uncategorized"


def infer_semantic_dtype(series: pd.Series) -> str:
    """Best-effort semantic type: Numeric, Categorical, Boolean, Datetime, Text."""
    if pd.api.types.is_bool_dtype(series):
        return "Boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "Numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "Datetime"

    # Try datetime parse on object columns with date-like name or values
    non_null = series.dropna()
    if len(non_null) > 0:
        sample = non_null.astype(str).head(20)
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed.notna().mean() > 0.8:
            return "Datetime"

    nunique_ratio = series.nunique(dropna=True) / max(len(series), 1)
    if nunique_ratio < 0.05 or series.nunique(dropna=True) < 20:
        return "Categorical"
    return "Text"


# ---------------------------------------------------------------------------
# T1: Dataset Structure Exploration
# ---------------------------------------------------------------------------

def task1_structure_exploration(df: pd.DataFrame, outdir: str) -> dict:
    print("\n[T1] Dataset Structure Exploration")
    summary = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
    }
    print(f"  Rows: {summary['n_rows']:,} | Columns: {summary['n_columns']}")
    print(f"  Memory footprint: {summary['memory_usage_mb']} MB")

    with open(os.path.join(outdir, "T1_structure_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    df.head(10).to_csv(os.path.join(outdir, "T1_sample_rows.csv"), index=False)
    return summary


# ---------------------------------------------------------------------------
# T2: Variable Type Identification
# ---------------------------------------------------------------------------

def task2_variable_typing(df: pd.DataFrame, outdir: str) -> pd.DataFrame:
    print("\n[T2] Variable Type Identification")
    records = []
    for col in df.columns:
        records.append({
            "column": col,
            "pandas_dtype": str(df[col].dtype),
            "semantic_type": infer_semantic_dtype(df[col]),
            "business_category": classify_column(col),
            "n_unique": int(df[col].nunique(dropna=True)),
            "sample_values": ", ".join(
                map(str, df[col].dropna().unique()[:3])
            ) if df[col].notna().any() else "",
        })
    data_dictionary = pd.DataFrame(records)
    print(data_dictionary.to_string(index=False))

    data_dictionary.to_csv(os.path.join(outdir, "T2_data_dictionary.csv"), index=False)
    return data_dictionary


# ---------------------------------------------------------------------------
# T3: Missing Value Profiling
# ---------------------------------------------------------------------------

def task3_missing_value_profiling(df: pd.DataFrame, outdir: str) -> pd.DataFrame:
    print("\n[T3] Missing Value Profiling")
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    report = pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct": missing_pct,
    }).sort_values("missing_count", ascending=False)
    report.index.name = "column"
    print(report[report["missing_count"] > 0].to_string())

    report.to_csv(os.path.join(outdir, "T3_missing_value_report.csv"))

    # Heatmap of missingness
    plt.figure(figsize=(min(1 + 0.5 * df.shape[1], 20), 6))
    sns.heatmap(df.isna(), cbar=False, cmap="viridis", yticklabels=False)
    plt.title("Missing Value Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "T3_missing_value_heatmap.png"), dpi=150)
    plt.close()

    # Bar chart of missing percentage
    nonzero = report[report["missing_count"] > 0]
    if not nonzero.empty:
        plt.figure(figsize=(8, max(3, 0.35 * len(nonzero))))
        sns.barplot(x=nonzero["missing_pct"], y=nonzero.index, color="steelblue")
        plt.xlabel("Missing (%)")
        plt.title("Missing Value Percentage by Column")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "T3_missing_value_barplot.png"), dpi=150)
        plt.close()

    return report


# ---------------------------------------------------------------------------
# T4: Duplicate & Inconsistency Detection
# ---------------------------------------------------------------------------

def task4_duplicate_inconsistency_detection(df: pd.DataFrame, outdir: str) -> dict:
    print("\n[T4] Duplicate & Inconsistency Detection")

    duplicate_rows = df[df.duplicated(keep=False)]
    n_duplicates = int(df.duplicated().sum())
    print(f"  Exact duplicate rows: {n_duplicates}")

    # ID-column duplicate check (if an identifier-like column exists)
    id_cols = [c for c in df.columns if classify_column(c) == "Identifier"]
    id_duplicate_summary = {}
    for col in id_cols:
        dup_ids = int(df[col].duplicated().sum())
        id_duplicate_summary[col] = dup_ids
        print(f"  Duplicate values in identifier column '{col}': {dup_ids}")

    # Inconsistency checks
    inconsistencies = []

    # 1. Whitespace / casing inconsistencies in text/categorical columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        stripped_differs = (df[col].dropna().astype(str) != df[col].dropna().astype(str).str.strip()).sum()
        case_variants = df[col].dropna().astype(str).str.lower().nunique()
        raw_variants = df[col].dropna().astype(str).nunique()
        if stripped_differs > 0:
            inconsistencies.append({
                "column": col, "issue": "Leading/trailing whitespace",
                "count": int(stripped_differs),
            })
        if raw_variants > case_variants:
            inconsistencies.append({
                "column": col, "issue": "Inconsistent casing (e.g. 'Male' vs 'male')",
                "count": int(raw_variants - case_variants),
            })

    # 2. Negative values in columns that look like counts/amounts (shouldn't be negative)
    for col in df.select_dtypes(include=np.number).columns:
        if classify_column(col) in ("Transaction History", "Customer Activity"):
            n_negative = int((df[col] < 0).sum())
            if n_negative > 0:
                inconsistencies.append({
                    "column": col, "issue": "Unexpected negative values",
                    "count": n_negative,
                })

    # 3. Out-of-range ages, if an age-like column exists
    for col in df.columns:
        if "age" in col.lower() and pd.api.types.is_numeric_dtype(df[col]):
            bad_age = int(((df[col] < 0) | (df[col] > 120)).sum())
            if bad_age > 0:
                inconsistencies.append({
                    "column": col, "issue": "Implausible age (<0 or >120)",
                    "count": bad_age,
                })

    inconsistency_df = pd.DataFrame(inconsistencies)
    if not inconsistency_df.empty:
        print(inconsistency_df.to_string(index=False))
    else:
        print("  No obvious inconsistencies detected via heuristics.")

    duplicate_rows.to_csv(os.path.join(outdir, "T4_duplicate_rows.csv"), index=False)
    inconsistency_df.to_csv(os.path.join(outdir, "T4_inconsistency_report.csv"), index=False)

    return {
        "n_exact_duplicate_rows": n_duplicates,
        "id_column_duplicates": id_duplicate_summary,
        "n_inconsistency_flags": int(len(inconsistency_df)),
    }


# ---------------------------------------------------------------------------
# T5: Data Cleaning & Preprocessing
# ---------------------------------------------------------------------------

def task5_clean_and_preprocess(df: pd.DataFrame, outdir: str) -> pd.DataFrame:
    print("\n[T5] Data Cleaning & Preprocessing")
    clean_df = df.copy()
    log = []

    # 1. Strip whitespace and normalize casing on text columns (title case for readability)
    text_cols = clean_df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        before = clean_df[col].copy()
        clean_df[col] = clean_df[col].astype(str).str.strip()
        clean_df.loc[df[col].isna(), col] = np.nan  # preserve true NaNs
        changed = int((before.astype(str) != clean_df[col].astype(str)).sum())
        if changed:
            log.append(f"Stripped whitespace in '{col}' ({changed} values)")

    # 2. Remove exact duplicate rows
    n_before = len(clean_df)
    clean_df = clean_df.drop_duplicates()
    n_removed = n_before - len(clean_df)
    if n_removed:
        log.append(f"Removed {n_removed} exact duplicate rows")

    # 3. Handle missing values
    for col in clean_df.columns:
        n_missing = clean_df[col].isna().sum()
        if n_missing == 0:
            continue
        semantic = infer_semantic_dtype(clean_df[col])
        if semantic == "Numeric":
            median_val = clean_df[col].median()
            clean_df[col] = clean_df[col].fillna(median_val)
            log.append(f"Filled {n_missing} missing values in '{col}' with median ({median_val})")
        elif semantic in ("Categorical", "Boolean", "Text"):
            mode_val = clean_df[col].mode(dropna=True)
            fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            clean_df[col] = clean_df[col].fillna(fill_val)
            log.append(f"Filled {n_missing} missing values in '{col}' with mode ('{fill_val}')")
        elif semantic == "Datetime":
            clean_df[col] = pd.to_datetime(clean_df[col], errors="coerce")
            log.append(f"Left {n_missing} missing datetime values in '{col}' as NaT (flag for review)")

    # 4. Fix negative values in transaction/activity numeric columns -> clip at 0
    for col in clean_df.select_dtypes(include=np.number).columns:
        if classify_column(col) in ("Transaction History", "Customer Activity"):
            n_negative = (clean_df[col] < 0).sum()
            if n_negative > 0:
                clean_df[col] = clean_df[col].clip(lower=0)
                log.append(f"Clipped {n_negative} negative values in '{col}' to 0")

    # 5. Standardize categorical text casing (e.g., "male"/"Male"/"MALE" -> "Male")
    for col in clean_df.select_dtypes(include=["object", "string"]).columns:
        if infer_semantic_dtype(clean_df[col]) == "Categorical":
            clean_df[col] = clean_df[col].astype(str).str.strip().str.title()

    print("  Cleaning steps applied:")
    for entry in log:
        print(f"   - {entry}")

    with open(os.path.join(outdir, "T5_cleaning_log.txt"), "w") as f:
        f.write("\n".join(log) if log else "No cleaning actions were necessary.")

    cleaned_path = os.path.join(outdir, "cleaned_customer_data.csv")
    clean_df.to_csv(cleaned_path, index=False)
    print(f"  Cleaned dataset saved to: {cleaned_path}")

    return clean_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1 (T1-T5) customer data pipeline")
    parser.add_argument("--input", required=True, help="Path to the raw customer CSV file")
    parser.add_argument("--outdir", default="outputs", help="Directory to write reports/cleaned data")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)

    task1_structure_exploration(df, args.outdir)
    task2_variable_typing(df, args.outdir)
    task3_missing_value_profiling(df, args.outdir)
    task4_duplicate_inconsistency_detection(df, args.outdir)
    clean_df = task5_clean_and_preprocess(df, args.outdir)

    print(f"\nAll Phase 1 (T1-T5) deliverables written to: {os.path.abspath(args.outdir)}")
    print(f"Final cleaned dataset shape: {clean_df.shape}")


if __name__ == "__main__":
    main()
