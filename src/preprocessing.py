"""
preprocessing.py
----------------
Runs inside a SageMaker Processing Job.
Reads raw application_train.csv from /opt/ml/processing/input,
engineers features, splits into train/test, writes to
/opt/ml/processing/output/train and /opt/ml/processing/output/test.
"""

import argparse
import os
import logging

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Feature engineering helpers
# ------------------------------------------------------------------

def drop_high_null_columns(df: pd.DataFrame, threshold: float = 0.4) -> pd.DataFrame:
    """Drop columns where null fraction exceeds threshold"""
    null_frac = df.isnull().mean()
    cols_to_drop = null_frac[null_frac > threshold].index.tolist()
    log.info(f"Dropping {len(cols_to_drop)} high-null columns (>{threshold*100:.0f}% nulls)")
    return df.drop(columns=cols_to_drop)

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-specific financial features"""
    # Credit ratios
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["CREDIT_TERM"]              = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1)
    df["GOODS_CREDIT_RATIO"]       = df["AMT_GOODS_PRICE"] / (df["AMT_CREDIT"] + 1)

    # Age and employment (stored as negative days — convert to positive years)
    df["AGE_YEARS"]                = -df["DAYS_BIRTH"]    / 365
    df["YEARS_EMPLOYED"]           = df["DAYS_EMPLOYED"].apply(
        lambda x: -x / 365 if x < 0 else 0          # positive anomaly = unemployed
    )
    df["EMPLOYMENT_AGE_RATIO"]     = df["YEARS_EMPLOYED"] / (df["AGE_YEARS"] + 1)

    # Document submission rate (proxy for reliability)
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT")]
    if doc_cols:
        df["DOCS_SUBMITTED"] = df[doc_cols].sum(axis=1)

    log.info("Feature engineering complete - addded 8 derived features")
    return df

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all object columns."""
    cat_cols = df.select_dtypes(include="object").columns.to_list()
    log.info(f"Encoding {len(cat_cols)} categorical columns")
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill remaining nulls with column median"""
    missing = df.isnull().sum().sum()
    log.info(f"Imputing {missing} remaining null values with median")
    return df.fillna(df.median(numeric_only=True))

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    input_path = args.input_data
    output_base = "/opt/ml/processing/output"
    train_out = os.path.join(output_base, "train")
    test_out = os.path.join(output_base, "test")
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(test_out, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────
    csv_path = os.path.join(input_path, "application_train.csv")
    log.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    log.info(f"Raw shape: {df.shape}")

    # ── Class balance ─────────────────────────────────────────────
    class_counts = df["TARGET"].value_counts()
    scale_pos_weight = class_counts[0] / class_counts[1]
    log.info(f"Class balance — 0: {class_counts[0]}, 1: {class_counts[1]}")
    log.info(f"Recommended scale_pos_weight: {scale_pos_weight:.2f}")

    # ── Pipeline ──────────────────────────────────────────────────
    df = drop_high_null_columns(df, threshold=0.4)
    df = feature_engineering(df)
    df = encode_categoricals(df)
    df = impute_missing(df)

    # ── Split ─────────────────────────────────────────────────────
    X = df.drop("TARGET", axis=1)
    y = df["TARGET"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state
    )

    log.info(f"Train size: {X_train.shape[0]} rows | Test size: {X_test.shape[0]} rows")

    # ── Save ──────────────────────────────────────────────────────
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df  = pd.concat([X_test,  y_test],  axis=1)

    train_path = os.path.join(train_out, "train.csv")
    test_path  = os.path.join(test_out,  "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,   index=False)

    log.info(f"Train data saved → {train_path}")
    log.info(f"Test  data saved → {test_path}")
    log.info("Preprocessing complete.")


if __name__ == "__main__":
    main()


