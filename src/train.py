"""
train.py
--------
Runs inside a SageMaker Training Job .
Reads train/test CSVs from SM_CHANNEL_TRAIN / SM_CHANNEL_VALIDATION,
trains an XGBoost model, saves artifact to SM_MODEL_DIR.
"""

import argparse
import os
import logging
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)




logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Evaluation function
# ------------------------------------------------------------------

def evaluate(model: xgb.Booster, dmatrix: xgb.DMatrix, y_true: pd.Series, split: str):
    preds = model.predict(dmatrix)
    auc = roc_auc_score(y_true, preds)
    ap = average_precision_score(y_true, preds)
    preds_binary = (preds >= 0.5).astype(int)

    log.info(f"\n{'='*40}")
    log.info(f"{split.upper()} METRICS")
    log.info(f"  AUC-ROC            : {auc:.4f}")
    log.info(f"  Avg Precision (AP) : {ap:.4f}")
    log.info(f"\nClassification Report:\n{classification_report(y_true, preds_binary)}")
    log.info(f"Confusion Matrix:\n{confusion_matrix(y_true, preds_binary)}")
    log.info(f"{'='*40}\n")

    return {"auc": auc, "avg_precision": ap}

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    # Hyperparameters — passed by SageMaker or HPO tuner
    parser.add_argument("--max_depth",           type=int,   default=6)
    parser.add_argument("--eta",                 type=float, default=0.1)
    parser.add_argument("--min_child_weight",    type=int,   default=5)
    parser.add_argument("--subsample",           type=float, default=0.8)
    parser.add_argument("--colsample_bytree",    type=float, default=0.8)
    parser.add_argument("--num_round",           type=int,   default=300)
    parser.add_argument("--early_stopping_rounds", type=int, default=20)
    parser.add_argument("--scale_pos_weight",    type=float, default=11.0)

    # SageMaker injects these automatically via environment variables
    parser.add_argument("--train",      type=str, default=os.environ.get("SM_CHANNEL_TRAIN",      ""))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", ""))
    parser.add_argument("--model_dir",  type=str, default=os.environ.get("SM_MODEL_DIR",          "/opt/ml/model"))
    parser.add_argument("--output_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR",    "/opt/ml/output"))

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    log.info(f"Hyperparameters: {vars(args)}")

    # ── Load data ─────────────────────────────────────────────────
    train_csv = os.path.join(args.train, "train.csv")
    val_csv = os.path.join(args.validation, "test.csv")

    log.info(f"Loading train data from: {train_csv}")
    train_df = pd.read_csv(train_csv)

    log.info(f"Loading validation data from: {val_csv}")
    val_df = pd.read_csv(val_csv)

    X_train = train_df.drop("TARGET", axis=1)
    y_train = train_df["TARGET"]
    X_val   = val_df.drop("TARGET", axis=1)
    y_val   = val_df["TARGET"]

    log.info(f"Train: {X_train.shape} | Val: {X_val.shape}")
    log.info(f"Train positive rate: {y_train.mean():.3f}")

    # ── Build DMatrix ─────────────────────────────────────────────
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X_train.columns))
    dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=list(X_val.columns))

    # ── Train ─────────────────────────────────────────────────────
    params = {
        "objective":        "binary:logistic",
        "eval_metric":      ["auc", "aucpr"],
        "max_depth":        args.max_depth,
        "eta":              args.eta,
        "min_child_weight": args.min_child_weight,
        "subsample":        args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "scale_pos_weight": args.scale_pos_weight,
        "seed":             42,
        "verbosity":        1
    }

    log.info("Starting XGBoost training...")
    evals_result = {}

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_round,
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=args.early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=50
    )

    log.info(f"Best iteration : {model.best_iteration}")
    log.info(f"Best val AUC   : {model.best_score:.4f}")

    # ── Evaluate ──────────────────────────────────────────────────
    train_metrics = evaluate(model, dtrain, y_train, "train")
    val_metrics   = evaluate(model, dval,   y_val,   "validation")

    # ── Feature importance ────────────────────────────────────────
    importance = model.get_score(importance_type="gain")
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
    log.info("Top 20 features by gain:")
    for feat, score in top_features:
        log.info(f"  {feat:<45} {score:.2f}")

    # ── Save model ────────────────────────────────────────────────
    model_path = os.path.join(args.model_dir, "model.xgb")
    model.save_model(model_path)
    log.info(f"Model saved → {model_path}")

    # Save metrics JSON for SageMaker Experiments
    metrics = {
        "train_auc":            train_metrics["auc"],
        "validation_auc":       val_metrics["auc"],
        "train_avg_precision":  train_metrics["avg_precision"],
        "val_avg_precision":    val_metrics["avg_precision"],
        "best_iteration":       model.best_iteration,
        "best_val_auc":         model.best_score
    }

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Metrics saved → {metrics_path}")

    log.info("Training complete.")


if __name__ == "__main__":
    # Print everything SageMaker injected — helps debug
    print("=== ENV VARS ===")
    for k, v in os.environ.items():
        if "SM_" in k or "SAGEMAKER" in k:
            print(f"  {k} = {v}")

    print("\n=== SYS.ARGV ===")
    import sys
    print(sys.argv)
    main()
