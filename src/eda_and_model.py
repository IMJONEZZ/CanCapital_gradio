#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End‑to‑end EDA → cleaning → feature engineering → modelling →
business insights (Permutation importance + PDP/ICE) for the loan‑collection CSV.

No SHAP dependency – works on Python 3.11.
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import List, Optional

# categorical encoding ---------------------------------------------------------
import category_encoders as ce  # TargetEncoder for high‑cardinality cols  # TargetEncoder for high‑cardinality cols
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_log_error,
    r2_score,
)

# --------------------------------------------------------------------------- #
# scikit‑learn ---------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
sns.set(style="whitegrid", font_scale=1.1)


# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(
        description="End‑to‑end EDA → cleaning → feature engineering → modelling → business insights (Permutation importance + PDP/ICE) for the loan‑collection CSV.\n\nNo SHAP dependency – works on Python 3.11."
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the raw CSV file.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Target column name. If not specified, will auto-detect numeric columns.",
    )
    parser.add_argument(
        "--save-model",
        default="trained_models",
        help="Folder where trained models are saved.",
    )
    parser.add_argument(
        "--output_dir",
        default="eda_output",
        help="Folder where all artefacts will be saved.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
def load_data(csv_path: Path) -> pd.DataFrame:
    print(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Shape: {df.shape}")
    return df


# -------------------------- AUTO-DETECTION --------------------------------- #
def auto_detect_column_types(df: pd.DataFrame) -> dict:
    """Automatically detect column types for any financial dataset."""
    detection = {
        "date_cols": [],
        "money_cols": [],
        "percent_cols": [],
        "categorical_threshold": 50,
    }

    for col in df.columns:
        col_lower = col.lower()

        # Date detection
        if any(keyword in col_lower for keyword in ["date", "time", "_dt"]):
            detection["date_cols"].append(col)
        # Money/currency detection
        elif any(
            keyword in col_lower
            for keyword in ["amount", "price", "value", "cap", "funding"]
        ):
            detection["money_cols"].append(col)
        # Percent detection
        elif "%" in str(df[col].dtype) or "percent" in col_lower:
            detection["percent_cols"].append(col)

    return detection


def detect_target_column(df: pd.DataFrame) -> None:
    """Automatically detect the most likely target column for prediction."""

    # Priority order for financial datasets
    priority_patterns = [
        "collection_case_days",  # Original dataset target
        "price_return",
        "return",
        "daily_return",
        "target_price",
        "future_price",
        "revenue",
        "profit",
        "loss",
        "volume",
        "trading_volume",
        "volatility",
        "risk_metric",
    ]

    # First check for exact matches
    available_cols = [col.lower() for col in df.columns]
    for pattern in priority_patterns:
        if pattern in available_cols:
            # Return the actual column name with correct casing
            for col in df.columns:
                if col.lower() == pattern:
                    return col

    # If no match found, look for numeric columns that could be targets
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Exclude obvious feature columns
    exclude_patterns = ["id", "index", "rank", "score"]
    potential_targets = []

    for col in numeric_cols:
        col_lower = col.lower()
        if not any(pattern in col_lower for pattern in exclude_patterns):
            # Check distribution - good targets usually have reasonable variance
            if df[col].var() > 0:
                potential_targets.append(col)

    # Return the first reasonable target if nothing else found
    return potential_targets[0] if potential_targets else None


# -------------------------- FLEXIBLE CLEANING ------------------------------- #
def flexible_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply flexible cleaning based on detected column types."""

    # Auto-detect column types
    detection = auto_detect_column_types(df)

    # Clean date columns
    if detection["date_cols"]:
        df = clean_dates(df, detection["date_cols"])

    # Clean money/currency columns
    if detection["money_cols"]:
        df = clean_money(df, detection["money_cols"])

    # Clean percent columns
    for col in df.columns:
        if "percent" in str(df[col].dtype).lower() or any(
            x in col.lower() for x in ["%", "percent"]
        ):
            df = clean_percent(df, col)

    # Handle categorical columns more carefully
    cat_cols = []
    for col in df.select_dtypes(include=["object"]).columns:
        # Skip if column appears to be text/description
        col_lower = col.lower()
        if any(
            skip_word in col_lower for skip_word in ["description", "notes", "text"]
        ):
            continue

        unique_count = df[col].nunique()

        # More sophisticated categorical detection
        if unique_count <= detection["categorical_threshold"] and unique_count > 1:
            cat_cols.append(col)

    if cat_cols:
        df = fill_categorical_na(df, cat_cols)

    return df


# -------------------------- CLEANING --------------------------------------- #
def clean_dates(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
    return df


def clean_money(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            continue
        cleaned = (
            df[c]
            .astype(str)
            .str.replace(r"[$,]", "", regex=True)
            .str.strip()
            .replace({"": np.nan})
        )
        cleaned = cleaned.replace({"-": "0", "$ -": "0"}).astype(float)
        df[c] = cleaned
    return df


def clean_percent(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = (
            df[col].astype(str).str.rstrip("%").replace({"": np.nan}).astype(float)
        )
    return df


def fill_categorical_na(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("object").fillna("Unknown")
    return df


# -------------------------- FEATURE ENGINEERING --------------------------- #
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # borrower age at funding
    if "pg_date_of_birth" in df.columns and "funded_date" in df.columns:
        df["borrower_age"] = (
            df["funded_date"] - pd.to_datetime(df["pg_date_of_birth"], errors="coerce")
        ).dt.days / 365.25
    else:
        df["borrower_age"] = np.nan

    # time between funding and case open
    if "funded_date" in df.columns and "collection_case_open_date" in df.columns:
        df["days_funded_to_open"] = (
            df["collection_case_open_date"] - df["funded_date"]
        ).dt.days
    else:
        df["days_funded_to_open"] = np.nan

    # time between last payment and close
    if "last_payment_date" in df.columns and "collection_case_close_date" in df.columns:
        df["days_lastpay_to_close"] = (
            df["collection_case_close_date"] - df["last_payment_date"]
        ).dt.days
    else:
        df["days_lastpay_to_close"] = np.nan

    # flags for missing important dates
    for col in ["collection_case_close_date", "demand_letter_sent_date"]:
        if col in df.columns:
            flag = f"has_{col}"
            df[flag] = df[col].notna().astype(int)

    # log‑transform target (kept for modelling)
    if "collection_case_days" in df.columns:
        df["log_collection_case_days"] = np.log1p(df["collection_case_days"])

    return df


# -------------------------- SPLIT ---------------------------------------- #
def chronological_split(df: pd.DataFrame, target: str):
    if "funded_date" not in df.columns:
        raise ValueError("`funded_date` is required for a chronological split.")
    df = df.sort_values("funded_date").reset_index(drop=True)

    train_cutoff = int(len(df) * 0.70)
    train_df, test_df = df.iloc[:train_cutoff], df.iloc[train_cutoff:]

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    return X_train, X_test, y_train, y_test


# -------------------------- PREPROCESSOR (with imputer) ------------------- #
def build_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """
    Numeric pipeline  : median imputer → scaler
    Categorical pipeline: most_frequent imputer → TargetEncoder
    """
    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("target_enc", ce.TargetEncoder(smoothing=1.0)),
        ]
    )

    preproc = ColumnTransformer(
        [
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )
    return preproc


# -------------------------- MODEL TRAINING ------------------------------- #
def train_and_evaluate(
    X_train,
    y_train,
    X_test,
    y_test,
    output_dir: Path,
    models_save_dir: str = "trained_models",
    target_column: Optional[str] = None,
):
    # Validate that target variables are numeric
    if not hasattr(y_train, "dtype") or y_train.dtype == "object":
        raise ValueError(
            "Target variable (y_train) must be numeric. Please ensure target column contains only numeric values."
        )
    if not hasattr(y_test, "dtype") or y_test.dtype == "object":
        raise ValueError(
            "Target variable (y_test) must be numeric. Please ensure target column contains only numeric values."
        )

    # Ensure targets are numpy arrays for consistent handling
    y_train = pd.Series(y_train).astype(float)
    y_test = pd.Series(y_test).astype(float)

    numeric = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Remove accidental target copies
    for col in ["collection_case_days", "log_collection_case_days"]:
        if col in numeric:
            numeric.remove(col)

    preproc = build_preprocessor(numeric, categorical)

    # ----- Linear Regression (baseline) -----
    lin_pipe = Pipeline([("preprocess", preproc), ("linreg", LinearRegression())])
    lin_pipe.fit(X_train, np.log1p(y_train))
    pred_lin = np.expm1(lin_pipe.predict(X_test))

    mae_lin = mean_absolute_error(y_test, pred_lin)
    rmsle_lin = np.sqrt(mean_squared_log_error(y_test, pred_lin))
    r2_lin = r2_score(y_test, pred_lin)

    # ----- Gradient Boosting Regressor (main model) -----
    gbr_pipe = Pipeline(
        [
            ("preprocess", preproc),
            (
                "gbr",
                GradientBoostingRegressor(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42,
                ),
            ),
        ]
    )
    gbr_pipe.fit(X_train, np.log1p(y_train))
    pred_gbr = np.expm1(gbr_pipe.predict(X_test))

    mae_gbr = mean_absolute_error(y_test, pred_gbr)
    rmsle_gbr = np.sqrt(mean_squared_log_error(y_test, pred_gbr))
    r2_gbr = r2_score(y_test, pred_gbr)

    # Save predictions
    pd.DataFrame({"actual": y_test, "predicted": pred_lin}).to_csv(
        output_dir / "linearreg_predictions.csv", index=False
    )
    pd.DataFrame({"actual": y_test, "predicted": pred_gbr}).to_csv(
        output_dir / "gbr_predictions.csv", index=False
    )

    print("\n=== Model performance ===")
    print(
        f"LinearReg   → MAE: {mae_lin:,.2f} | RMSLE: {rmsle_lin:.4f} | R²: {r2_lin:.3f}"
    )
    print(
        f"GradientBoost→ MAE: {mae_gbr:,.2f} | RMSLE: {rmsle_gbr:.4f} | R²: {r2_gbr:.3f}"
    )

    # Save models to dedicated directory
    save_models(
        models_save_dir,
        gbr_pipe,
        mae_gbr,
        args.target if "args" in locals() else "target",
        lin_pipe,
        mae_lin,
    )

    # Save models to dedicated directory
    save_models(
        models_save_dir,
        gbr_pipe,
        mae_gbr,
        target_column or "auto_detected",
        lin_pipe,
        mae_lin,
    )

    best_pipe = gbr_pipe if mae_gbr < mae_lin else lin_pipe
    return best_pipe, preproc


def save_models(
    models_save_dir: str,
    gbr_model,
    mae_gbr: float,
    target_column: str,
    linreg_model=None,
    mae_lin: Optional[float] = None,
):
    """Save trained models with metadata for future use."""

    import json
    from datetime import datetime

    # Create models directory if it doesn't exist
    save_path = Path(models_save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Create a timestamp-based model directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = save_path / f"model_{timestamp}"
    model_dir.mkdir(exist_ok=True)

    # Determine best model based on MAE
    if mae_gbr < mae_lin:
        best_model = gbr_model
        model_type = "GradientBoostingRegressor"
        best_mae = mae_gbr
    else:
        best_model = linreg_model
        model_type = "LinearRegression"
        best_mae = mae_lin

    # Save the trained models
    with open(model_dir / "best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open(model_dir / "gbr_model.pkl", "wb") as f:
        pickle.dump(gbr_model, f)

    with open(model_dir / "linearreg_model.pkl", "wb") as f:
        pickle.dump(linreg_model, f)

    # Save metadata
    metadata = {
        "target_column": target_column,
        "model_type": model_type,
        "best_mae": float(best_mae),
        "gbr_mae": float(mae_gbr),
        "linreg_mae": float(mae_lin),
        "trained_at": timestamp,
        "description": f"CANCapital loan collection prediction model for '{target_column}'",
    }

    with open(model_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"📁 Models saved to: {model_dir}")


# -------------------------- INTERPRETABILITY ---------------------------- #
def _original_feature_names(
    preproc: ColumnTransformer, X_sample: pd.DataFrame
) -> List[str]:
    """
    Return a flat list with the *raw* column names that actually make it
    through preprocessing. This handles cases where features get dropped.
    """
    # Get the transformers that actually exist
    numeric_cols = []
    categorical_cols = []

    if len(preproc.transformers_) > 0:
        # Numeric transformer
        if preproc.transformers_[0][0] == "num":
            numeric_cols = list(preproc.transformers_[0][2])

    if len(preproc.transformers_) > 1:
        # Categorical transformer
        if preproc.transformers_[1][0] == "cat":
            categorical_cols = list(preproc.transformers_[1][2])

    # Test with a small sample to see which features actually survive
    try:
        X_transformed = preproc.transform(X_sample.head(10))

        # If transformation worked, return the processed feature names
        if hasattr(X_transformed, "shape"):
            n_features = X_transformed.shape[1]

            # Get total expected features from transformers
            expected_numeric = len(numeric_cols)
            expected_categorical = len(categorical_cols)

            # If the numbers don't match, some features were dropped
            if n_features != (expected_numeric + expected_categorical):
                # Return only the features that actually processed
                if hasattr(X_transformed, "columns"):
                    return list(X_transformed.columns)
                else:
                    # Fallback: limit to actual processed features
                    all_expected = list(numeric_cols) + list(categorical_cols)
                    return all_expected[:n_features]

            # Numbers match, return original feature names
            return list(numeric_cols) + list(categorical_cols)

    except Exception:
        # If transformation fails, fall back to original logic
        return list(numeric_cols) + list(categorical_cols)


def plot_permutation_importance(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
):
    """
    Same as before, but the feature‑name list that we write to CSV and
    use for plotting now comes from ``_original_feature_names`` so it
    matches the column names in the original dataframe.
    """
    preproc = model.named_steps["preprocess"]
    X_enc = preproc.transform(X_test)

    # underlying estimator (gbr or linreg)
    est_name = [n for n in model.named_steps.keys() if n != "preprocess"][0]
    estimator = model.named_steps[est_name]

    result = permutation_importance(
        estimator,
        X_enc,
        np.log1p(y_test),
        n_repeats=10,
        random_state=42,
        scoring="neg_mean_absolute_error",
    )

    # Get **original** feature names (no "num__"/"cat__")
    feat_names = _original_feature_names(preproc, X_test)

    importance_df = pd.DataFrame(
        {
            "feature": feat_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    # ----- Plot top 20 -------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="importance_mean",
        y="feature",
        data=importance_df.head(20),
        palette="viridis",
    )
    plt.title("Permutation Importance – Top 20 Features")
    plt.xlabel("Mean increase in MAE after shuffling")
    plt.tight_layout()
    plt.savefig(output_dir / "perm_importance.png")
    plt.close()

    # ----- Save full table ---------------------------------------------
    importance_df.to_csv(output_dir / "permutation_importance_full.csv", index=False)


def plot_partial_dependence(
    model: Pipeline,
    X_test: pd.DataFrame,
    output_dir: Path,
    features: Optional[List[str]] = None,
):
    """
    Generates PDP + ICE plots for the most important variables.
    If ``features`` is ``None`` we automatically pick the top‑5
    according to the permutation‑importance table that was saved by the
    previous function.  The crucial fix is that we **pre‑process** X_test
    before handing it to PartialDependenceDisplay, so no string columns
    are left for the quantile routine.
    """
    preproc = model.named_steps["preprocess"]
    estimator_name = [n for n in model.named_steps.keys() if n != "preprocess"][0]
    estimator = model.named_steps[estimator_name]

    # ----- Determine which (raw) features to plot -----------------------
    if features is None:
        perm_path = output_dir / "permutation_importance_full.csv"
        if not perm_path.is_file():
            raise FileNotFoundError(
                f"Permutation‑importance file not found at {perm_path}. "
                "Run `plot_permutation_importance` first or pass a list of feature names."
            )
        perm_df = pd.read_csv(perm_path)
        features_raw = perm_df["feature"].head(5).tolist()  # raw column names
    else:
        features_raw = features

    # ----- Map raw column names to the **encoded** column indices -------
    all_raw_names = _original_feature_names(
        preproc, X_test
    )  # order used by transformer
    idx_map = {name: pos for pos, name in enumerate(all_raw_names)}
    feature_indices = [idx_map[name] for name in features_raw]

    # ----- Transform X_test once (numeric + encoded categorical) -------
    X_enc = preproc.transform(X_test)

    # ----- Partial dependence on the *encoded* matrix --------------------
    # We give the estimator (not the whole pipeline) because X is already
    # transformed.  Feature names for the plot are taken from the raw list.
    display = PartialDependenceDisplay.from_estimator(
        estimator,
        X_enc,
        feature_indices,
        kind="both",  # average PDP + ICE curves
        subsample=200,
        n_jobs=-1,
        grid_resolution=50,
        random_state=42,
    )
    # Relabel the y‑axis with the original column names for readability
    display.axes_.ravel()[0].set_ylabel("Partial dependence")
    for ax, raw_name in zip(display.axes_.ravel(), features_raw):
        ax.set_title(raw_name)

    plt.tight_layout()
    plt.savefig(output_dir / "pdp_ice.png")
    plt.close()


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    csv_path = Path(args.csv_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------- Load & clean ----------------------------------- #
    df_raw = load_data(csv_path)

    print(f"📊 Dataset shape: {df_raw.shape}")
    print("🔍 Auto-detecting column types...")

    # Use flexible cleaning
    df = flexible_clean_data(df_raw)

    print("🔧 Engineering features...")

    # -------------------- Target detection and validation ------------------ #
    if args.target is None:
        print("🎯 Auto-detecting target column...")
        detected_target = detect_target_column(df)
        if not detected_target:
            raise ValueError(
                "Could not auto-detect target column. Please specify with --target flag."
            )
        args.target = detected_target
        print(f"✅ Using target column: '{args.target}'")

    # Validate that the target exists and is numeric
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in dataset.")

    # Ensure target is numeric for modeling
    if df[args.target].dtype == "object":
        # Try to convert to numeric, handling potential string values
        try:
            df[args.target] = pd.to_numeric(df[args.target], errors="coerce")
            print(f"⚠️ Converted target column '{args.target}' to numeric")
        except Exception as e:
            raise ValueError(
                f"Target column '{args.target}' cannot be converted to numeric: {e}"
            )

    # Remove rows where target is still missing (NaN values)
    initial_rows = len(df)
    df = df[~df[args.target].isna()].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(
            "No valid rows remaining after removing missing target values."
        )

    if len(df) < initial_rows * 0.5:  # If more than half the data was removed
        print(
            f"⚠️ Warning: {initial_rows - len(df)} rows ({((initial_rows - len(df)) / initial_rows) * 100:.1f}%) removed due to missing target values"
        )

    # -------------------- Feature engineering --------------------------- #
    df = engineer_features(df)

    # Drop rows where target is still missing (after feature engineering)
    df = df[~df[args.target].isna()].reset_index(drop=True)

    # -------------------- Train / test split ---------------------------- #
    X_train, X_test, y_train, y_test = chronological_split(df, args.target)

    # -------------------- Model training -------------------------------- #
    best_model, preproc = train_and_evaluate(
        X_train, y_train, X_test, y_test, out_dir, args.save_model, args.target
    )

    # -------------------- Interpretation -------------------------------- #
    plot_permutation_importance(best_model, X_test, y_test, out_dir)
    plot_partial_dependence(best_model, X_test, out_dir)  # auto‑pick top 5
    print("\nAll artefacts written to:", out_dir.resolve())

    print(f"📁 Models saved to: {args.save_model}")


if __name__ == "__main__":
    main()
