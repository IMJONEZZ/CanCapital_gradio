#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Prediction Script for CANCapital

This script loads saved models and makes predictions on new data.
Supports multiple model types from sklearn with automatic detection.

Enhanced with robust currency formatting handling for financial datasets.
"""

import json
import os
import sys
from pathlib import Path


def load_model_metadata(model_dir):
    """Load model metadata from JSON file."""
    try:
        metadata_path = Path(model_dir) / "model_metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        return metadata
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None


def detect_available_models(model_dir):
    """Detect which model files are available."""
    models = []

    # Check for common sklearn model file patterns
    model_files = {
        "best_model.pkl": "Best Model (Auto-selected)",
        "gbr_model.pkl": "Gradient Boosting Regressor",
        "linearreg_model.pkl": "Linear Regression",
        "rf_model.pkl": "Random Forest",
        "model.pkl": "Primary Model",
    }

    for filename, display_name in model_files.items():
        file_path = Path(model_dir) / filename
        if file_path.exists():
            models.append((filename, display_name))

    return models


def load_model(model_path):
    """Load a pickled model."""
    try:
        import joblib

        with open(model_path, "rb") as f:
            model = joblib.load(f)

        print(f"Successfully loaded model from: {model_path}")
        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def engineer_prediction_features(df):
    """Engineer features exactly like in training pipeline."""
    try:
        import numpy as np
        import pandas as pd

        print(f"Applying feature engineering for prediction...")

        # Make a copy to avoid modifying original
        processed_df = df.copy()

        # Ensure date columns are datetime
        date_columns = [
            "funded_date",
            "pg_date_of_birth",
            "last_payment_date",
            "demand_letter_sent_date",
            "collection_case_open_date",
            "collection_case_close_date",
        ]

        for col in date_columns:
            if col in processed_df.columns:
                try:
                    processed_df[col] = pd.to_datetime(
                        processed_df[col], errors="coerce"
                    )
                except:
                    print(f"Warning: Could not convert {col} to datetime")

        # borrower age at funding
        if (
            "pg_date_of_birth" in processed_df.columns
            and "funded_date" in processed_df.columns
        ):
            processed_df["borrower_age"] = (
                processed_df["funded_date"]
                - pd.to_datetime(processed_df["pg_date_of_birth"], errors="coerce")
            ).dt.days / 365.25
        else:
            processed_df["borrower_age"] = np.nan

        # time between funding and case open
        if (
            "funded_date" in processed_df.columns
            and "collection_case_open_date" in processed_df.columns
        ):
            processed_df["days_funded_to_open"] = (
                processed_df["collection_case_open_date"] - processed_df["funded_date"]
            ).dt.days
        else:
            processed_df["days_funded_to_open"] = np.nan

        # time between last payment and close
        if (
            "last_payment_date" in processed_df.columns
            and "collection_case_close_date" in processed_df.columns
        ):
            processed_df["days_lastpay_to_close"] = (
                processed_df["collection_case_close_date"]
                - processed_df["last_payment_date"]
            ).dt.days
        else:
            processed_df["days_lastpay_to_close"] = np.nan

        # flags for missing important dates
        for col in ["collection_case_close_date", "demand_letter_sent_date"]:
            if col in processed_df.columns:
                flag = f"has_{col}"
                processed_df[flag] = processed_df[col].notna().astype(int)

        print(f"Feature engineering complete. Added engineered features:")
        new_features = ["borrower_age", "days_funded_to_open", "days_lastpay_to_close"]
        for feature in new_features:
            if feature in processed_df.columns:
                non_null = processed_df[feature].notna().sum()
                print(f"  {feature}: {non_null}/{len(processed_df)} non-null values")

        for col in ["has_collection_case_close_date", "has_demand_letter_sent_date"]:
            if col in processed_df.columns:
                print(
                    f"  {col}: {processed_df[col].sum()}/{len(processed_df)} have the date"
                )

        return processed_df

    except Exception as e:
        print(f"Error in feature engineering: {e}")
        return df


def preprocess_data(df, target_column=None):
    """Preprocess data for prediction - with enhanced currency handling."""
    try:
        import pandas as pd
        
        print(f"Original data shape: {df.shape}")

        # Import currency utilities
        try:
            from currency_utils import enhanced_preprocess_for_prediction
            
            # Use the enhanced preprocessing pipeline
            print("🔧 Using enhanced currency-aware preprocessing...")
            
            processed_df, preprocessor_report = enhanced_preprocess_for_prediction(df, target_column)
            
            # Print preprocessing summary
            if preprocessor_report['currency_columns_detected']:
                print(f"✅ Currency processing completed:")
                print(f"   - Detected currency columns: {preprocessor_report['currency_columns_detected']}")
                print(f"   - Converted values: {preprocessor_report['data_cleaning_summary'].get('total_values_converted', 0)}")
            else:
                print("ℹ️ No currency columns detected in this dataset")
                
        except ImportError as e:
            print(f"⚠️ Enhanced currency utilities not available ({e})")
            print("🔄 Falling back to standard preprocessing...")
            
            # Fallback: Apply feature engineering first
            processed_df = engineer_prediction_features(df)

            # Handle missing values - simple imputation for numeric columns
            numeric_columns = processed_df.select_dtypes(include=["number"]).columns
            for col in numeric_columns:
                if processed_df[col].isnull().sum() > 0:
                    mean_value = processed_df[col].mean()
                    if pd.isna(mean_value):  # If still NaN, use a default
                        mean_value = 0 if "age" in col.lower() else 50
                    processed_df.loc[:, col] = processed_df[col].fillna(mean_value)
                    print(
                        f"Filled {processed_df[col].isnull().sum()} missing values in '{col}' with mean: {mean_value:.2f}"
                    )

            # Handle categorical columns - simple approach
            categorical_columns = processed_df.select_dtypes(
                include=["object", "category"]
            ).columns
            for col in categorical_columns:
                if processed_df[col].isnull().sum() > 0:
                    mode_value = processed_df[col].mode()
                    if len(mode_value) > 0:
                        processed_df.loc[:, col] = processed_df[col].fillna(mode_value[0])
                    else:
                        processed_df.loc[:, col] = processed_df[col].fillna("Unknown")

        # Remove target column if present (for prediction)
        if target_column and target_column in processed_df.columns:
            print(f"Removing target column '{target_column}' from prediction data")
            processed_df = processed_df.drop(columns=[target_column])

        # Remove non-feature columns that aren't used in modeling
        drop_columns = ["collection_case_id"]  # Common ID column to remove
        for col in drop_columns:
            if col in processed_df.columns:
                processed_df = processed_df.drop(columns=[col])

        print(f"Final processed data shape: {processed_df.shape}")
        
        # Final validation
        numeric_cols = processed_df.select_dtypes(include=["number"]).columns.tolist()
        non_numeric_cols = processed_df.select_dtypes(exclude=["number", "category"]).columns.tolist()
        
        # Exclude ID columns
        id_columns = []
        for col in processed_df.columns:
            if any(keyword in col.lower() for keyword in ['id', 'loan_id']) and processed_df[col].dtype == 'object':
                id_columns.append(col)
        
        remaining_non_numeric = [col for col in non_numeric_cols if col not in id_columns]
        
        if remaining_non_numeric:
            print(f"⚠️ Warning: Non-numeric columns remain (will be excluded from modeling): {remaining_non_numeric}")
            # Remove remaining non-numeric columns
            processed_df = processed_df.drop(columns=remaining_non_numeric)
        
        return processed_df

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return df


def make_predictions(model, data, model_name="Model"):
    """Make predictions using the loaded model."""
    try:
        import pandas as pd

        if hasattr(model, "predict"):
            # Debug: Show what columns we have vs what model expects
            print(f"Data columns available ({len(data.columns)}): {list(data.columns)}")

            # Try to get feature names if model has them
            try:
                if hasattr(model, "feature_names_in_"):
                    expected_features = model.feature_names_in_
                    print(
                        f"Model expects features ({len(expected_features)}): {list(expected_features)}"
                    )

                    # Check for missing columns
                    missing_cols = set(expected_features) - set(data.columns)
                    extra_cols = set(data.columns) - set(expected_features)

                    if missing_cols:
                        print(f"❌ Missing required columns: {missing_cols}")

                        # Try to add missing columns with default values
                        for col in missing_cols:
                            if "age" in col.lower():
                                data[col] = 35.0
                                print(
                                    f"Added missing column '{col}' with default value 35.0"
                                )
                            elif "days" in col.lower():
                                data[col] = 30.0
                                print(
                                    f"Added missing column '{col}' with default value 30.0"
                                )
                            elif "has_" in col:
                                data[col] = 1
                                print(
                                    f"Added missing column '{col}' with default value 1"
                                )
                            else:
                                data[col] = 0.5
                                print(
                                    f"Added missing column '{col}' with default value 0.5"
                                )

                    if extra_cols:
                        # Remove extra columns that model doesn't expect
                        data = data.drop(columns=list(extra_cols))
                        print(f"Removed extra columns: {list(extra_cols)}")

                    # Reorder to match expected features
                    data = data.reindex(columns=expected_features, fill_value=0)

                else:
                    print("Model doesn't have feature_names_in_ - using data as-is")
            except Exception as e:
                print(f"Feature name debugging failed: {e}")

            predictions = model.predict(data)

            # Determine if it's classification or regression
            if hasattr(model, "predict_proba"):
                # Classification model - get probabilities too
                try:
                    proba = model.predict_proba(data)

                    print(f"\n{'=' * 60}")
                    print(f"CLASSIFICATION PREDICTIONS - {model_name}")
                    print(f"{'=' * 60}")

                    # Handle binary vs multi-class
                    if proba.shape[1] == 2:
                        print(f"Binary Classification Results:")
                        for i, (pred, prob_neg, prob_pos) in enumerate(
                            zip(predictions, proba[:, 0], proba[:, 1])
                        ):
                            confidence = max(prob_neg, prob_pos)
                            print(
                                f"Sample {i + 1}: Prediction = {pred} (Confidence: {confidence:.3f})"
                            )
                    else:
                        print(f"Multi-class Classification Results:")
                        classes = model.classes_
                        for i, pred in enumerate(predictions):
                            class_idx = (
                                list(classes).index(pred) if pred in classes else 0
                            )
                            confidence = max(proba[i])
                            print(
                                f"Sample {i + 1}: Prediction = {pred} (Confidence: {confidence:.3f})"
                            )

                    return predictions, proba

                except Exception as e:
                    print(f"Could not get prediction probabilities: {e}")
                    predictions = model.predict(data)

            else:
                # Regression model
                print(f"\n{'=' * 60}")
                print(f"REGRESSION PREDICTIONS - {model_name}")
                print(f"{'=' * 60}")

                # Show summary statistics
                mean_pred = predictions.mean()
                std_pred = predictions.std()
                min_pred = predictions.min()
                max_pred = predictions.max()

                print(f"Prediction Summary:")
                print(f"  Mean: {mean_pred:.2f}")
                print(f"  Std Dev: {std_pred:.2f}")
                print(f"  Range: [{min_pred:.2f}, {max_pred:.2f}]")
                print(f"  Count: {len(predictions)}")

                # Show first few predictions
                print(f"\nFirst 10 Predictions:")
                for i, pred in enumerate(predictions[:10]):
                    print(f"  Sample {i + 1}: {pred:.2f}")

                if len(predictions) > 10:
                    print(f"  ... and {len(predictions) - 10} more")

            return predictions, None

        else:
            print("Loaded object is not a valid sklearn model")
            return None, None

    except Exception as e:
        print(f"Error making predictions: {e}")
        return None, None


def main():
    """Main prediction function."""

    if len(sys.argv) != 3:
        print(
            "Usage: python predict_with_saved_model.py <model_directory> <data_file.csv>"
        )
        print(
            "Example: python predict_with_saved_model.py ./trained_models/model_20251113_163053 new_data.csv"
        )
        sys.exit(1)

    model_directory = sys.argv[1]
    data_file = sys.argv[2]

    print("=" * 80)
    print("CANCapital Model Prediction - Enhanced with Currency Handling")
    print("=" * 80)

    # Load metadata
    print(f"\nLoading model from: {model_directory}")
    metadata = load_model_metadata(model_directory)

    if not metadata:
        print("Failed to load model metadata. Exiting.")
        sys.exit(1)

    print(f"Model Information:")
    for key, value in metadata.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Detect available models
    available_models = detect_available_models(model_directory)
    print(f"\nAvailable model files: {[name for name, _ in available_models]}")

    # Load data
    try:
        import pandas as pd

        if not Path(data_file).exists():
            print(f"Data file not found: {data_file}")
            sys.exit(1)

        df = pd.read_csv(data_file, low_memory=False)
        print(f"\nLoaded data file: {data_file}")
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Preprocess data
    target_column = metadata.get("target_column")
    
    try:
        processed_data = preprocess_data(df, target_column)
        
        if processed_data is None or len(processed_data) == 0:
            print("❌ Data preprocessing failed completely")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Data preprocessing failed with error: {e}")
        sys.exit(1)

    # Try to load and use the best model first
    predictions_made = False

    # Priority order for model selection
    model_priority = [
        "best_model.pkl",
        "gbr_model.pkl",
        "linearreg_model.pkl",
        "rf_model.pkl",
        "model.pkl",
    ]

    for model_filename, display_name in available_models:
        if any(priority in model_filename for priority in model_priority):
            try:
                model_path = Path(model_directory) / model_filename
                print(f"\n{'-' * 40}")
                print(f"Loading {display_name} from: {model_filename}")
                print(f"{'-' * 40}")

                model = load_model(model_path)
                if model is not None:
                    predictions, probabilities = make_predictions(
                        model, processed_data, display_name
                    )

                    if predictions is not None:
                        print(
                            f"\n✅ Successfully made predictions using {display_name}"
                        )

                        # Save results to file
                        output_file = f"predictions_{Path(model_filename).stem}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"

                        with open(output_file, "w") as f:
                            f.write("CANCapital Model Prediction Results\n")
                            f.write(f"Generated: {pd.Timestamp.now()}\n")
                            f.write("=" * 50 + "\n\n")
                            f.write(f"Model Used: {display_name}\n")
                            f.write(f"Model File: {model_filename}\n")
                            f.write(f"Target Column: {target_column}\n\n")

                            if probabilities is not None:
                                f.write("Predictions with Probabilities:\n")

                                if probabilities.shape[1] == 2:
                                    # Binary classification
                                    for i, (pred, prob_neg, prob_pos) in enumerate(
                                        zip(
                                            predictions,
                                            probabilities[:, 0],
                                            probabilities[:, 1],
                                        )
                                    ):
                                        confidence = max(prob_neg, prob_pos)
                                        f.write(
                                            f"Sample {i + 1}: Prediction = {pred} (Confidence: {confidence:.3f})\n"
                                        )
                                else:
                                    # Multi-class
                                    classes = model.classes_
                                    for i, pred in enumerate(predictions):
                                        class_idx = (
                                            list(classes).index(pred)
                                            if pred in classes
                                            else 0
                                        )
                                        confidence = max(probabilities[i])
                                        f.write(
                                            f"Sample {i + 1}: Prediction = {pred} (Confidence: {confidence:.3f})\n"
                                        )
                            else:
                                # Regression or classification without probabilities
                                f.write("Predictions:\n")
                                for i, pred in enumerate(predictions):
                                    f.write(f"Sample {i + 1}: {pred:.4f}\n")

                            f.write(f"\nTotal predictions: {len(predictions)}\n")
                            if probabilities is None:
                                stats = {
                                    "mean": float(predictions.mean()),
                                    "std": float(predictions.std()),
                                    "min": float(predictions.min()),
                                    "max": float(predictions.max()),
                                }
                                f.write(
                                    f"Statistics: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}, Min={stats['min']:.2f}, Max={stats['max']:.2f}\n"
                                )

                        print(f"\n📄 Prediction results saved to: {output_file}")
                        predictions_made = True

                        # If this was the best model, we can stop here
                        if "best" in model_filename:
                            break

            except Exception as e:
                print(f"❌ Failed to use {display_name}: {e}")
                continue

    if not predictions_made:
        print("\n❌ No models could successfully make predictions on this data")
        
        # Additional debugging info
        print("\n🔍 Debugging Information:")
        try:
            import pandas as pd
            
            df_debug = pd.read_csv(data_file, low_memory=False)
            
            print(f"Data file shape: {df_debug.shape}")
            print(f"Columns with numeric dtype:")
            for col in df_debug.select_dtypes(include=["number"]).columns:
                print(f"  - {col}")
            
            print(f"\nColumns with object dtype (potential currency/formatted columns):")
            for col in df_debug.select_dtypes(include=["object"]).columns:
                sample_vals = df_debug[col].dropna().head(3).tolist()
                print(f"  - {col}: Sample values = {sample_vals}")
                
        except Exception as debug_e:
            print(f"Debug info failed: {debug_e}")
            
        sys.exit(1)

    print(f"\n{'=' * 80}")
    print("PREDICTION COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
