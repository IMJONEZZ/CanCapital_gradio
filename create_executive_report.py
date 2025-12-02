#!/usr/bin/env python3
"""
Create executive-ready prediction report
Merges predictions with original dataset
"""
import pandas as pd
import numpy as np

# Load original dataset
print("Loading original dataset...")
df_original = pd.read_csv("data/Collection Case Data 2025.08.21.xlsx - Result 1.csv")
print(f"Original data: {df_original.shape}")

# Load predictions
print("Loading predictions...")
predictions = []
with open("predictions_best_model_20251202_124159.txt", "r") as f:
    for line in f:
        if line.startswith("Sample"):
            # Parse: "Sample 1: 5.2403"
            parts = line.strip().split(":")
            if len(parts) == 2:
                pred_value = float(parts[1].strip())
                predictions.append(pred_value)

print(f"Loaded {len(predictions)} predictions")

# Add predictions to dataframe
if len(predictions) == len(df_original) - 1:  # Excluding header
    df_original['predicted_collection_days'] = [np.nan] + predictions
elif len(predictions) == len(df_original):
    df_original['predicted_collection_days'] = predictions
else:
    print(f"Warning: Prediction count mismatch. Original: {len(df_original)}, Predictions: {len(predictions)}")
    df_original['predicted_collection_days'] = predictions[:len(df_original)]

# Add prediction categories for easy filtering
df_original['prediction_category'] = pd.cut(
    df_original['predicted_collection_days'],
    bins=[0, 2.5, 4.0, 6.0],
    labels=['Quick Resolution (< 2.5 days)', 'Medium (2.5-4 days)', 'Extended (> 4 days)']
)

# Create summary statistics
print("\n" + "="*60)
print("PREDICTION SUMMARY")
print("="*60)
print(f"Total cases analyzed: {len(df_original)}")
print(f"\nPrediction Statistics:")
print(f"  Mean: {df_original['predicted_collection_days'].mean():.2f} days")
print(f"  Median: {df_original['predicted_collection_days'].median():.2f} days")
print(f"  Min: {df_original['predicted_collection_days'].min():.2f} days")
print(f"  Max: {df_original['predicted_collection_days'].max():.2f} days")
print(f"\nBy Category:")
print(df_original['prediction_category'].value_counts().sort_index())

# Save to CSV
output_file = "executive_predictions_report.csv"
df_original.to_csv(output_file, index=False)
print(f"\n✅ Executive report saved to: {output_file}")

# Create a summary-only version
summary_cols = [
    'contract_number', 
    'borrower',
    'funded_date',
    'collection_case_days',  # Actual (if available)
    'predicted_collection_days',
    'prediction_category',
    'collection_status',
    ' original_funded_amount ',
    'percent_paid'
]

# Only include columns that exist
available_cols = [col for col in summary_cols if col in df_original.columns]
df_summary = df_original[available_cols]

summary_file = "executive_predictions_summary.csv"
df_summary.to_csv(summary_file, index=False)
print(f"✅ Summary report saved to: {summary_file}")

print("\n" + "="*60)
print("Files ready for executive presentation!")
print("="*60)
