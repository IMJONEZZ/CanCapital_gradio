# Import necessary libraries
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# Set plot style for better visuals
sns.set_style("whitegrid")
plt.style.use("seaborn-v0_8-dark-palette")


def run_complete_eda(csv_file_path):
    """
    Run complete EDA analysis on a dataset and return results as text.

    Args:
        csv_file_path: Path to the CSV file to analyze

    Returns:
        str: Complete EDA results as formatted text
    """

    # Create plots directory if it doesn't exist
    plots_dir = Path("eda_plots")
    plots_dir.mkdir(exist_ok=True)

    results_text = []

    try:
        # Load your dataset
        df = pd.read_csv(csv_file_path)

        # Basic information about the dataset
        results_text.append("Dataset Shape:" + str(df.shape))
        results_text.append("\nFirst 5 Rows:")
        results_text.append(str(df.head()))

        results_text.append("\nDataset Info:")
        # Capture info output
        import io

        buffer = io.StringIO()
        df.info(buf=buffer)
        results_text.append(buffer.getvalue())

        results_text.append("\nStatistical Summary:")
        results_text.append(str(df.describe().T))  # Transposed for better readability

        # Check for missing values
        results_text.append("\nMissing Values:")
        missing_values = df.isnull().sum()
        missing_cols = missing_values[missing_values > 0]
        if len(missing_cols) > 0:
            results_text.append(str(missing_cols))
        else:
            results_text.append("No missing values found.")

        # Handle missing values (example strategies)
        # For numerical columns: fill with median
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())

        # For categorical columns: fill with mode
        for col in df.select_dtypes(include=["object"]).columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])

        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        results_text.append(f"\nNumber of duplicate rows: {duplicates}")
        if duplicates > 0:
            df = df.drop_duplicates()

        # Encode categorical variables for correlation analysis
        df_encoded = df.copy()
        label_encoders = {}
        for col in df_encoded.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le

        # Univariate Analysis: Distribution of numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        n_cols = len(numerical_cols)

        if n_cols > 0:
            results_text.append(f"\nFound {n_cols} numerical columns for analysis.")

            n_rows = (n_cols // 3) + 1 if n_cols % 3 != 0 else n_cols // 3

            plt.figure(figsize=(18, 6 * n_rows))
            for i, col in enumerate(numerical_cols, 1):
                plt.subplot(n_rows, 3, i)
                sns.histplot(df[col], kde=True)
                plt.title(f"Distribution of {col}")
            plt.tight_layout()
            plt.savefig(plots_dir / "distributions.png", dpi=300, bbox_inches="tight")
            plt.close()

            # Box plots to identify outliers
            plt.figure(figsize=(18, 6 * n_rows))
            for i, col in enumerate(numerical_cols, 1):
                plt.subplot(n_rows, 3, i)
                sns.boxplot(y=df[col])
                plt.title(f"Box Plot of {col}")
            plt.tight_layout()
            plt.savefig(plots_dir / "box_plots.png", dpi=300, bbox_inches="tight")
            plt.close()

        # Count plots for categorical variables
        categorical_cols = df.select_dtypes(include=["object"]).columns

        if len(categorical_cols) > 0:
            results_text.append(
                f"\nFound {len(categorical_cols)} categorical columns for analysis."
            )

            n_cat_cols = len(categorical_cols)
            n_cat_rows = (
                (n_cat_cols // 3) + 1 if n_cat_cols % 3 != 0 else n_cat_cols // 3
            )

            plt.figure(figsize=(18, 6 * n_cat_rows))
            for i, col in enumerate(
                categorical_cols[:12], 1
            ):  # Limit to first 12 categorical columns
                if i > 12:  # Don't create too many plots
                    break
                plt.subplot(n_cat_rows, 3, i)

                # Get value counts and limit to top 10 for readability
                value_counts = df[col].value_counts().head(10)

                if (
                    len(value_counts) > 5
                ):  # If too many categories, use horizontal bar plot
                    sns.countplot(y=df[col], order=value_counts.index)
                else:
                    sns.countplot(x=df[col])
                plt.title(f"Count of {col}")
            plt.tight_layout()
            plt.savefig(
                plots_dir / "categorical_counts.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

        # Correlation Analysis
        if len(df_encoded.columns) > 1:
            plt.figure(figsize=(12, 8))
            correlation_matrix = df_encoded.corr()
            sns.heatmap(
                correlation_matrix,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                linewidths=0.5,
            )
            plt.title("Correlation Matrix")
            plt.tight_layout()
            plt.savefig(
                plots_dir / "correlation_matrix.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

        # Identify highly correlated features (potential predictors)
        def get_high_correlation_pairs(df, threshold=0.7):
            corr_matrix = df.corr().abs()
            upper_triangle = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            high_corr = (
                corr_matrix.where(upper_triangle).stack().sort_values(ascending=False)
            )
            return high_corr[high_corr > threshold]

        if len(df_encoded.columns) > 1:
            high_corr_pairs = get_high_correlation_pairs(df_encoded, 0.7)
            if not high_corr_pairs.empty:
                results_text.append(
                    "Highly Correlated Feature Pairs (|correlation| > 0.7):"
                )
                for idx, value in high_corr_pairs.items():
                    results_text.append(f"{idx[0]} and {idx[1]}: {value:.2f}")
            else:
                results_text.append(
                    "\nNo highly correlated feature pairs found (threshold: 0.7)"
                )

        # Bivariate Analysis (example with first numerical column as target)
        if len(numerical_cols) > 1:
            target_col = numerical_cols[0]  # Change this to your target variable
            results_text.append(
                f"\nAnalyzing relationships with {target_col} (as example target):"
            )

            # Correlation with target variable
            target_corr = df_encoded.corr()[target_col].sort_values(ascending=False)
            results_text.append(f"Top 10 features correlated with {target_col}:")
            for feature, corr_value in target_corr[
                1:11
            ].items():  # Exclude the target itself
                results_text.append(f"  {feature}: {corr_value:.3f}")

            # Scatter plots for top correlated features
            if len(target_corr) > 1:  # If there are other features to plot
                top_features = target_corr[1:6].index  # Top 5 features (excluding self)
                if len(top_features) > 0:
                    plt.figure(figsize=(18, 4))
                    for i, feature in enumerate(top_features[:5], 1):
                        plt.subplot(1, min(len(top_features), 5), i)
                        sns.scatterplot(x=df[feature], y=df[target_col])
                        plt.title(f"{feature} vs {target_col}")
                        plt.xlabel(feature)
                        plt.ylabel(target_col)
                    plt.tight_layout()
                    plt.savefig(
                        plots_dir / "scatter_plots.png", dpi=300, bbox_inches="tight"
                    )
                    plt.close()

        # Multivariate Analysis: Pairplot for a subset of numerical features
        if len(numerical_cols) > 1:
            subset_cols = numerical_cols[
                : min(5, len(numerical_cols))
            ]  # Limit to first 5 columns
            if len(subset_cols) > 1:
                sns.pairplot(df[subset_cols])
                plt.suptitle("Pairplot of Numerical Features", y=1.02)
                plt.savefig(plots_dir / "pairplot.png", dpi=300, bbox_inches="tight")
                plt.close()

        # Summary of findings
        results_text.append("\n=== EDA SUMMARY ===")
        results_text.append(f"Dataset shape: {df.shape}")
        results_text.append(
            f"Numerical columns ({len(numerical_cols)}): " + str(list(numerical_cols))
        )
        results_text.append(
            f"Categorical columns ({len(categorical_cols)}): "
            + str(list(categorical_cols))
        )

        missing_after = df.isnull().sum()[df.isnull().sum() > 0]
        if len(missing_after) > 0:
            results_text.append("Missing values after handling:")
            results_text.append(str(missing_after))
        else:
            results_text.append("Missing values after handling: None")

        # Final note about predictive features
        results_text.append("\nPotential Predictive Features:")
        if "target_corr" in locals() and not target_corr.empty:
            results_text.append(
                "Based on correlation analysis, the following features might be predictive:"
            )
            for feature, corr_value in target_corr[1:6].items():
                results_text.append(f"  - {feature}: correlation = {corr_value:.3f}")
        else:
            results_text.append(
                "No strong correlations found. Consider domain knowledge or advanced feature engineering."
            )

        # Dataset quality assessment
        results_text.append("\n=== DATASET QUALITY ASSESSMENT ===")
        results_text.append(f"Total rows: {len(df):,}")
        results_text.append(f"Total columns: {len(df.columns)}")
        results_text.append(f"Duplicate rows removed: {duplicates}")

        if len(missing_cols) > 0:
            results_text.append(f"Columns with missing data: {len(missing_cols)}")
            worst_missing = missing_cols.max()
            results_text.append(
                f"Worst missing data: {worst_missing:.1%} in column '{missing_cols.idxmax()}'"
            )
        else:
            results_text.append(
                "Dataset appears complete - no missing values detected!"
            )

        # Data types summary
        type_summary = df.dtypes.value_counts()
        results_text.append(f"\nData types distribution:")
        for dtype, count in type_summary.items():
            results_text.append(f"  {dtype}: {count} columns")

        return "\n".join(results_text)

    except Exception as e:
        error_msg = f"Error during EDA analysis: {str(e)}"
        results_text.append(error_msg)
        return "\n".join(results_text)


# Legacy standalone execution (for backward compatibility)
if __name__ == "__main__":
    csv_file_path = "dataset.csv"

    if Path(csv_file_path).exists():
        results = run_complete_eda(csv_file_path)
        print(results)
    else:
        print(
            f"Dataset file '{csv_file_path}' not found. Please ensure the dataset exists."
        )
