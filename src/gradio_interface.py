#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clean Essential Gradio Interface for CANCapital EDA & Modeling Pipeline

This interface provides the core functionality:
1. Basic EDA Analysis (universal compatibility)
2. ML Pipeline with dataset compatibility checking
3. Model Prediction using saved models

Designed for maximum compatibility and reliability.
"""

import os
import subprocess
import sys
from pathlib import Path

# Try to import gradio with fallback handling
try:
    import gradio as gr
except ImportError:
    print("Error: Gradio not installed. Install with: pip install gradio")
    sys.exit(1)


def run_eda_analysis(csv_file_path):
    """Run complete EDA analysis on uploaded data and return full results."""
    try:
        if not csv_file_path or not Path(csv_file_path).exists():
            return "Please upload a CSV file first!", None

        # Copy uploaded file temporarily for processing
        import shutil
        from datetime import datetime

        temp_file = "temp_eda_data.csv"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(csv_file_path, temp_file)

        # Import and run the complete EDA function
        sys.path.append(".")
        from src.eda import run_complete_eda

        # Run complete EDA analysis
        full_results = run_complete_eda(temp_file)

        # Clean up temp file
        if Path(temp_file).exists():
            os.remove(temp_file)

        # Create downloadable text file
        download_filename = f"eda_results_{timestamp}.txt"
        with open(download_filename, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("COMPLETE EDA ANALYSIS RESULTS\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            f.write(full_results)
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("ANALYSIS COMPLETE - All plots saved to 'eda_plots' directory\n")
            f.write("=" * 80 + "\n")

        # Prepare display results (first part for readability)
        if len(full_results) > 8000:  # If results are very long, show first part
            display_results = (
                f"EDA Analysis Complete!\n\n(full results available in downloadable file)\n\n"
                + full_results[:8000]
                + "\n\n... (results truncated for display, see downloadable file for complete results)"
            )
        else:
            display_results = f"EDA Analysis Complete!\n\n{full_results}"

        return display_results, download_filename

    except Exception as e:
        error_msg = f"Error running EDA: {str(e)}"
        return error_msg, None


def update_eda_button_state(file_path):
    """Enable/disable Run button based on file upload status."""
    if file_path and Path(file_path).exists():
        return (
            gr.Button(interactive=True, elem_classes=[]),
            '<span class="status-badge status-ready">✓ File ready</span>'
        )
    else:
        return (
            gr.Button(interactive=False, elem_classes=["eda-btn-disabled"]),
            '<span class="status-badge status-pending">No file uploaded</span>'
        )


def update_ml_button_state(file_path, columns_loaded):
    """Enable/disable ML Run button based on file + columns state."""
    if file_path and columns_loaded:
        return (
            gr.Button(interactive=True, elem_classes=[]),
            '<span class="status-badge status-ready">✓ Ready to train</span>'
        )
    elif file_path:
        return (
            gr.Button(interactive=False, elem_classes=["eda-btn-disabled"]),
            '<span class="status-badge status-pending">Load columns first</span>'
        )
    else:
        return (
            gr.Button(interactive=False, elem_classes=["eda-btn-disabled"]),
            '<span class="status-badge status-pending">Upload file first</span>'
        )


def reset_ml_state(file_path):
    """Reset ML state when file changes - columns need reloading."""
    if file_path and Path(file_path).exists():
        return (
            gr.Button(interactive=False, elem_classes=["eda-btn-disabled"]),
            '<span class="status-badge status-pending">Load columns first</span>',
            False  # columns_loaded = False
        )
    else:
        return (
            gr.Button(interactive=False, elem_classes=["eda-btn-disabled"]),
            '<span class="status-badge status-pending">Upload file first</span>',
            False  # columns_loaded = False
        )


def update_prediction_button_state(model_selected, file_uploaded):
    """Enable/disable Predict button based on model + file state."""
    if model_selected and file_uploaded:
        return (
            gr.Button(interactive=True, elem_classes=[]),
            '<span class="status-badge status-ready">✓ Ready to predict</span>'
        )
    elif model_selected:
        return (
            gr.Button(interactive=False, elem_classes=["eda-btn-disabled"]),
            '<span class="status-badge status-pending">Upload data file</span>'
        )
    elif file_uploaded:
        return (
            gr.Button(interactive=False, elem_classes=["eda-btn-disabled"]),
            '<span class="status-badge status-pending">Select a model</span>'
        )
    else:
        return (
            gr.Button(interactive=False, elem_classes=["eda-btn-disabled"]),
            '<span class="status-badge status-pending">Select model and upload file</span>'
        )


def update_prediction_file_state(file_path):
    """Update file upload state and status badge."""
    if file_path and Path(file_path).exists():
        return (
            True,
            '<span class="status-badge status-ready">✓ File uploaded</span>'
        )
    else:
        return (
            False,
            '<span class="status-badge status-pending">No file uploaded</span>'
        )


def update_model_selection_state(model_name):
    """Update model selection state and status badge."""
    if model_name:
        return (
            True,
            f'<span class="status-badge status-ready">✓ {model_name}</span>'
        )
    else:
        return (
            False,
            '<span class="status-badge status-pending">No model selected</span>'
        )


def run_ml_pipeline(csv_file_path, target_column=None, output_dir="ml_results"):
    """Run the enhanced ML pipeline."""
    try:
        if not csv_file_path or not Path(csv_file_path).exists():
            return "Please upload a CSV file first!", None

        # Check dataset compatibility first
        compatibility = check_ml_pipeline_compatibility(csv_file_path)

        if not compatibility["compatible"]:
            missing_cols = ", ".join(compatibility.get("missing_required", []))
            error_msg = f"""Dataset Not Compatible with ML Pipeline

This dataset appears to be missing loan collection specific columns required for the ML pipeline.

**Missing Required Columns:** {", ".join(compatibility.get("missing_required", ["Unknown"]))}

**Recommendation:**
• Use "Basic EDA Analysis" tab instead - it works with ANY dataset format
• The ML pipeline is specifically designed for loan collection datasets

**Dataset Info:**
• Total columns: {compatibility.get("total_columns", "Unknown")}
• File: {Path(csv_file_path).name}

For universal data analysis regardless of format, the Basic EDA provides comprehensive insights."""

            if compatibility.get("suggestion"):
                error_msg += f"\n\nSuggestion: {compatibility['suggestion']}"

            return error_msg, None

        # Handle target column - could be string, dict, or None
        if target_column:
            if isinstance(target_column, dict):
                # Extract the actual column name from dictionary
                target_col = target_column.get("value", str(target_column))
            else:
                # Already a string or other format
                target_col = str(target_column)
        else:
            target_col = None

        # Copy uploaded file temporarily
        import shutil

        temp_file = "temp_ml_data.csv"
        shutil.copy2(csv_file_path, temp_file)

        # Build command
        cmd = [sys.executable, "./src/eda_and_model.py", temp_file]

        if target_col:
            cmd.extend(["--target", target_col])

        cmd.extend(["--output_dir", output_dir])

        # Run ML pipeline
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes for ML pipeline
        )

        # Clean up temp file
        if Path(temp_file).exists():
            os.remove(temp_file)

        # Add debug info to results if there was an issue
        if result.returncode != 0:
            error_msg = f"ML Pipeline Failed\n\n"

            # Check if it's a compatibility issue (missing columns)
            if (
                "keyerror:" in result.stderr.lower()
                or "not in df.columns" in result.stderr.lower()
            ):
                error_msg += """Likely Dataset Compatibility Issue

The pipeline failed because your dataset is missing specific loan collection columns that the ML pipeline expects.

**This typically happens when:**
• Using a non-loan dataset with the ML Pipeline
• Dataset lacks financial collection case columns

**SOLUTION:**
Use "Basic EDA Analysis" tab instead - it works with ANY dataset format
Don't force ML Pipeline on incompatible datasets

**Technical Details:**"""

            # Show stderr (actual errors)
            if result.stderr:
                error_msg += f"\n\nError Details:\n{result.stderr[:500]}"
                if len(result.stderr) > 500:
                    error_msg += "\n... (error truncated for display)\n"

            # Show stdout for additional debugging info
            if result.stdout:
                error_msg += f"\n\nProcessing Log:\n{result.stdout[:300]}"
                if len(result.stdout) > 300:
                    error_msg += "\n... (log truncated for display)\n"

            if target_column:
                error_msg += f"\n\nDebug info: Target column processing completed"
            else:
                error_msg += "\n\nTip: Consider specifying a target column or using Basic EDA for universal analysis."
            return error_msg, None

        if result.returncode == 0:
            # Extract important pipeline information
            lines = result.stdout.split("\n")
            key_lines = []

            for line in lines:
                if any(
                    keyword in line.lower()
                    for keyword in [
                        "auto-detecting target column",
                        "using target column",
                        "pipeline completed successfully",
                        "model performance:",
                        "trained model info:",
                    ]
                ):
                    key_lines.append(line.strip())

            # Add debug info to results (both success and partial)
            result_text = f"ML Pipeline Complete!\n\n"

            # Include initial processing info for transparency
            summary_lines = []
            for line in lines:
                if any(
                    keyword in line.lower()
                    for keyword in [
                        "dataset shape:",
                        "auto-detecting target column:",
                        "using target column:",
                    ]
                ):
                    summary_lines.append(line.strip())

            if summary_lines:
                result_text += (
                    "Processing Summary:\n" + "\n".join(summary_lines) + "\n\n"
                )

            if target_column:
                result_text += f"[Debug] Target column: {target_col}\n\n"
            result_text += "\n".join(key_lines[:10])

            # If there are any warnings in stdout, include them
            if result.stdout and (
                "warning:" in result.stdout.lower() or "error:" in result.stdout.lower()
            ):
                warnings = []
                for line in lines:
                    if (
                        "warning:" in result.stdout.lower()
                        or "error:" in result.stdout.lower()
                    ):
                        warnings.append(line.strip())
                if warnings:
                    result_text += f"\n\nAdditional Info:\n" + "\n".join(warnings[:3])

            # Add model location info to result
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Find the most recent model folder
            trained_models_dir = Path("trained_models")
            model_folder_name = "model_" + timestamp[:8]  # YYYYMMDD prefix
            if trained_models_dir.exists():
                model_folders = sorted(trained_models_dir.iterdir(), key=lambda x: x.name, reverse=True)
                if model_folders:
                    model_folder_name = model_folders[0].name

            result_text += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📁 TRAINED MODEL LOCATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Your model has been saved to:
  trained_models/{model_folder_name}/

Files created:
  • best_model.pkl - Trained model
  • preprocessor.pkl - Feature transformer
  • model_metadata.json - Training info
  • feature_columns.txt - Features used

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
➡️ NEXT STEP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Go to the "Model Prediction" tab to:
  1. Click "Scan for Models" to find your trained model
  2. Upload new data for predictions
  3. Select your model and make predictions
"""
            download_filename = f"ml_pipeline_results_{timestamp}.txt"
            with open(download_filename, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("COMPLETE ML PIPELINE RESULTS\n")
                f.write(
                    f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write("=" * 80 + "\n\n")
                f.write(result_text)
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("FULL PIPELINE OUTPUT:\n")
                f.write(result.stdout)
                f.write("\n\n" + "=" * 80 + "\n")

            return result_text, download_filename

    except subprocess.TimeoutExpired:
        return "ML Pipeline timed out (10 minutes limit)", None
    except Exception as e:
        return f"Error running ML Pipeline: {str(e)}", None


def scan_for_models(models_dir="trained_models"):
    """Scan for available saved models."""
    try:
        if not Path(models_dir).exists():
            return f"Models directory '{models_dir}' not found!"

        models = []

        for model_subdir in Path(models_dir).iterdir():
            if model_subdir.is_dir():
                metadata_file = model_subdir / "model_metadata.json"

                if metadata_file.exists():
                    import json

                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)

                    model_info = (
                        f"{model_subdir.name} | "
                        f"Target: {metadata.get('target_column', 'Unknown')} | "
                        f"Type: {metadata.get('model_type', 'Unknown')}"
                    )
                    models.append(f"FOLDER {model_info}")
                else:
                    models.append(f"{model_subdir.name} | (metadata file missing)")

        if not models:
            return f"No trained models found in '{models_dir}'"

        result = f"Found {len(models)} trained model(s):\n\n"
        for i, model in enumerate(models, 1):
            result += f"{i}. {model}\n"

        return result

    except Exception as e:
        return f"Error scanning models: {str(e)}"


def run_model_prediction(model_directory, new_data_file):
    """Run prediction using a saved model."""
    try:
        if not model_directory or not new_data_file:
            return "Please provide both a model directory and data file!", None

        if not Path(f"./trained_models/{model_directory}").exists():
            return f"Model directory '{model_directory}' not found!", None

        if not new_data_file or not Path(new_data_file).exists():
            return "Please upload a data file for prediction!", None

        # Copy uploaded file temporarily
        import shutil

        temp_file = "temp_prediction_data.csv"
        shutil.copy2(new_data_file, temp_file)

        # Build command with PYTHONPATH to enable src imports
        cmd = [
            sys.executable,
            "src/predict_with_saved_model.py",
            f"./trained_models/{model_directory}",
            temp_file,
        ]

        # Set up environment with PYTHONPATH
        import os
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path.cwd())

        # Run prediction
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes
            cwd=Path.cwd(),  # Explicitly set working directory
            env=env,  # Pass environment with PYTHONPATH
        )

        # Clean up temp file
        if Path(temp_file).exists():
            os.remove(temp_file)

        if result.returncode == 0:
            # Create downloadable results file
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            download_filename = f"prediction_results_{timestamp}.txt"

            with open(download_filename, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("MODEL PREDICTION RESULTS\n")
                f.write(
                    f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(f"Model Directory: {model_directory}\n")
                f.write("=" * 80 + "\n\n")
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n\nWarnings/Errors:\n")
                    f.write(result.stderr)
                f.write("\n\n" + "=" * 80 + "\n")

            return (
                f"Prediction Complete!\n\nOutput:\n{result.stdout[:2000]}",
                download_filename,
            )
        else:
            # Enhanced error reporting
            error_msg = f"Prediction Failed:\n\n"
            error_msg += f"Return Code: {result.returncode}\n\n"

            if result.stderr:
                error_msg += f"STDERR:\n{result.stderr}\n\n"
            else:
                error_msg += "STDERR: (empty)\n\n"

            if result.stdout:
                error_msg += f"STDOUT:\n{result.stdout}"  # Show all output
            else:
                error_msg += "STDOUT: (empty)"

            return error_msg, None

    except subprocess.TimeoutExpired:
        return "Prediction timed out (5 minutes limit)", None
    except Exception as e:
        return f"Error running prediction: {str(e)}", None


def check_ml_pipeline_compatibility(csv_file_path):
    """Check if dataset is compatible with ML pipeline requirements."""

    required_columns = [
        "funded_date",  # Required for chronological split
    ]

    optional_but_used_columns = [
        "collection_case_days",
        "collection_case_close_date",
        "last_payment_date",
        "pg_date_of_birth",
        "demand_letter_sent_date",
        "collection_case_open_date",
    ]

    try:
        if not csv_file_path or not Path(csv_file_path).exists():
            return {"compatible": False, "missing_required": [], "total_columns": 0}

        import pandas as pd

        df = pd.read_csv(csv_file_path)
        dataset_columns = set(df.columns.tolist())

        # Check required columns
        missing_required = []
        for col in required_columns:
            if col not in dataset_columns:
                missing_required.append(col)

        # Check optional but used columns
        missing_optional = []
        for col in optional_but_used_columns:
            if col not in dataset_columns:
                missing_optional.append(col)

        # Determine compatibility
        if missing_required:
            compatible = False
            suggestion = f"Required column '{missing_required[0]}' is missing. This dataset may not be a loan collection dataset."
        elif len(missing_optional) >= 3:  # If missing most of the loan-specific columns
            compatible = False
            suggestion = "Dataset appears to lack many loan collection specific columns. Consider using Basic EDA instead."
        else:
            compatible = True
            suggestion = "Dataset appears compatible with ML pipeline requirements."

        return {
            "compatible": compatible,
            "missing_required": missing_required,
            "missing_optional": missing_optional,
            "total_columns": len(df.columns),
            "dataset_name": Path(csv_file_path).stem,
            "suggestion": suggestion,
        }

    except Exception as e:
        return {
            "compatible": False,
            "missing_required": [],
            "total_columns": 0,
            "error": str(e),
            "suggestion": f"Error reading dataset: {e}",
        }


def get_csv_columns(csv_file_path):
    """Get column names from uploaded CSV for dropdown."""
    try:
        if not csv_file_path or not Path(csv_file_path).exists():
            return []

        import pandas as pd

        df = pd.read_csv(csv_file_path)

        # Return simple list of column names for dropdown
        return df.columns.tolist()

    except Exception as e:
        print(f"Error reading CSV columns: {e}")
        return []


def get_trained_model_folders(models_dir="trained_models"):
    """Get list of folder names from trained_models directory for dropdown."""
    try:
        if not Path(models_dir).exists():
            return []

        folders = []
        for item in Path(models_dir).iterdir():
            if item.is_dir() and not item.name.startswith("."):
                folders.append(item.name)

        return sorted(folders)
    except Exception as e:
        print(f"Error reading model folders: {e}")
        return []


def create_clean_interface():
    """Create a clean, essential Gradio interface."""

    # Custom CSS for dark mode appearance
    css = """
    .gradio-container {
        width: 1200px !important;
        max-width: 1200px !important;
        min-width: 1200px !important;
        margin: auto !important;
    }

    .header {
        text-align: center;
        background: linear-gradient(90deg, #2c3e50 0%, #34495e 100%);
        color: #ecf0f1;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 30px;
    }

    .tab-header {
        color: #ecf0f1;
        font-size: 1.5em;
        margin-bottom: 15px;
    }

    .success {
        color: #2ecc71;
    }

    .error {
        color: #e74c3c;
    }

    /* Dark mode styling */
    body {
        background-color: #2c3e50;
        color: #ecf0f1;
    }

    .gradio-container * {
        background-color: #2c3e50 !important;
        color: #ecf0f1 !important;
    }

    .gradio-tabs {
        background-color: #34495e !important;
    }

    .gradio-tab-content {
        background-color: #2c3e50 !important;
        color: #ecf0f1 !important;
    }

    .gradio-textbox input,
    .gradio-textbox textarea {
        background-color: #34495e !important;
        color: #ecf0f1 !important;
        border-color: #7f8c8d !important;
    }

    .gradio-button {
        background-color: #3498db !important;
        color: white !important;
    }

    .gradio-file {
        background-color: #34495e !important;
        color: #ecf0f1 !important;
    }

    .gradio-dropdown {
        background-color: #34495e !important;
        color: #ecf0f1 !important;
    }

    .gradio-row {
        background-color: #2c3e50 !important;
    }

    .gradio-column {
        background-color: #2c3e50 !important;
    }

    /* Override any light backgrounds */
    .gradio-container,
    .gradio-row,
    .gradio-column,
    div.gradio-container {
        background-color: #2c3e50 !important;
    }

    /* Fix any white backgrounds */
    * {
        background-color: inherit;
        color: inherit;
    }

    /* Ensure proper contrast */
    h1, h2, h3, h4 {
        color: #ecf0f1 !important;
    }

    .success { color: #2ecc71; }
    .error { color: #e74c3c; }

    /* ===== TAB 1: BOLD VISUAL HIERARCHY ===== */

    /* Step container with colored left border */
    .eda-step {
        position: relative;
        padding: 20px;
        margin-bottom: 15px;
        border-radius: 8px;
        border-left: 5px solid;
        background-color: #34495e !important;
    }

    .eda-step-1 { border-left-color: #3498db !important; }
    .eda-step-2 { border-left-color: #9b59b6 !important; }
    .eda-step-3 { border-left-color: #2ecc71 !important; }

    /* Numbered step badges */
    .step-header {
        display: flex;
        align-items: center;
        margin-bottom: 15px;
    }

    .step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        font-weight: bold;
        font-size: 16px;
        margin-right: 12px;
        color: white !important;
    }

    .step-number-1 { background-color: #3498db !important; }
    .step-number-2 { background-color: #9b59b6 !important; }
    .step-number-3 { background-color: #2ecc71 !important; }

    .step-title {
        font-size: 1.2em;
        font-weight: 600;
        color: #ecf0f1 !important;
    }

    /* Visual connector arrow */
    .step-connector {
        text-align: center;
        padding: 8px 0;
        color: #7f8c8d !important;
        font-size: 20px;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.85em;
        font-weight: 500;
        margin-top: 10px;
    }

    .status-pending {
        background-color: rgba(231, 76, 60, 0.2) !important;
        color: #e74c3c !important;
        border: 1px solid #e74c3c;
    }

    .status-ready {
        background-color: rgba(46, 204, 113, 0.2) !important;
        color: #2ecc71 !important;
        border: 1px solid #2ecc71;
    }

    /* Disabled button state */
    .eda-btn-disabled {
        background-color: #5a6d7e !important;
        color: #95a5a6 !important;
        cursor: not-allowed !important;
        opacity: 0.6 !important;
    }

    /* Helper text styling */
    .eda-helper-text {
        background-color: #2c3e50;
        padding: 12px 15px;
        border-radius: 5px;
        border-left: 3px solid #f39c12;
        margin-top: 15px;
    }

    .eda-helper-text code {
        background: rgba(52, 152, 219, 0.2);
        padding: 2px 6px;
        border-radius: 3px;
        font-family: monospace;
    }

    /* ===== TAB 2: ML PIPELINE VISUAL HIERARCHY ===== */

    /* ML Pipeline step containers - reuse eda-step base styling */
    .ml-step {
        position: relative;
        padding: 20px;
        margin-bottom: 15px;
        border-radius: 8px;
        border-left: 5px solid;
        background-color: #34495e !important;
    }

    .ml-step-1 { border-left-color: #3498db !important; }  /* Blue - Upload */
    .ml-step-2 { border-left-color: #9b59b6 !important; }  /* Purple - Configure */
    .ml-step-3 { border-left-color: #e67e22 !important; }  /* Orange - Train */
    .ml-step-4 { border-left-color: #2ecc71 !important; }  /* Green - Results */

    /* Step number background for step 4 */
    .step-number-4 { background-color: #2ecc71 !important; }

    /* ML-specific helper styles */
    .ml-time-estimate {
        color: #7f8c8d;
        font-size: 0.85em;
        margin-top: 10px;
    }

    .ml-next-steps {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        padding: 15px;
        border-radius: 5px;
        margin-top: 15px;
    }

    .ml-next-steps p {
        margin: 0;
        color: white !important;
    }

    /* Columns loaded status badge */
    .status-columns-loaded {
        background-color: rgba(155, 89, 182, 0.2) !important;
        color: #9b59b6 !important;
        border: 1px solid #9b59b6;
    }

    /* ML helper text - orange accent like EDA */
    .ml-helper-text {
        background-color: #2c3e50;
        padding: 12px 15px;
        border-radius: 5px;
        border-left: 3px solid #f39c12;
        margin-top: 15px;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }

    .ml-helper-text code {
        background: rgba(52, 152, 219, 0.2);
        padding: 2px 6px;
        border-radius: 3px;
        font-family: monospace;
    }

    /* ===== TAB 3: PREDICTION VISUAL HIERARCHY ===== */

    /* Prediction step colors - reuse eda-step base */
    .pred-step-1 { border-left-color: #3498db !important; }  /* Blue - Select Model */
    .pred-step-2 { border-left-color: #9b59b6 !important; }  /* Purple - Upload Data */
    .pred-step-3 { border-left-color: #2ecc71 !important; }  /* Green - Predictions */
    """

    with gr.Blocks(css=css, title="CANCapital EDA & Modeling Interface") as interface:
        # Header
        gr.HTML("""
        <div class="header">
            <h1>CANCapital EDA & Modeling Interface</h1>
            <p>Flexible analysis for any financial dataset • Basic EDA • ML Pipeline • Model Prediction</p>
        </div>
        """)

        # Tab 1: Basic EDA - Bold Visual Hierarchy Design
        with gr.Tab("Basic EDA Analysis"):
            gr.HTML('<div class="tab-header">Exploratory Data Analysis</div>')

            # ===== STEP 1: UPLOAD =====
            gr.HTML('''
            <div class="step-header">
                <span class="step-number step-number-1">1</span>
                <span class="step-title">Upload Your Data</span>
            </div>
            ''')

            with gr.Group(elem_classes=["eda-step", "eda-step-1"]):
                gr.HTML('<p style="color: #bdc3c7; margin: 0 0 15px 0;">Upload any CSV file for comprehensive statistical analysis.</p>')

                eda_file = gr.File(
                    label="CSV Dataset",
                    file_types=[".csv"]
                )

                file_status = gr.HTML(
                    value='<span class="status-badge status-pending">No file uploaded</span>'
                )

            # Connector
            gr.HTML('<div class="step-connector">↓</div>')

            # ===== STEP 2: ANALYZE =====
            gr.HTML('''
            <div class="step-header">
                <span class="step-number step-number-2">2</span>
                <span class="step-title">Run Analysis</span>
            </div>
            ''')

            with gr.Group(elem_classes=["eda-step", "eda-step-2"]):
                gr.HTML('<p style="color: #bdc3c7; margin: 0 0 15px 0;">Generates statistics, correlations, and 6 visualization plots.</p>')

                eda_run_btn = gr.Button(
                    "Run EDA Analysis",
                    variant="primary",
                    interactive=False,
                    elem_classes=["eda-btn-disabled"]
                )

                gr.HTML('<p style="color: #7f8c8d; font-size: 0.85em; margin-top: 10px;">⏱️ Analysis typically takes 10-60 seconds depending on dataset size.</p>')

            # Connector
            gr.HTML('<div class="step-connector">↓</div>')

            # ===== STEP 3: RESULTS =====
            gr.HTML('''
            <div class="step-header">
                <span class="step-number step-number-3">3</span>
                <span class="step-title">View Results</span>
            </div>
            ''')

            with gr.Group(elem_classes=["eda-step", "eda-step-3"]):
                eda_output = gr.Textbox(
                    label="Analysis Results",
                    lines=20,
                    placeholder="""📋 How to use this tab:

Step 1: Upload your CSV file above
Step 2: Click 'Run EDA Analysis' to start
Step 3: Results appear here • Plots saved to eda_plots/ folder

Supports any CSV format - financial data, research data, surveys, and more."""
                )

                eda_download = gr.File(label="Download Report (.txt)")

                # Plot location helper text
                gr.HTML('''
                <div class="eda-helper-text">
                    <strong style="color: #f39c12;">📊 Visualizations:</strong>
                    <span style="color: #bdc3c7;">
                        6 PNG plots are saved to <code>eda_plots/</code> folder
                        (distributions, box plots, correlations, scatter plots, pairplot)
                    </span>
                </div>
                ''')

            # ===== EVENT HANDLERS =====
            eda_file.change(
                fn=update_eda_button_state,
                inputs=[eda_file],
                outputs=[eda_run_btn, file_status]
            )

            eda_run_btn.click(
                fn=run_eda_analysis,
                inputs=[eda_file],
                outputs=[eda_output, eda_download]
            )

        # Tab 2: ML Pipeline - Bold Visual Hierarchy Design (4 Steps)
        with gr.Tab("Machine Learning Pipeline"):
            gr.HTML('<div class="tab-header">Full ML Pipeline with Model Training</div>')

            # Dataset compatibility - collapsible accordion (less alarming)
            with gr.Accordion("Dataset Requirements (click to expand)", open=False):
                gr.HTML("""
                <div style="padding: 10px; color: #bdc3c7;">
                    <p><strong>This pipeline is optimized for loan collection datasets.</strong></p>
                    <p><strong>Required:</strong> <code>funded_date</code> (for chronological data split)</p>
                    <p><strong>Optional but expected:</strong></p>
                    <ul style="margin: 5px 0;">
                        <li><code>collection_case_days</code>, <code>collection_case_close_date</code></li>
                        <li><code>last_payment_date</code>, <code>pg_date_of_birth</code></li>
                        <li><code>demand_letter_sent_date</code>, <code>collection_case_open_date</code></li>
                    </ul>
                    <p style="color: #f39c12;"><strong>For other datasets:</strong> Use the Basic EDA tab instead.</p>
                </div>
                """)

            # State for tracking columns loaded
            columns_loaded_state = gr.State(False)

            # ===== STEP 1: UPLOAD =====
            gr.HTML('''
            <div class="step-header">
                <span class="step-number step-number-1">1</span>
                <span class="step-title">Upload Dataset</span>
            </div>
            ''')

            with gr.Group(elem_classes=["ml-step", "ml-step-1"]):
                gr.HTML('<p style="color: #bdc3c7; margin: 0 0 15px 0;">Upload a CSV file with loan collection data.</p>')

                ml_file = gr.File(
                    label="CSV Dataset (Loan Collection Format)",
                    file_types=[".csv"]
                )

                ml_file_status = gr.HTML(
                    value='<span class="status-badge status-pending">No file uploaded</span>'
                )

            # Connector
            gr.HTML('<div class="step-connector">↓</div>')

            # ===== STEP 2: CONFIGURE =====
            gr.HTML('''
            <div class="step-header">
                <span class="step-number step-number-2">2</span>
                <span class="step-title">Configure Pipeline</span>
            </div>
            ''')

            with gr.Group(elem_classes=["ml-step", "ml-step-2"]):
                # ===== Sub-step 2a: Validate Dataset =====
                gr.HTML('''
                <p style="color: #9b59b6; font-weight: 600; margin: 0 0 8px 0;">
                    2a. Validate Dataset
                </p>
                <p style="color: #95a5a6; font-size: 0.85em; margin: 0 0 10px 0;">
                    Checks compatibility and loads column names from your CSV.
                </p>
                ''')

                get_cols_btn = gr.Button("Validate & Load Columns", variant="primary")

                ml_columns_status = gr.HTML(
                    value='<span class="status-badge status-pending">Not validated yet</span>'
                )

                # Divider between sub-steps
                gr.HTML('<hr style="border: none; border-top: 1px solid #4a5568; margin: 15px 0;">')

                # ===== Sub-step 2b: Select Target =====
                gr.HTML('''
                <p style="color: #9b59b6; font-weight: 600; margin: 0 0 8px 0;">
                    2b. Select Target <span style="color: #95a5a6; font-weight: normal;">(Optional)</span>
                </p>
                ''')

                target_column = gr.Dropdown(
                    choices=[],
                    label="Target Column",
                    interactive=False  # Disabled until columns loaded
                )

                gr.HTML('''
                <p style="color: #7f8c8d; font-size: 0.85em; margin: 8px 0 0 0;">
                    ℹ️ Leave empty to auto-detect. Looks for <code>collection_case_days</code> or similar.
                </p>
                ''')

                # Advanced options - collapsed
                with gr.Accordion("Advanced Options", open=False):
                    output_directory = gr.Textbox(
                        value="ml_results",
                        label="Output Directory"
                    )

            # Connector
            gr.HTML('<div class="step-connector">↓</div>')

            # ===== STEP 3: TRAIN =====
            gr.HTML('''
            <div class="step-header">
                <span class="step-number step-number-3">3</span>
                <span class="step-title">Train Models</span>
            </div>
            ''')

            with gr.Group(elem_classes=["ml-step", "ml-step-3"]):
                gr.HTML('<p style="color: #bdc3c7; margin: 0 0 15px 0;">Train multiple ML models and select the best performer.</p>')

                ml_run_btn = gr.Button(
                    "Run ML Pipeline",
                    variant="primary",
                    interactive=False,
                    elem_classes=["eda-btn-disabled"]
                )

                ml_run_status = gr.HTML(
                    value='<span class="status-badge status-pending">Upload file first</span>'
                )

                gr.HTML('''
                <p class="ml-time-estimate">
                    ⏱️ Pipeline typically takes <strong>1-10 minutes</strong> depending on dataset size.<br>
                    Do not refresh the page during processing.
                </p>
                ''')

            # Connector
            gr.HTML('<div class="step-connector">↓</div>')

            # ===== STEP 4: RESULTS =====
            gr.HTML('''
            <div class="step-header">
                <span class="step-number step-number-4">4</span>
                <span class="step-title">Results & Next Steps</span>
            </div>
            ''')

            with gr.Group(elem_classes=["ml-step", "ml-step-4"]):
                ml_output = gr.Textbox(
                    label="Pipeline Results",
                    lines=20,
                    placeholder="""📋 How to use this tab:

Step 1: Upload your loan collection CSV file
Step 2: Click 'Validate & Load Columns' to check compatibility
Step 3: (Optional) Select a target column or leave empty to auto-detect
Step 4: Click 'Run ML Pipeline' to train models
Step 5: View results here • Model saved to trained_models/ folder

Note: This pipeline requires 'funded_date' column for time-based data splitting."""
                )

                ml_download = gr.File(label="Download Complete Results (.txt)")

                # Model location helper text
                gr.HTML('''
                <div class="ml-helper-text">
                    <strong style="color: #f39c12;">📁 Trained Model Location:</strong>
                    <span style="color: #bdc3c7;">
                        After training, your model is saved to <code>trained_models/model_YYYYMMDD_HHMMSS/</code>
                        including <code>best_model.pkl</code>, <code>preprocessor.pkl</code>, and <code>model_metadata.json</code>
                    </span>
                </div>
                ''')

                # Next steps guidance
                gr.HTML('''
                <div class="ml-next-steps">
                    <p><strong>✅ Next Step:</strong> After training completes, go to the <strong>"Model Prediction"</strong> tab to make predictions on new data.</p>
                </div>
                ''')

            # ===== EVENT HANDLERS =====

            # Helper function for column loading with state update
            def update_target_options_with_state(file_path):
                """Load columns and update state with rich feedback."""
                if file_path:
                    columns = get_csv_columns(file_path)
                    compatibility = check_ml_pipeline_compatibility(file_path)
                    col_count = len(columns) if columns else 0

                    # Check for auto-detectable target column
                    auto_target = None
                    target_hints = ['collection_case_days', 'target', 'label', 'y']
                    for col in columns or []:
                        col_lower = col.lower()
                        if any(hint in col_lower for hint in target_hints):
                            auto_target = col
                            break

                    if not compatibility["compatible"]:
                        missing = ', '.join(compatibility.get('missing_required', []))
                        status = f'<span class="status-badge status-columns-loaded">⚠️ {col_count} columns (missing: {missing})</span>'
                        return (
                            gr.update(
                                choices=columns,
                                value=None,
                                interactive=True
                            ) if columns else gr.update(choices=[], value=None, interactive=False),
                            True,  # columns_loaded = True (even if warning)
                            status
                        )

                    # Build success status with column count and auto-detect info
                    status_parts = [f"✓ {col_count} columns"]
                    if auto_target:
                        status_parts.append(f"Auto-detect: {auto_target}")
                    status = f'<span class="status-badge status-columns-loaded">{" • ".join(status_parts)}</span>'

                    return (
                        gr.update(choices=columns, value=None, interactive=True) if columns else gr.update(choices=[], value=None, interactive=False),
                        True,  # columns_loaded = True
                        status
                    )
                return (
                    gr.update(choices=[], value=None, interactive=False),
                    False,
                    '<span class="status-badge status-pending">Upload a file first</span>'
                )

            # File upload changes - reset state
            ml_file.change(
                fn=reset_ml_state,
                inputs=[ml_file],
                outputs=[ml_run_btn, ml_run_status, columns_loaded_state]
            )

            # Also update file status badge
            ml_file.change(
                fn=lambda f: '<span class="status-badge status-ready">✓ File uploaded</span>' if f else '<span class="status-badge status-pending">No file uploaded</span>',
                inputs=[ml_file],
                outputs=[ml_file_status]
            )

            # Load columns button
            get_cols_btn.click(
                fn=update_target_options_with_state,
                inputs=[ml_file],
                outputs=[target_column, columns_loaded_state, ml_columns_status]
            )

            # Update run button when columns state changes
            columns_loaded_state.change(
                fn=update_ml_button_state,
                inputs=[ml_file, columns_loaded_state],
                outputs=[ml_run_btn, ml_run_status]
            )

            # Run ML pipeline button
            ml_run_btn.click(
                fn=run_ml_pipeline,
                inputs=[ml_file, target_column, output_directory],
                outputs=[ml_output, ml_download]
            )

        # Tab 3: Model Prediction - Bold Visual Hierarchy Design (3 Steps)
        with gr.Tab("Model Prediction"):
            gr.HTML('<div class="tab-header">Use Saved Models for Predictions</div>')

            # State for tracking selections
            model_selected_state = gr.State(False)
            file_uploaded_state = gr.State(False)

            # ===== STEP 1: SELECT MODEL =====
            gr.HTML('''
            <div class="step-header">
                <span class="step-number step-number-1">1</span>
                <span class="step-title">Select Model</span>
            </div>
            ''')

            with gr.Group(elem_classes=["eda-step", "pred-step-1"]):
                gr.HTML('<p style="color: #bdc3c7; margin: 0 0 15px 0;">Choose a trained model from the ML Pipeline tab.</p>')

                model_directory = gr.Dropdown(
                    choices=[],
                    label="Trained Model",
                    info="Models are auto-loaded from trained_models/"
                )

                model_status = gr.HTML(
                    value='<span class="status-badge status-pending">No model selected</span>'
                )

                # Empty state guidance
                gr.HTML('''
                <p style="color: #7f8c8d; font-size: 0.85em; margin: 10px 0 0 0;">
                    ℹ️ No models available? Train one in the <strong>ML Pipeline</strong> tab first.
                </p>
                ''')

                # Advanced: Models directory (rarely needed)
                with gr.Accordion("Advanced Options", open=False):
                    models_dir = gr.Textbox(value="trained_models", label="Models Directory Path")
                    scan_btn = gr.Button("Rescan Directory", variant="secondary")
                    scan_output = gr.Textbox(label="Scan Results", lines=4)

            # Connector
            gr.HTML('<div class="step-connector">↓</div>')

            # ===== STEP 2: UPLOAD DATA =====
            gr.HTML('''
            <div class="step-header">
                <span class="step-number step-number-2">2</span>
                <span class="step-title">Upload New Data</span>
            </div>
            ''')

            with gr.Group(elem_classes=["eda-step", "pred-step-2"]):
                gr.HTML('<p style="color: #bdc3c7; margin: 0 0 15px 0;">Upload a CSV file with the same format as your training data.</p>')

                prediction_file = gr.File(
                    label="CSV Data for Predictions",
                    file_types=[".csv"]
                )

                file_status = gr.HTML(
                    value='<span class="status-badge status-pending">No file uploaded</span>'
                )

            # Connector
            gr.HTML('<div class="step-connector">↓</div>')

            # ===== STEP 3: GET PREDICTIONS =====
            gr.HTML('''
            <div class="step-header">
                <span class="step-number step-number-3">3</span>
                <span class="step-title">Get Predictions</span>
            </div>
            ''')

            with gr.Group(elem_classes=["eda-step", "pred-step-3"]):
                predict_btn = gr.Button(
                    "Make Predictions",
                    variant="primary",
                    interactive=False,
                    elem_classes=["eda-btn-disabled"]
                )

                predict_status = gr.HTML(
                    value='<span class="status-badge status-pending">Select model and upload file first</span>'
                )

                prediction_output = gr.Textbox(
                    label="Prediction Results",
                    lines=15,
                    placeholder="""📋 How to use this tab:

Step 1: Select a trained model from the dropdown above
Step 2: Upload a CSV file with new data to predict
Step 3: Click 'Make Predictions' to generate results

Models are automatically loaded from trained_models/ folder.
No models? Train one in the ML Pipeline tab first."""
                )

                prediction_download = gr.File(label="Download Predictions (.csv)")

            # ===== EVENT HANDLERS =====

            # Load model folders for dropdown on page load
            def populate_model_folders():
                return gr.update(choices=get_trained_model_folders())

            interface.load(fn=populate_model_folders, outputs=[model_directory])

            # Model dropdown change - update state and status
            model_directory.change(
                fn=update_model_selection_state,
                inputs=[model_directory],
                outputs=[model_selected_state, model_status]
            )

            # File upload change - update state and status
            prediction_file.change(
                fn=update_prediction_file_state,
                inputs=[prediction_file],
                outputs=[file_uploaded_state, file_status]
            )

            # Update predict button when model state changes
            model_selected_state.change(
                fn=update_prediction_button_state,
                inputs=[model_selected_state, file_uploaded_state],
                outputs=[predict_btn, predict_status]
            )

            # Update predict button when file state changes
            file_uploaded_state.change(
                fn=update_prediction_button_state,
                inputs=[model_selected_state, file_uploaded_state],
                outputs=[predict_btn, predict_status]
            )

            # Connect scan button (in Advanced accordion)
            scan_btn.click(
                fn=scan_for_models, inputs=[models_dir], outputs=[scan_output]
            )

            # Connect prediction button
            predict_btn.click(
                fn=run_model_prediction,
                inputs=[model_directory, prediction_file],
                outputs=[prediction_output, prediction_download]
            )

        # Tab 4: Help & Instructions
        with gr.Tab("Help & Instructions"):
            gr.HTML("""
            <h3>How to Use This Interface</h3>

            <div style="background-color: #34495e; padding: 20px; border-radius: 5px;">

            <h4>Basic EDA Tab</h4>
            <ul>
                <li><strong>Purpose:</strong> Traditional exploratory data analysis</li>
                <li><strong>Input:</strong> Any CSV dataset</li>
                <li><strong>Output:</strong> Statistical summaries, correlations, data insights</li>
            </ul>

            <h4>ML Pipeline Tab</h4>
            <ul>
                <li><strong>Purpose:</strong> End-to-end machine learning pipeline</li>
                <li><strong>Features:</strong></li>
                <ul>
                    <li>Automatic data cleaning and preprocessing</li>
                    <li>Feature engineering with date calculations</li>
                    <li>Model training and evaluation</li>
                    <li>Permutation importance analysis</li>
                </ul>
                <li><strong>Compatible datasets:</strong> Loan collection, stock trading, crypto data, etc.</li>
            </ul>

            <h4>Model Prediction Tab</h4>
            <ul>
                <li><strong>Purpose:</strong> Make predictions using saved models</li>
                <li><strong>Required:</strong> Models trained with the ML Pipeline tab</li>
            </ul>

            <h4>Dataset Compatibility Guide</h4>
            <p><strong>Basic EDA:</strong> Works with ANY dataset format</p>
            <p><strong>ML Pipeline:</strong> Optimized for loan collection datasets with columns like 'funded_date', 'collection_case_days'</p>

            <h4>Quick Start</h4>
            <ol>
                <li><strong>For Any Dataset:</strong> Use Basic EDA tab</li>
                <li><strong>For Loan Collection Data:</strong> Use ML Pipeline tab</li>
                <li><strong>To Make Predictions:</strong> Train a model with ML Pipeline, then use Model Prediction tab</li>
            </ol>

            <h4>Supported File Types</h4>
            <ul>
                <li><strong>CSV Files:</strong> All tabs support CSV format</li>
            </ul>

            <h4>Troubleshooting</h4>
            <p>If ML Pipeline fails with compatibility errors, your dataset likely lacks loan collection specific columns. Use Basic EDA for universal analysis instead.</p>

            </div>
            """)

    return interface


def main():
    """Main entry point for the Gradio interface."""

    # Check that supporting files exist
    required_files = ["./src/eda.py", "./src/eda_and_model.py"]
    missing_files = []

    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)

    if missing_files:
        print("Please ensure these files are in the current directory:")
        for file in missing_files:
            print(f"  - {file}")
        return

    print("Starting CANCapital Clean Gradio Interface...")

    try:
        interface = create_clean_interface()

        print("Interface created successfully")
        print("Launching at: http://localhost:7860")

        # Launch interface
        interface.launch(
            server_name="0.0.0.0", server_port=7860, show_error=True, share=True
        )

    except Exception as e:
        print(f"Error launching interface: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
