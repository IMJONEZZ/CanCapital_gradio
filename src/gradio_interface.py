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
        from eda import run_complete_eda

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
        cmd = [sys.executable, "eda_and_model.py", temp_file]

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
                    if "warning:" in result.stdout.lower() or "error:" in result.stdout.lower():
                        warnings.append(line.strip())
                if warnings:
                    result_text += f"\n\nAdditional Info:\n" + "\n".join(warnings[:3])

            # Create downloadable text file
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            download_filename = f"ml_pipeline_results_{timestamp}.txt"
            with open(download_filename, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("COMPLETE ML PIPELINE RESULTS\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
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

        # Build command
        cmd = [
            sys.executable,
            "predict_with_saved_model.py",
            f"./trained_models/{model_directory}",
            temp_file,
        ]

        # Run prediction
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes
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
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model Directory: {model_directory}\n")
                f.write("=" * 80 + "\n\n")
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n\nWarnings/Errors:\n")
                    f.write(result.stderr)
                f.write("\n\n" + "=" * 80 + "\n")
            
            return f"Prediction Complete!\n\nOutput:\n{result.stdout[:2000]}", download_filename
        else:
            return f"Prediction Failed:\n\nError: {result.stderr}", None

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
            if item.is_dir() and not item.name.startswith('.'):
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
        max-width: 1200px !important;
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
    """

    with gr.Blocks(css=css, title="CANCapital EDA & Modeling Interface") as interface:
        # Header
        gr.HTML("""
        <div class="header">
            <h1>CANCapital EDA & Modeling Interface</h1>
            <p>Flexible analysis for any financial dataset • Basic EDA • ML Pipeline • Model Prediction</p>
        </div>
        """)

        # Tab 1: Basic EDA
        with gr.Tab("Basic EDA Analysis"):
            gr.HTML('<div class="tab-header">Exploratory Data Analysis</div>')

            # Universal compatibility notice
            gr.HTML("""
            <div style="background-color: #34495e; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 4px solid #3498db;">
                <h4 style="margin-top: 0; color: #ecf0f1;">Universal Dataset Compatibility</h4>
                <p style="margin-bottom: 5px; color: #ecf0f1;"><strong>Basic EDA works with ANY dataset format:</strong></p>
                <ul style="margin: 5px 0; color: #ecf0f1;">
                    <li>Financial data, sales records, customer analytics</li>
                    <li>Scientific research, survey responses, sensor data</li>
                    <li>Any CSV with mixed column types (numbers, dates, text)</li>
                </ul>
                <p style="margin-bottom: 0; color: #ecf0f1;"><strong>Perfect for:</strong> First-time data exploration, understanding your dataset structure, identifying patterns and relationships</p>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    eda_file = gr.File(
                        label="Upload CSV Dataset (Any Format)", file_types=[".csv"]
                    )
                with gr.Column(scale=1):
                    eda_run_btn = gr.Button("Run EDA Analysis", variant="primary")

            eda_output = gr.Textbox(
                label="EDA Results",
                lines=20,
                placeholder="Upload a CSV file and click 'Run EDA Analysis' to see results here...",
            )

            eda_download = gr.File(label="Download Complete Results (.txt)")

            # Connect EDA button
            eda_run_btn.click(
                fn=run_eda_analysis,
                inputs=[eda_file],
                outputs=[eda_output, eda_download],
            )

        # Tab 2: ML Pipeline
        with gr.Tab("Machine Learning Pipeline"):
            gr.HTML(
                '<div class="tab-header">Full ML Pipeline with Model Training</div>'
            )

            # Dataset compatibility warning
            gr.HTML("""
            <div style="background-color: #e67e22; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 4px solid #d35400;">
                <h4 style="margin-top: 0; color: #ecf0f1;">Dataset Compatibility Notice</h4>
                <p style="margin-bottom: 0; color: #ecf0f1;"><strong>ML Pipeline works best with loan collection datasets</strong> containing these columns:</p>
                <ul style="margin: 10px 0; color: #ecf0f1;">
                    <li><code>funded_date</code> (required for chronological data split)</li>
                    <li><code>collection_case_days</code>, <code>collection_case_close_date</code></li>
                    <li><code>last_payment_date</code>, <code>pg_date_of_birth</code></li>
                    <li><code>demand_letter_sent_date</code>, <code>collection_case_open_date</code></li>
                </ul>
                <p style="margin-bottom: 0; color: #ecf0f1;"><strong>For other datasets:</strong> Use Basic EDA tab for analysis</p>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    ml_file = gr.File(
                        label="Upload CSV Dataset (Loan Collection Format)",
                        file_types=[".csv"],
                    )
                with gr.Column(scale=1):
                    output_directory = gr.Textbox(
                        value="ml_results", label="Output Directory"
                    )

            with gr.Row():
                target_column = gr.Dropdown(
                    choices=[], label="Target Column (Optional - Auto-detect if empty)"
                )

                # Button to populate target column dropdown
                get_cols_btn = gr.Button("Load Columns")

            ml_run_btn = gr.Button("Run ML Pipeline", variant="primary")

            ml_output = gr.Textbox(
                label="Pipeline Results",
                lines=20,
                placeholder="Configure options and click 'Run ML Pipeline'...",
            )

            ml_download = gr.File(label="Download Complete Results (.txt)")

            # Add compatibility checker info
            gr.HTML("""
            <div style="background-color: #27ae60; padding: 10px; border-radius: 5px; margin-top: 10px;">
                <p style="margin: 0; color: #ecf0f1;"><strong>Compatibility Check:</strong> The interface now automatically checks if your dataset is compatible with the ML pipeline requirements!</p>
            </div>
            """)

            # Load columns for target selection
            def update_target_options(file_path):
                if file_path:
                    columns = get_csv_columns(file_path)

                    # Add compatibility info
                    compatibility = check_ml_pipeline_compatibility(file_path)

                    if not compatibility["compatible"]:
                        return (
                            gr.update(
                                choices=columns,
                                value=None,
                                info=f"Warning: Dataset may not be compatible with ML Pipeline. Missing: {', '.join(compatibility.get('missing_required', []))}",
                            )
                            if columns
                            else gr.update(choices=[], value=None)
                        )

                    return (
                        gr.update(choices=columns, value=None)
                        if columns
                        else gr.update(choices=[], value=None)
                    )
                return gr.update(choices=[], value=None)

            get_cols_btn.click(
                fn=update_target_options, inputs=[ml_file], outputs=[target_column]
            )

            # Connect ML pipeline button
            ml_run_btn.click(
                fn=run_ml_pipeline,
                inputs=[ml_file, target_column, output_directory],
                outputs=[ml_output, ml_download],
            )

        # Tab 3: Model Prediction
        with gr.Tab("Model Prediction"):
            gr.HTML('<div class="tab-header">Use Saved Models for Predictions</div>')

            with gr.Row():
                with gr.Column(scale=1):
                    models_dir = gr.Textbox(
                        value="trained_models", label="Models Directory Path"
                    )
                with gr.Column(scale=1):
                    scan_btn = gr.Button("Scan for Models")

            scan_output = gr.Textbox(
                label="Available Models",
                lines=8,
                placeholder="Enter models directory and click 'Scan for Models'...",
            )

            with gr.Row():
                with gr.Column(scale=1):
                    prediction_file = gr.File(
                        label="Upload New Data (CSV)", file_types=[".csv"]
                    )
                with gr.Column(scale=1):
                    model_directory = gr.Dropdown(
                        choices=[], label="Selected Model Directory", 
                        info="Select from trained model folders"
                    )

            predict_btn = gr.Button("Make Predictions", variant="primary")

            prediction_output = gr.Textbox(label="Prediction Results", lines=15)
            
            prediction_download = gr.File(label="Download Prediction Results (.txt)")

            # Load model folders for dropdown
            def populate_model_folders():
                return gr.update(choices=get_trained_model_folders())

            # Populate model dropdown on page load
            interface.load(fn=populate_model_folders, outputs=[model_directory])

            # Connect scan button
            scan_btn.click(
                fn=scan_for_models, inputs=[models_dir], outputs=[scan_output]
            )

            # Connect prediction button
            predict_btn.click(
                fn=run_model_prediction,
                inputs=[model_directory, prediction_file],
                outputs=[prediction_output, prediction_download],
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
    required_files = ["eda.py", "eda_and_model.py"]
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
