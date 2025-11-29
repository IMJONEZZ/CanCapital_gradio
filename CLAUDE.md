# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CanCapital Gradio** is a web-based ML workflow platform for loan collection performance analysis. Built with Gradio Blocks API, it provides three core workflows: EDA analysis, ML pipeline execution, and model predictions.

**Key Domain:** Financial loan collection analysis (datasets require `funded_date` column for ML pipeline)

## Development Commands

```bash
# Environment
pixi install                                    # Install dependencies (Python 3.11)

# Run Application
pixi run python src/gradio_interface.py         # Launch at http://localhost:7860

# Tests (standalone scripts, no pytest)
pixi run python tests/test_setup.py             # Setup validation
pixi run python tests/test_column_extraction.py # CSV column extraction

# Core Modules (CLI)
pixi run python src/eda_and_model.py data/dataset.csv --target collection_case_days --output_dir ml_results
pixi run python src/predict_with_saved_model.py trained_models/model_20251120_105353 data/new_data.csv

# Dependencies
pixi add <package-name>                         # Add to pypi-dependencies
```

## Architecture

```
UI Layer (Gradio Blocks)
    ↓ Event Handlers (.click, .load)
Backend Wrappers (gradio_interface.py)
    ↓ Direct call OR Subprocess
Core ML Modules (eda.py, eda_and_model.py, predict_with_saved_model.py)
```

### Source Files

| File | Purpose | Execution |
|------|---------|-----------|
| `src/gradio_interface.py` | Main UI (4 tabs), backend wrappers, CSS theme (lines 493-593), UI components (lines 595-848) | Entry point |
| `src/eda.py` | EDA: stats, distributions, correlations → `eda_plots/` | Direct import |
| `src/eda_and_model.py` | Full ML pipeline (CLI): cleaning → feature engineering → 3 models → interpretability | Subprocess (600s timeout) |
| `src/predict_with_saved_model.py` | Model inference (CLI): loads joblib models, handles currency formatting | Subprocess (300s timeout) |
| `src/currency_utils.py` | Currency parsing utilities: `$15,000.00` → `15000.0`, auto-detection, imputation | Imported by prediction |

### Data Flow Patterns

| Workflow | Trigger | Backend | Execution |
|----------|---------|---------|-----------|
| **EDA Analysis** | `eda_run_btn.click()` | `run_eda_analysis()` → `eda.run_complete_eda()` | Direct call |
| **ML Pipeline** | `ml_run_btn.click()` | `run_ml_pipeline()` → subprocess `eda_and_model.py` | Subprocess, 600s timeout |
| **Predictions** | `predict_btn.click()` | `run_model_prediction()` → subprocess `predict_with_saved_model.py` | Subprocess, 300s timeout |
| **Dynamic Dropdown** | `get_cols_btn.click()` | `update_target_options()` → `check_ml_pipeline_compatibility()` | Direct call |

### Critical: Dataset Compatibility

**ML Pipeline requires `funded_date` column** for time-based feature engineering (year, month, cyclical encoding).

`check_ml_pipeline_compatibility()` (line 387) validates this before pipeline execution.

### State Management

**Stateless** - no `gr.State()`. Each tab interaction is independent; cannot share datasets between tabs.

## Output Directories

```
eda_plots/                    # EDA visualizations (overwrites on re-run)
ml_results/                   # ML pipeline outputs
trained_models/model_YYYYMMDD_HHMMSS/
  ├── best_model.pkl          # Joblib-serialized sklearn model
  ├── preprocessor.pkl        # Feature transformation pipeline
  ├── model_metadata.json     # Training config & metrics
  └── feature_columns.txt
results/                      # predictions_YYYYMMDD_HHMMSS.csv
```

## Key Implementation Details

### Backend Function Pattern
```python
def backend_function(inputs):
    try:
        if not Path(input_file).exists():
            return ("Error: File not found", None)
        result = process_data(input_file)
        return (success_message, output_file_path)  # Tuple matches Gradio outputs
    except Exception as e:
        return (f"Error: {str(e)}", None)
```

### Feature Engineering (eda_and_model.py)
- **Time features**: Year, Month, Day, DayOfWeek, cyclical encoding (month_sin/cos, day_sin/cos)
- **Categorical**: Target Encoding via `category_encoders.TargetEncoder`
- **Models**: LinearRegression, GradientBoostingRegressor, RandomForestRegressor

### Currency Handling (currency_utils.py)
Parses `$15,000.00`, `($1,234.56)` (negative), strips symbols/commas, handles €/£/¥. Auto-detects currency columns by pattern and column name hints.

## Common Modifications

### Adding a New Tab
1. Add `gr.Tab()` block in `gradio_interface.py` (after line 595)
2. Create backend wrapper function (follow existing patterns)
3. Bind events: `component.click(fn=backend_fn, inputs=[...], outputs=[...])`

### Adding ML Models
Modify `src/eda_and_model.py` around line 200+ (model definitions). Test standalone before UI integration.

### UI Theme
CSS in `gradio_interface.py` lines 493-593. Colors: Primary `#2c3e50`, Accent `#3498db`, Success `#2ecc71`, Error `#e74c3c`.

## Documentation

- **Full architecture**: `docs/GRADIO_APP_ARCHITECTURE.md`
- **Tab-specific docs**: `docs/UI/TAB_*.md`
- **Gradio docs**: https://www.gradio.app/docs/gradio/interface

## Known Limitations

1. No multi-file upload per tab
2. No session persistence between tabs (stateless)
3. Platform: osx-arm64 only (add platforms in pixi.toml)
4. `share=True` creates public URL - use `share=False` for local-only
