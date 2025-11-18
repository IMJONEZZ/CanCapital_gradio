# Agent Guidelines for CANCapital

## Commands
- **Run EDA**: `python src/eda.py`
- **Run full pipeline**: `python src/eda_and_model.py data/dataset.csv --target collection_case_days --output_dir results`
- **Run Gradio Interface**: `pixi run python src/gradio_interface.py` or use launcher: `python src/run_interface.py`
- **Single test**: `python -m pytest` (if tests added later)
- **Lint code**: `ruff check .` or `flake8 *.py`
- **Format code**: `black *.py` or `ruff format .`

## Gradio Interface Usage
The Gradio interface provides a web-based UI for running EDA and ML pipeline analysis:

1. **Basic EDA**: Upload CSV → Select "Basic EDA" → Run Analysis
   - Generates visualizations, statistical summaries, correlation analysis
   
2. **Full Pipeline**: Upload CSV → Select "ML Pipeline" → Choose Target Column → Run Analysis
   - Complete data cleaning, feature engineering, model training with interpretability

Access the interface at: `http://localhost:7860`

## Code Style
- **Python version**: 3.11, use type hints for function parameters and returns (see src/eda_and_model.py:45-64)
- **Imports**: Standard library → third-party → local, group with blank lines (src/eda_and_model.py:11-41)
- **Formatting**: 88-char line length, 4-space indentation, use `black` formatting
- **Naming**: snake_case for functions/variables, PascalCase for classes (engineer_features:119, clean_dates:76)
- **Error handling**: Use try/except with specific exceptions, handle missing data gracefully (src/eda.py:34-43)
- **Documentation**: Add docstrings for functions, use descriptive variable names
- **Data handling**: Use pandas with low_memory=False for large CSVs (src/eda_and_model.py:70)
- **Performance**: Use vectorized operations, avoid loops over DataFrames where possible