# CANCapital Gradio - Client Delivery

## Overview
A web-based interface for running Exploratory Data Analysis and Machine Learning pipelines on CSV datasets, designed for loan collection performance analysis.

## Quick Start

### Run the Interface
```bash
pixi run python src/gradio_interface.py
```
Then open browser to `http://localhost:7860`

## Features

### Basic EDA
- Dataset Overview & Statistical Summary
- Missing Value Analysis  
- Distribution Plots & Outlier Detection
- Correlation Analysis

### ML Pipeline
- Data Cleaning & Feature Engineering
- Linear Regression & Gradient Boosting Models
- Model Evaluation (MAE, RMSLE, R??)
- Interpretability & Predictions

## Project Structure
```
src/          # Core Python modules
data/         # Essential datasets (dataset.csv, sample_dataset.csv)
config/       # Configuration files (pixi.toml)
tests/        # Test utilities
trained_models/# Latest trained models
results/      # Analysis outputs (predictions, visualizations)
```

## Output Files
- `linearreg_predictions.csv` - Linear regression predictions
- `gbr_predictions.csv` - Gradient boosting predictions  
- `perm_importance.png` - Feature importance visualization
- `pdp_ice.png` - Partial dependence plots

## Technical Details
- Framework: Gradio 5.x web interface
- Environment: Python 3.11 with Pixi dependency management
- Dependencies: pandas, numpy, matplotlib, seaborn, scikit-learn

**Client Delivery Version - November 2025**
