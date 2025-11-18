# 🚀 CANCapital Gradio Interface

A web-based interface for running Exploratory Data Analysis and Machine Learning pipelines on CSV datasets.

## 🎯 Features

### 📊 Basic EDA
- **Dataset Overview**: Shape, first rows, info, statistical summary
- **Missing Value Analysis**: Identification and handling strategies  
- **Univariate Analysis**: Distribution plots for numerical variables
- **Outlier Detection**: Box plots for identifying outliers
- **Categorical Analysis**: Count plots for categorical variables
- **Correlation Analysis**: Heatmap and highly correlated feature pairs

### 🤖 Full ML Pipeline  
- **Data Cleaning**: Automatic handling of dates, money columns, percentages
- **Feature Engineering**: Borrower age, time-based features, missing flags
- **Model Training**: Linear Regression & Gradient Boosting Regressor
- **Model Evaluation**: MAE, RMSLE, R² metrics with performance comparison
- **Interpretability**: Permutation importance plots and Partial Dependence (PDP/ICE) analysis
- **Predictions**: Downloadable prediction files for both models

## 🛠️ Installation & Setup

All dependencies are managed through `pixi`:

```bash
# Start the pixi environment (if not already activated)
source ~/.pixi/bin/pixi

# All dependencies are installed via pixi.toml
# No additional pip install needed!
```

## 🚀 Running the Interface

### Option 1: Direct Launch
```bash
pixi run python src/gradio_interface.py
```

### Option 2: Using Launcher Script  
```bash
python src/run_interface.py
```

## 🌐 Usage

1. **Open Browser**: Navigate to `http://localhost:7860`
2. **Upload CSV**: Click "Upload CSV Dataset" and select your file
3. **Choose Analysis Type**:
   - Select "Basic EDA" for quick exploratory analysis
   - Select "Full Pipeline (ML)" for complete ML pipeline
4. **Configure Settings** (for ML Pipeline):
   - Choose target column from dropdown
   - Optionally change output directory name
5. **Run Analysis**: Click "🚀 Run Analysis" button
6. **View Results**: Check console output and generated files

## 📁 Output Structure

### Basic EDA
- Console output with statistical summaries and analysis results
- Generated plots displayed in real-time

### Full Pipeline  
Results saved to `results/` directory:
```
results/
├── linearreg_predictions.csv      # Linear regression predictions
├── gbr_predictions.csv           # Gradient boosting predictions  
├── perm_importance.png           # Permutation importance plot
├── pdp_ice.png                   # Partial dependence plots
└── permutation_importance_full.csv  # Full importance metrics
```

## 🔧 Technical Details

- **Framework**: Gradio 5.x for web interface
- **Environment**: Pixi-managed Python 3.11 environment  
- **Dependencies**: pandas, numpy, matplotlib, seaborn, scikit-learn, category-encoders
- **Script Integration**: Seamless integration with existing `eda.py` and `eda_and_model.py`

## 🧪 Testing

```bash
# Run setup test to verify everything works
pixi run python tests/test_setup.py

# This will:
# 1. Test all package imports
# 2. Create a sample dataset  
# 3. Verify interface startup
```

## 📋 Sample Dataset

A test dataset is generated at `tests/sample_dataset.csv` with:
- 1,000 loan records
- Realistic loan collection data structure