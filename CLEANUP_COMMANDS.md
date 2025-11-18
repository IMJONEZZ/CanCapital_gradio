# CANCapital Cleanup Commands
## Essential File Setup for Gradio Interface

This file provides commands to clean up the CANCapital project and keep only the essential files needed for running the Gradio interface.

## 🎯 Quick Cleanup Commands

### Remove Demo and Test Files
```bash
# Remove test scripts
rm -f test_flexible_pipeline.py demo_*.py test_column_extraction.py test_setup.py

# Remove demo workflow files  
rm -f demo_complete_workflow.py demo_flexible_system.py

# Remove generated analysis outputs
rm -f core-loan-performance-metrics.csv features-related-to-length-of-case.csv missing-data-patterns.csv
rm -f r-greater-than-0.7\ relationships\ mostly\ identifiers\ vs\ dates.csv

# Remove generated visualizations
rm -f Figure_*.png
```

### Remove Old Interface Versions
```bash
# Keep only the clean working interface
rm -f gradio_interface.py  # Original with compatibility issues
rm -f run_interface.py     # Redundant runner script  
```

### Remove All Output Directories
```bash
# Remove all generated output directories
rm -rf results/
rm -rf auto_demo/ demo_eda_output/ demo_workflow/ explicit_demo/
rm -rf ml_pipeline_trading/ prediction_test/ quick_test/ test_simple/
rm -rf test_with_saving/ ml_pipeline_demo/ saved_model_demo/ 
rm -rf prediction_results/

# Remove any other test directories
find . -type d -name "test_*" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "demo_*" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "ml_*" -exec rm -rf {} + 2>/dev/null || true
```

### Remove Optional Dataset Files (Keep Only Main Test Data)
```bash
# Keep dataset.csv for testing, remove other datasets if present
rm -f sample_dataset.csv  # If exists as alternative

# Keep trading dataset files if you want them for testing flexibility
ls -1 *trading*.csv  # Check what exists
```

## ✅ Final Essential File Structure

After cleanup, your directory should contain:

### Core Application Files (MANDATORY)
```
✅ eda.py                           # Basic EDA analysis
✅ eda_and_model.py                 # Fixed ML pipeline with auto-detection  
✅ gradio_interface_clean.py        # Clean working Gradio interface
```

### Test Data (RECOMMENDED)
```
✅ dataset.csv                      # Original loan collection data for testing
```

### Documentation (KEEP)
```
✅ AGENTS.md                        # Project guidelines and usage instructions  
✅ README_GRADIO.md                 # Interface documentation (if exists)
```

### Configuration Files (KEEP)
```
✅ pixi.toml                        # Project configuration
✅ pixi.lock                        # Dependency lock file  
```

## 🚀 Verification Commands

After cleanup, verify everything works:

### Test Basic Functionality
```bash
# Test the fixed ML pipeline (should complete without errors)
python3 eda_and_model.py dataset.csv --output_dir final_test

# Verify it detects target automatically and runs successfully
```

### Test Gradio Interface  
```bash
# Start the clean interface
python3 gradio_interface_clean.py

# Or with pixi if preferred:
pixi run python gradio_interface_clean.py
```

## 📊 Expected Results

### Successful Test Output:
```
Loading dataset.csv …
Shape: (4217, 30)
📊 Dataset shape: (4217, 30)  
🔍 Auto-detecting column types...
🔧 Engineering features...
🎯 Auto-detecting target column...
✅ Using target column: 'collection_case_days'

=== Model performance ===
LinearReg   → MAE: 19.17 | RMSLE: 1.5513 | R²: -0.002
GradientBoost→ MAE: 9.97 | RMSLE: 0.7595 | R²: 0.576

🎉 Pipeline completed successfully!
```

## 🔄 Complete Cleanup Script

Run this entire section to perform full cleanup:

```bash
#!/bin/bash
echo "🧹 Starting CANCapital cleanup..."

# Remove all unnecessary files
rm -f test_flexible_pipeline.py demo_*.py *.csv (Figure_* .* ) test_column_extraction.py test_setup.py
rm -f gradio_interface.py run_interface.py sample_dataset.csv

# Remove all output directories  
find . -type d -name "test_*" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "demo_*" -exec rm -rf {} + 2>/dev/null || true  
find . -type d -name "ml_*" -exec rm -rf {} + 2>/dev/null || true
rm -rf results/ prediction_test/ quick_test/

echo "✅ Cleanup complete!"
echo ""
echo "Essential files remaining:"
ls -1 *.py *.csv *.md *.toml 2>/dev/null | head -10
```

## 🎯 What Each Essential File Does

| File | Purpose |
|------|---------|
| **eda.py** | Traditional exploratory data analysis with statistical summaries |
| **eda_and_model.py** | Enhanced ML pipeline with flexible dataset support and auto-detection |
| **gradio_interface_clean.py** | Clean Gradio web interface (3 tabs: EDA, ML Pipeline, Prediction) |
| **dataset.csv** | Test dataset for verifying functionality works |

## 💡 Usage After Cleanup

### Method 1: Command Line
```bash
# Basic EDA analysis  
python3 eda.py

# ML pipeline with auto-detection
python3 eda_and_model.py dataset.csv --output_dir results

# ML pipeline with model saving
python3 eda_and_model.py dataset.csv --output_dir results --save_model

# Predictions with saved model
python3 eda_and_model.py new_data.csv --load_model results/trained_models/dataset --output_dir predictions
```

### Method 2: Gradio Interface  
```bash
python3 gradio_interface_clean.py
# Then open http://localhost:7860 in browser

# Three tabs available:
# 📊 Basic EDA Analysis
# 🤖 Machine Learning Pipeline  
# 🔮 Model Prediction
```

## ⚠️ Important Notes

- **Backup First**: Consider backing up any custom files before cleanup
- **Keep Test Data**: Retain `dataset.csv` for verifying the system works  
- **Model Files**: If you have saved models, note their locations before cleanup
- **Custom Datasets**: Preserve any important custom datasets you want to keep

## 🔧 Troubleshooting After Cleanup

If something breaks after cleanup:

1. **Check file permissions**: `chmod +x *.py`
2. **Verify dependencies**: Ensure all packages in pixi.toml are installed
3. **Test step by step**: Verify each component separately:
   ```bash
   python3 -c "import pandas, numpy, sklearn; print('Dependencies OK')"
   python3 eda.py  # Test basic functionality
   ```

---

**Last Updated**: $(date)  
**Purpose**: Keep essential files only for production use