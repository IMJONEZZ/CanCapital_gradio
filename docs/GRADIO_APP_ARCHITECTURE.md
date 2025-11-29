# CanCapital Gradio Application - Architecture & Roadmap

**Generated:** 2025-11-28
**Purpose:** Complete architectural documentation for Claude Code and development reference

---

## Table of Contents

1. [Application Overview](#application-overview)
2. [Technology Stack](#technology-stack)
3. [File Structure](#file-structure)
4. [UI Architecture](#ui-architecture)
5. [Component Inventory](#component-inventory)
6. [Data Flow Patterns](#data-flow-patterns)
7. [Backend Integration](#backend-integration)
8. [Event System](#event-system)
9. [Custom Styling](#custom-styling)
10. [Development Roadmap](#development-roadmap)

---

## Application Overview

### What This Application Does

A **multi-function ML workflow platform** built with Gradio Blocks API that provides:

1. **Exploratory Data Analysis (EDA)** - Statistical analysis and visualization
2. **ML Pipeline Execution** - Full model training workflow
3. **Model Predictions** - Inference on new data using trained models
4. **Interactive Help** - Embedded documentation and troubleshooting

### Why Gradio Blocks (Not Interface)

- **Blocks API**: Advanced layout control, multi-tab workflows, custom CSS
- **Interface API**: Simple single-function wrappers (not suitable for this use case)
- **Decision**: Complex multi-step workflow requires Blocks architecture

### Entry Point

```bash
python src/gradio_interface.py
```

**Launch Configuration:**
```python
interface.launch(
    server_name="0.0.0.0",  # Network accessible
    server_port=7860,        # Default Gradio port
    show_error=True,         # Display errors in UI
    share=True               # Public share link
)
```

**Access URL:** `http://localhost:7860`

---

## Technology Stack

### Core Dependencies

```toml
[pypi-dependencies]
gradio = ">=5.49.1, <6"      # UI framework
pandas = ">=2.3.2, <3"        # Data manipulation
numpy = ">=2.3.2, <3"         # Numerical computing
scikit-learn = ">=1.7.1, <2"  # ML models
matplotlib = ">=3.10.6, <4"   # Plotting
seaborn = ">=0.13.2, <0.14"   # Statistical viz
category-encoders = ">=2.8.1, <3"  # Feature encoding
```

### Package Manager

**PIXI** - Handles both Python and JavaScript/TypeScript dependencies

```bash
pixi install          # Install dependencies
pixi run <command>    # Execute scripts
```

---

## File Structure

```
CanCapital_gradio/
├── src/
│   ├── gradio_interface.py      # 🎯 MAIN UI (848 lines)
│   ├── eda.py                    # EDA analysis backend
│   ├── eda_and_model.py          # ML pipeline backend
│   └── predict_with_saved_model.py  # Prediction backend
├── tests/
│   ├── test_setup.py             # Package import tests
│   └── test_column_extraction.py # CSV column extraction tests
├── eda_plots/                    # Generated visualization output
├── ml_results/                   # ML pipeline outputs
├── trained_models/               # Saved model artifacts
├── data/                         # Input datasets
├── docs/                         # Documentation (YOU ARE HERE)
├── pixi.toml                     # Dependency configuration
└── README.md                     # Project overview
```

### Critical File: `src/gradio_interface.py`

**Line Breakdown:**
- Lines 1-27: Imports
- Lines 28-79: EDA backend function (`run_eda_analysis`)
- Lines 82-272: ML pipeline backend function (`run_ml_pipeline`)
- Lines 275-312: Model scanning function (`scan_for_models`)
- Lines 315-384: Prediction backend function (`run_model_prediction`)
- Lines 387-451: Compatibility checking (`check_ml_pipeline_compatibility`)
- Lines 454-486: Utility functions (column extraction, model folder detection)
- Lines 493-593: **Custom CSS styling** (100 lines of dark mode theme)
- Lines 595-848: **Gradio Blocks UI definition** (main interface)

---

## UI Architecture

### High-Level Structure

```
gr.Blocks(css=css, title="CANCapital EDA & Modeling Interface")
  │
  ├─ HTML Header (App title + description)
  │
  ├─ Tab 1: "Basic EDA Analysis"
  │   ├─ File upload (CSV)
  │   ├─ Run button
  │   ├─ Output textbox
  │   └─ Download results
  │
  ├─ Tab 2: "Machine Learning Pipeline"
  │   ├─ File upload (CSV)
  │   ├─ Output directory input
  │   ├─ Get columns button
  │   ├─ Target column dropdown
  │   ├─ Run pipeline button
  │   ├─ Output textbox
  │   └─ Download results
  │
  ├─ Tab 3: "Model Prediction"
  │   ├─ Models directory input
  │   ├─ Scan models button
  │   ├─ Scan output display
  │   ├─ New data file upload
  │   ├─ Model selection dropdown
  │   ├─ Predict button
  │   ├─ Prediction output
  │   └─ Download predictions
  │
  └─ Tab 4: "Help & Instructions"
      └─ HTML documentation
```

### Layout Pattern: Rows & Columns

```python
with gr.Tab("Example Tab"):
    with gr.Row():                    # Horizontal layout
        with gr.Column(scale=2):       # Left column (wider)
            # Input components
        with gr.Column(scale=1):       # Right column (narrower)
            # Output components
```

**Scale Parameter:** Controls relative width (2:1 ratio = left twice as wide)

---

## Component Inventory

### Tab 1: Basic EDA Analysis (Lines 605-643)

| Component | Type | Purpose | Config |
|-----------|------|---------|--------|
| `eda_file` | `gr.File()` | Upload CSV | `file_types=[".csv"]` |
| `eda_run_btn` | `gr.Button()` | Trigger analysis | `variant="primary"` |
| `eda_output` | `gr.Textbox()` | Display results | `lines=20, interactive=False` |
| `eda_download` | `gr.File()` | Download report | `label="Download Full Results"` |

**Event Handler:**
```python
eda_run_btn.click(
    fn=run_eda_analysis,           # Backend function
    inputs=[eda_file],              # File upload component
    outputs=[eda_output, eda_download]  # Results display + download
)
```

---

### Tab 2: Machine Learning Pipeline (Lines 646-737)

| Component | Type | Purpose | Config |
|-----------|------|---------|--------|
| `ml_file` | `gr.File()` | Upload CSV | `file_types=[".csv"]` |
| `output_directory` | `gr.Textbox()` | Output path | `value="ml_results"` |
| `get_cols_btn` | `gr.Button()` | Load columns | `variant="secondary"` |
| `target_column` | `gr.Dropdown()` | Select target | Dynamically populated |
| `ml_run_btn` | `gr.Button()` | Execute pipeline | `variant="primary"` |
| `ml_output` | `gr.Textbox()` | Display results | `lines=20` |
| `ml_download` | `gr.File()` | Download report | - |

**Event Handlers:**

1. **Dynamic Column Loading:**
```python
get_cols_btn.click(
    fn=update_target_options,      # Extracts CSV columns + checks compatibility
    inputs=[ml_file],
    outputs=[target_column]        # Updates dropdown choices
)
```

2. **Pipeline Execution:**
```python
ml_run_btn.click(
    fn=run_ml_pipeline,
    inputs=[ml_file, target_column, output_directory],
    outputs=[ml_output, ml_download]
)
```

**Special Feature: Compatibility Checking**

Function `check_ml_pipeline_compatibility()` (lines 387-451):
- ✅ Required: `funded_date` column
- ⚠️  Optional: Loan-specific columns (Funded_Amount, Defaults, etc.)
- 📊 Returns: Compatibility status + suggestions

---

### Tab 3: Model Prediction (Lines 740-792)

| Component | Type | Purpose | Config |
|-----------|------|---------|--------|
| `models_dir` | `gr.Textbox()` | Model path | `value="trained_models"` |
| `scan_btn` | `gr.Button()` | Find models | `variant="secondary"` |
| `scan_output` | `gr.Textbox()` | Show models | `lines=10` |
| `prediction_file` | `gr.File()` | Upload data | `file_types=[".csv"]` |
| `model_directory` | `gr.Dropdown()` | Select model | Auto-populated on load |
| `predict_btn` | `gr.Button()` | Run predictions | `variant="primary"` |
| `prediction_output` | `gr.Textbox()` | Display results | `lines=15` |
| `prediction_download` | `gr.File()` | Download CSV | - |

**Event Handlers:**

1. **On Page Load (Auto-populate models):**
```python
interface.load(
    fn=populate_model_folders,     # Scans trained_models/ directory
    outputs=[model_directory]       # Updates dropdown on startup
)
```

2. **Manual Model Scan:**
```python
scan_btn.click(
    fn=scan_for_models,
    inputs=[models_dir],
    outputs=[scan_output]           # Displays list in textbox
)
```

3. **Prediction Execution:**
```python
predict_btn.click(
    fn=run_model_prediction,
    inputs=[model_directory, prediction_file],
    outputs=[prediction_output, prediction_download]
)
```

---

### Tab 4: Help & Instructions (Lines 795-847)

**Single Component:** `gr.HTML()` with extensive documentation

**Content Includes:**
- Overview of each tab
- Dataset compatibility requirements
- Step-by-step usage instructions
- Troubleshooting common issues
- File format specifications

---

## Data Flow Patterns

### Pattern 1: Simple Click → Execute → Display

```
User Action (Button Click)
    ↓
Event Handler (.click)
    ↓
Python Backend Function
    ↓
Processing Logic
    ↓
Return Results (tuple)
    ↓
Update Output Components
```

**Example: EDA Analysis**

```python
# User clicks "Run Analysis"
eda_run_btn.click(
    fn=run_eda_analysis,                # Function in gradio_interface.py
    inputs=[eda_file],                   # Input: uploaded CSV path
    outputs=[eda_output, eda_download]   # Outputs: text + file
)

# Backend function returns tuple
def run_eda_analysis(csv_file_path):
    # ... processing ...
    return (display_text, download_file_path)
```

---

### Pattern 2: Dynamic Component Update

```
User Action
    ↓
Event Handler
    ↓
Extract Data from Input
    ↓
Return Updated Choices
    ↓
Gradio Auto-Updates Component
```

**Example: Populating Target Column Dropdown**

```python
# User clicks "Get Columns from Dataset"
get_cols_btn.click(
    fn=update_target_options,       # Reads CSV columns
    inputs=[ml_file],                # Input: uploaded file
    outputs=[target_column]          # Output: updated dropdown
)

# Backend function returns gr.Dropdown.update()
def update_target_options(file_path):
    columns = get_csv_columns(file_path)
    compatibility = check_ml_pipeline_compatibility(file_path)

    return gr.Dropdown.update(
        choices=columns,
        value=columns[0],
        info=compatibility['message']
    )
```

---

### Pattern 3: Subprocess Execution

```
Gradio Event Handler
    ↓
Python Wrapper Function
    ↓
subprocess.run([...], timeout=600)
    ↓
External Python Script Execution
    ↓
Core ML Module Processing
    ↓
Capture stdout/stderr
    ↓
Return Results to Gradio
```

**Example: ML Pipeline Execution**

```python
def run_ml_pipeline(csv_file_path, target_column, output_dir):
    # Run subprocess
    result = subprocess.run(
        [
            "python", "./src/eda_and_model.py",
            csv_file_path,
            "--target", target_column,
            "--output_dir", output_dir
        ],
        capture_output=True,
        text=True,
        timeout=600  # 10-minute timeout
    )

    # Parse output
    if result.returncode == 0:
        return (success_message, results_file_path)
    else:
        return (error_message, None)
```

**Why Subprocess?**
- Long-running operations don't block UI
- Isolates ML pipeline execution
- Allows separate argument parsing
- Easier to run independently from CLI

---

## Backend Integration

### Module Responsibilities

| Module | Function | Called By | Execution Mode |
|--------|----------|-----------|----------------|
| `eda.py` | `run_complete_eda()` | `run_eda_analysis()` | Direct import |
| `eda_and_model.py` | CLI script | `run_ml_pipeline()` | Subprocess |
| `predict_with_saved_model.py` | CLI script | `run_model_prediction()` | Subprocess |

---

### Backend Function: `run_eda_analysis()` (Lines 28-79)

**Purpose:** Execute exploratory data analysis and generate report

**Input:** `csv_file_path` (str) - Path to uploaded CSV

**Output:** Tuple of `(display_text, download_file_path)`

**Process:**
1. Validate file path exists
2. Call `src.eda.run_complete_eda(csv_file_path)`
3. Generate visualizations → `eda_plots/` directory
4. Create timestamped results file
5. Return summary + download link

**Error Handling:**
- File validation before processing
- Try/except wrapper with user-friendly error messages
- Returns error text in display component

---

### Backend Function: `run_ml_pipeline()` (Lines 82-272)

**Purpose:** Execute full ML training pipeline via subprocess

**Inputs:**
- `csv_file_path` (str)
- `target_column` (str)
- `output_dir` (str, default="ml_results")

**Output:** Tuple of `(display_text, download_file_path)`

**Process:**
1. Check dataset compatibility (`check_ml_pipeline_compatibility()`)
2. Display compatibility warnings if not loan dataset
3. Build subprocess command:
   ```bash
   python ./src/eda_and_model.py <csv_path> --target <target> --output_dir <dir>
   ```
4. Execute with 10-minute timeout
5. Parse stdout/stderr for results
6. Return formatted output + results file

**Subprocess Arguments:**
- `capture_output=True` - Capture stdout/stderr
- `text=True` - Return strings (not bytes)
- `timeout=600` - 10-minute maximum execution

**Special Feature:** Dataset compatibility checking before execution prevents wasted processing time on incompatible data.

---

### Backend Function: `run_model_prediction()` (Lines 315-384)

**Purpose:** Load trained model and generate predictions

**Inputs:**
- `model_directory` (str) - Path to saved model folder
- `new_data_file` (str) - Path to uploaded CSV

**Output:** Tuple of `(display_text, download_file_path)`

**Process:**
1. Validate inputs exist
2. Build subprocess command:
   ```bash
   python src/predict_with_saved_model.py <model_dir> <data_file>
   ```
3. Execute with 5-minute timeout
4. Parse prediction results
5. Return summary + predictions CSV

**Model Structure Expected:**
```
trained_models/
  └── model_YYYYMMDD_HHMMSS/
      ├── model.joblib         # Trained sklearn model
      ├── preprocessor.joblib  # Feature preprocessing pipeline
      └── metadata.json        # Model configuration
```

---

### Utility Function: `check_ml_pipeline_compatibility()` (Lines 387-451)

**Purpose:** Verify dataset has required columns for ML pipeline

**Required Columns:**
- `funded_date` (essential for time-based features)

**Optional Loan Columns:**
- `Funded_Amount`, `Defaults`, `States`, `Terms`, `Sector`, etc.

**Returns:** Dict with:
```python
{
    'is_compatible': bool,
    'has_funded_date': bool,
    'has_loan_columns': bool,
    'message': str  # User-facing message
}
```

**Used By:** `update_target_options()` to display warnings in dropdown info text

---

## Event System

### Event Types Used

| Event | Trigger | Use Case | Example |
|-------|---------|----------|---------|
| `.click()` | Button clicked | Primary actions | Run analysis, execute pipeline |
| `.load()` | Page loads | Initialization | Auto-populate model dropdown |
| `.change()` | Input changes | Live updates | **NOT CURRENTLY USED** |

---

### Event Pattern: Button Click

```python
button_component.click(
    fn=function_name,              # Function to execute
    inputs=[input_comp1, input_comp2],  # Input components (as list)
    outputs=[output_comp1, output_comp2]  # Output components (as list)
)
```

**Input Mapping:**
- Component values passed as function arguments in order
- `inputs=[file, dropdown]` → `def func(file_path, dropdown_value)`

**Output Mapping:**
- Function returns tuple matching outputs list
- `outputs=[text, file]` → `return ("result text", "file_path")`

---

### Event Pattern: Interface Load

```python
interface.load(
    fn=populate_model_folders,     # Runs on page load
    outputs=[model_directory]       # Updates dropdown
)
```

**Use Case:** Pre-populate components with existing data (e.g., scan for trained models on startup)

---

### State Management

**Current Implementation:** Stateless
- No `gr.State()` components used
- Each interaction is independent
- File uploads are temporary (Gradio handles cleanup)

**Potential Enhancement:** Add state for:
- Persisting selected dataset across tabs
- Tracking analysis history
- User session preferences

---

## Custom Styling

### CSS Theme: Dark Mode (Lines 493-593)

**Color Palette:**
```css
Primary Background: #2c3e50   /* Dark blue-gray */
Secondary Background: #34495e /* Lighter blue-gray */
Text Color: #ecf0f1           /* Light gray */
Accent Color: #3498db         /* Medium blue */
Success Color: #2ecc71        /* Green */
Error Color: #e74c3c          /* Red */
```

### Styled Components

**Container:**
```css
.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background-color: #2c3e50;
}
```

**Buttons:**
```css
.gradio-button.primary {
    background-color: #3498db !important;
    color: white !important;
}

.gradio-button.primary:hover {
    background-color: #2980b9 !important;
}
```

**Tabs:**
```css
.tab-nav button {
    background-color: #34495e !important;
    color: #ecf0f1 !important;
}

.tab-nav button.selected {
    background-color: #3498db !important;
}
```

**File Upload:**
```css
.file-upload {
    background-color: #34495e !important;
    border: 2px dashed #3498db !important;
}
```

### Custom Classes Applied

Components use `elem_classes` parameter:
```python
gr.Button("Run Analysis", variant="primary", elem_classes="run-button")
gr.Textbox(..., elem_classes="output-text")
```

---

## Development Roadmap

### Current Capabilities ✅

- [x] Dark-themed professional UI
- [x] CSV file upload and validation
- [x] Dynamic component population (dropdowns)
- [x] EDA analysis with visualizations
- [x] Full ML pipeline execution
- [x] Model training and persistence
- [x] Prediction on new data
- [x] Results download functionality
- [x] Dataset compatibility checking
- [x] Subprocess-based backend execution
- [x] Error handling and user feedback
- [x] Embedded help documentation

### Potential Enhancements 🔄

#### UI/UX Improvements
- [ ] Add progress bars for long-running operations
- [ ] Implement real-time log streaming during ML pipeline
- [ ] Add data preview table after file upload
- [ ] Create visualization gallery for EDA plots (instead of separate downloads)
- [ ] Add model comparison dashboard
- [ ] Implement dark/light mode toggle

#### Functionality Extensions
- [ ] Support multiple file formats (Excel, Parquet, JSON)
- [ ] Add data preprocessing tab (handling missing values, outliers)
- [ ] Implement hyperparameter tuning interface
- [ ] Add model explainability features (SHAP values, feature importance)
- [ ] Create batch prediction mode
- [ ] Add model performance monitoring over time

#### State & Session Management
- [ ] Implement `gr.State()` for session persistence
- [ ] Add analysis history tracking
- [ ] Create user workspace for multiple projects
- [ ] Add "Save Configuration" feature for reproducibility

#### Advanced ML Features
- [ ] Multi-model ensemble predictions
- [ ] A/B testing framework for model comparison
- [ ] Automated model retraining scheduler
- [ ] Feature engineering suggestions
- [ ] Data drift detection

#### Integration & Deployment
- [ ] Add database connectivity (PostgreSQL, MongoDB)
- [ ] API endpoint generation for predictions
- [ ] Docker containerization
- [ ] Authentication and user management
- [ ] Cloud storage integration (S3, GCS)

#### Testing & Quality
- [ ] Expand unit test coverage
- [ ] Add integration tests for UI workflows
- [ ] Implement end-to-end testing with Gradio test client
- [ ] Performance benchmarking
- [ ] Accessibility compliance (WCAG)

---

## Quick Reference: Key Functions

### UI Component Creation

```python
# File upload
file = gr.File(file_types=[".csv"], label="Upload CSV")

# Button
btn = gr.Button("Click Me", variant="primary")

# Textbox (output)
output = gr.Textbox(lines=20, interactive=False, label="Results")

# Dropdown (dynamic)
dropdown = gr.Dropdown(choices=[], label="Select Option")

# Download file
download = gr.File(label="Download Results")
```

### Event Binding

```python
# Button click
button.click(fn=my_function, inputs=[input1], outputs=[output1])

# Page load
interface.load(fn=init_function, outputs=[dropdown])
```

### Dynamic Updates

```python
# Update dropdown choices
return gr.Dropdown.update(choices=new_choices, value=default_value)

# Update textbox
return gr.Textbox.update(value="New text")
```

### Subprocess Execution

```python
result = subprocess.run(
    ["python", "script.py", arg1, arg2],
    capture_output=True,
    text=True,
    timeout=300
)
```

---

## File Paths Reference

### Input Paths
- User uploads → Temporary Gradio storage (auto-cleaned)
- CSV files → Validated before processing

### Output Paths
- EDA plots → `eda_plots/`
- EDA results → `eda_results_YYYYMMDD_HHMMSS.txt`
- ML pipeline results → `ml_results/`
- Trained models → `trained_models/model_YYYYMMDD_HHMMSS/`
- Predictions → `predictions_YYYYMMDD_HHMMSS.csv`

### Model Directory Structure
```
trained_models/
  └── model_20251120_105353/
      ├── model.joblib
      ├── preprocessor.joblib
      ├── feature_columns.txt
      └── training_metadata.json
```

---

## Troubleshooting Guide

### Common Issues

**1. "No columns found in dataset"**
- Cause: CSV parsing error or empty file
- Solution: Verify CSV has headers, check encoding (UTF-8)

**2. "ML pipeline requires 'funded_date' column"**
- Cause: Dataset missing required column
- Solution: Add `funded_date` column or use EDA-only workflow

**3. "Model directory not found"**
- Cause: No trained models exist or incorrect path
- Solution: Train a model first via ML Pipeline tab

**4. "Subprocess timeout"**
- Cause: Large dataset exceeds 10-minute limit
- Solution: Increase timeout in code or reduce dataset size

**5. "Prediction CSV format mismatch"**
- Cause: New data columns don't match training data
- Solution: Ensure same feature columns as training set

---

## Testing Commands

### Run Unit Tests
```bash
pixi run pytest tests/
```

### Test Specific Module
```bash
pixi run pytest tests/test_column_extraction.py -v
```

### Launch Application
```bash
pixi run python src/gradio_interface.py
```

---

## Architecture Decisions Log

### Why Blocks API over Interface API?
- **Need:** Multi-tab workflow, custom layouts, dynamic components
- **Decision:** Blocks API provides granular control
- **Trade-off:** More complex code vs. Interface simplicity

### Why Subprocess for ML Pipeline?
- **Need:** Long-running operations without UI blocking
- **Decision:** Subprocess execution with timeout
- **Alternative:** Threading (rejected - harder to manage)

### Why Dark Theme?
- **Need:** Professional appearance for data science app
- **Decision:** Custom CSS dark mode
- **Implementation:** 100-line CSS block with blue accent colors

### Why No State Management?
- **Need:** Simple stateless interactions sufficient for current use case
- **Decision:** No `gr.State()` components yet
- **Future:** Add state if multi-step workflows needed

---

## Contributing Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use descriptive variable names
- Add docstrings to all functions
- Comment complex logic blocks

### Adding New Tabs
1. Define components in new `gr.Tab()` block
2. Create backend function in `gradio_interface.py`
3. Add event handlers (`.click()`, `.load()`)
4. Update this documentation

### Modifying Existing Features
1. Test changes with sample datasets in `data/`
2. Update relevant docstrings
3. Run test suite before committing
4. Document breaking changes in README

---

## Resources

### Official Documentation
- Gradio Docs: https://www.gradio.app/docs/gradio/interface
- Gradio Blocks Guide: https://www.gradio.app/guides/blocks-and-event-listeners
- Scikit-learn: https://scikit-learn.org/stable/

### Project Files
- Main UI: `src/gradio_interface.py`
- EDA Module: `src/eda.py`
- ML Pipeline: `src/eda_and_model.py`
- Predictions: `src/predict_with_saved_model.py`

### Support
- Report issues in project repository
- Check Help tab in UI for usage instructions

---

**Document Version:** 1.0
**Last Updated:** 2025-11-28
**Maintained By:** Development Team
**Generated By:** Claude Code (Lucy)
