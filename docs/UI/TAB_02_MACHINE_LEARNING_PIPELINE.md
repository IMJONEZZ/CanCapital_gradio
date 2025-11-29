# Tab 2: Machine Learning Pipeline - Complete Interaction Map

**Generated:** 2025-11-28
**Location:** `src/gradio_interface.py` lines 815-907
**Backend Function:** `run_ml_pipeline()` lines 96-286
**Core Module:** `src/eda_and_model.py` (executed via subprocess)

---

## 1. INPUTS AND FUNCTIONALITY

### 1.1 UI Components Inventory

| Component | Type | Purpose | Configuration | Lines |
|-----------|------|---------|---------------|-------|
| **Header** | `gr.HTML()` | Tab title | "Full ML Pipeline with Model Training" | 817-818 |
| **Compatibility Warning** | `gr.HTML()` | Orange warning box | Lists required/optional columns | 821-834 |
| **ml_file** | `gr.File()` | CSV upload | `file_types=[".csv"]`, label mentions "Loan Collection Format" | 838-841 |
| **output_directory** | `gr.Textbox()` | Output path | Default: `"ml_results"` | 843-845 |
| **target_column** | `gr.Dropdown()` | Select prediction target | Dynamically populated from CSV | 848-850 |
| **get_cols_btn** | `gr.Button()` | Load columns into dropdown | "Load Columns" | 853 |
| **ml_run_btn** | `gr.Button()` | Execute pipeline | `variant="primary"` | 855 |
| **ml_output** | `gr.Textbox()` | Display results | 20 lines | 857-861 |
| **ml_download** | `gr.File()` | Download results | `.txt` file | 863 |
| **Compatibility Info** | `gr.HTML()` | Green info box | Explains auto-compatibility checking | 866-870 |

---

### 1.2 Input Requirements

#### Primary Input: CSV File

**Required Column:**
| Column | Purpose | Why Required |
|--------|---------|--------------|
| `funded_date` | Chronological train/test split | ML pipeline uses time-based splitting to prevent data leakage |

**Optional but Expected Columns (Loan Collection Specific):**
| Column | Purpose |
|--------|---------|
| `collection_case_days` | Common target variable |
| `collection_case_close_date` | Date feature engineering |
| `last_payment_date` | Payment behavior features |
| `pg_date_of_birth` | Borrower age calculation |
| `demand_letter_sent_date` | Collection timeline features |
| `collection_case_open_date` | Case duration features |

**Compatibility Check Logic (`check_ml_pipeline_compatibility()`):**
- If `funded_date` missing → **Incompatible** (hard fail)
- If 3+ optional columns missing → **Incompatible** (likely not loan data)
- Otherwise → **Compatible**

---

#### Secondary Input: Target Column

| Setting | Behavior |
|---------|----------|
| **Not specified** | Auto-detects based on column names (looks for `collection_case_days` or similar) |
| **Specified** | Uses the selected column as the prediction target |

---

#### Tertiary Input: Output Directory

| Default | Purpose |
|---------|---------|
| `ml_results` | Where pipeline outputs are saved (plots, metrics, logs) |

---

### 1.3 Functionality: What the ML Pipeline Does

The pipeline executes `src/eda_and_model.py` as a **subprocess** with a 10-minute timeout.

#### Pipeline Stages

```
Stage 1: Data Loading & Validation
    ↓
Stage 2: Feature Engineering
    • Date parsing (funded_date, close dates, etc.)
    • Time-based features (Year, Month, Day, DayOfWeek)
    • Cyclical encoding (month_sin/cos, day_sin/cos)
    • Borrower age calculation from pg_date_of_birth
    ↓
Stage 3: Data Cleaning
    • Handle missing values (median for numeric, mode for categorical)
    • Remove duplicates
    • Currency parsing (if financial columns detected)
    ↓
Stage 4: Categorical Encoding
    • High-cardinality → Target Encoding (category_encoders.TargetEncoder)
    • Low-cardinality → Label Encoding
    ↓
Stage 5: Train/Test Split
    • Chronological split using funded_date
    • Prevents data leakage from future → past
    ↓
Stage 6: Model Training
    • LinearRegression
    • GradientBoostingRegressor
    • RandomForestRegressor
    ↓
Stage 7: Model Evaluation
    • MAE (Mean Absolute Error)
    • RMSLE (Root Mean Squared Log Error)
    • R² (Coefficient of Determination)
    ↓
Stage 8: Interpretability
    • Permutation Importance
    • Partial Dependence Plots (PDP)
    • ICE Plots
    ↓
Stage 9: Model Persistence
    • Save best model to trained_models/model_YYYYMMDD_HHMMSS/
    • Save preprocessor pipeline
    • Save metadata JSON
```

---

### 1.4 Outputs Generated

#### Text Outputs

| File | Location | Content |
|------|----------|---------|
| `ml_pipeline_results_YYYYMMDD_HHMMSS.txt` | Project root | Full pipeline log + results summary |
| Download in UI | Via `ml_download` component | Same as above |

#### Model Artifacts

| File | Location | Content |
|------|----------|---------|
| `best_model.pkl` | `trained_models/model_YYYYMMDD_HHMMSS/` | Best performing sklearn model (joblib serialized) |
| `preprocessor.pkl` | Same directory | Feature transformation pipeline |
| `model_metadata.json` | Same directory | Target column, feature names, metrics, timestamp |
| `feature_columns.txt` | Same directory | List of feature columns used |

#### Visualization Outputs

| File | Location | Content |
|------|----------|---------|
| `perm_importance.png` | `ml_results/` or specified output_dir | Permutation importance bar chart |
| `pdp_ice.png` | Same | Partial Dependence + ICE plots |

---

### 1.5 Event Handlers

| Event | Trigger | Function | Inputs | Outputs |
|-------|---------|----------|--------|---------|
| `get_cols_btn.click()` | Click "Load Columns" | `update_target_options()` | `[ml_file]` | `[target_column]` |
| `ml_run_btn.click()` | Click "Run ML Pipeline" | `run_ml_pipeline()` | `[ml_file, target_column, output_directory]` | `[ml_output, ml_download]` |

---

### 1.6 Error Handling

| Error Condition | User Feedback |
|-----------------|---------------|
| No file uploaded | "Please upload a CSV file first!" |
| Missing `funded_date` | Detailed compatibility error with recommendation to use Basic EDA |
| Missing 3+ optional columns | Warning that dataset may not be loan data |
| Subprocess timeout (>10 min) | "ML Pipeline timed out (10 minutes limit)" |
| KeyError in pipeline | Suggests dataset incompatibility, recommends Basic EDA |
| Generic exception | Shows error message with debugging info |

---

## 2. EXPECTED USER EXPERIENCE

### 2.1 User Journey Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: Land on Tab                                                │
│  User sees: Orange compatibility warning + Upload + Options         │
└───────────────────────────────┬─────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2: Upload CSV                                                 │
│  User action: Click upload, browse, select loan collection CSV      │
│  Feedback: Filename appears in upload component                     │
└───────────────────────────────┬─────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3: Load Columns (OPTIONAL but recommended)                    │
│  User action: Click "Load Columns" button                           │
│  Feedback: Dropdown populates with column names                     │
│            If incompatible: Warning appears in dropdown info        │
└───────────────────────────────┬─────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 4: Select Target Column (OPTIONAL)                            │
│  User action: Select target from dropdown OR leave empty            │
│  Default: Auto-detection if left empty                              │
└───────────────────────────────┬─────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 5: Configure Output Directory (OPTIONAL)                      │
│  User action: Change from "ml_results" if desired                   │
│  Default: ml_results                                                │
└───────────────────────────────┬─────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 6: Run ML Pipeline                                            │
│  User action: Click "Run ML Pipeline" button                        │
│  Feedback: Loading spinner (up to 10 minutes)                       │
└───────────────────────────────┬─────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 7: View Results                                               │
│  Results appear in textbox showing:                                 │
│    • Processing summary                                             │
│    • Target column used                                             │
│    • Model performance metrics                                      │
│  Download link appears for full results                             │
└───────────────────────────────┬─────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 8: Access Trained Model                                       │
│  User navigates to: trained_models/model_YYYYMMDD_HHMMSS/           │
│  Contains: best_model.pkl, preprocessor.pkl, metadata               │
│  Ready for: Tab 3 (Model Prediction)                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 2.2 User Actions Available

| # | Action | Required? | Component | Expected Outcome |
|---|--------|-----------|-----------|------------------|
| 1 | Upload CSV | **Yes** | `ml_file` | File accepted, ready for processing |
| 2 | Load Columns | Recommended | `get_cols_btn` | Dropdown populates, compatibility checked |
| 3 | Select Target | Optional | `target_column` | Specific target set; or auto-detect |
| 4 | Set Output Dir | Optional | `output_directory` | Custom output location |
| 5 | Run Pipeline | **Yes** | `ml_run_btn` | 10-min processing, model trained |
| 6 | View Results | Passive | `ml_output` | Results appear automatically |
| 7 | Download | Optional | `ml_download` | Full results as `.txt` file |

---

### 2.3 Comparison to Tab 1 (Basic EDA)

| Aspect | Tab 1: Basic EDA | Tab 2: ML Pipeline |
|--------|------------------|-------------------|
| **Dataset Requirements** | None - any CSV | Requires `funded_date` + loan columns |
| **Primary Output** | Statistical summary + 6 plots | Trained ML model + metrics |
| **Execution** | Direct function call | Subprocess (10-min timeout) |
| **Complexity** | Simple: upload → run | Multi-step: upload → config → run |
| **Use Case** | Exploration, any data | Model training, loan data only |
| **Processing Time** | 10-60 seconds | 1-10 minutes |

---

### 2.4 Success Criteria

User successfully completes ML Pipeline when:

- [ ] CSV file uploaded (loan collection format)
- [ ] "Load Columns" shows no compatibility warnings
- [ ] "Run ML Pipeline" completes without timeout
- [ ] Results show "ML Pipeline Complete!" message
- [ ] Model metrics displayed (MAE, RMSLE, R²)
- [ ] Download link appears for full results
- [ ] `trained_models/model_YYYYMMDD_HHMMSS/` directory created with:
  - [ ] `best_model.pkl`
  - [ ] `preprocessor.pkl`
  - [ ] `model_metadata.json`

---

### 2.5 Common User Confusion Points

| Confusion Point | Why It Happens | Current Mitigation |
|-----------------|----------------|-------------------|
| "Why does it fail?" | Dataset missing `funded_date` | Orange warning box explains requirements |
| "What target should I choose?" | Many columns available | Auto-detection if left empty |
| "How long will this take?" | No progress indication | 10-minute timeout exists |
| "Where is my model?" | Not in UI, on filesystem | Must navigate to `trained_models/` |
| "Can I use any CSV?" | Assumes yes from Tab 1 | Compatibility check warns if incompatible |

---

### 2.6 Workflow: From ML Pipeline to Predictions

```
Tab 2: ML Pipeline
    │
    │  Trains model, saves to:
    │  trained_models/model_YYYYMMDD_HHMMSS/
    │
    ↓
Tab 3: Model Prediction
    │
    │  1. Scan for Models (finds trained model)
    │  2. Upload new data (same format)
    │  3. Select model from dropdown
    │  4. Make Predictions
    │
    ↓
Output: predictions_YYYYMMDD_HHMMSS.csv
```

---

## 3. TECHNICAL DETAILS

### 3.1 Backend Execution Pattern

```python
# Subprocess execution (run_ml_pipeline function)
cmd = [sys.executable, "./src/eda_and_model.py", temp_file]
if target_col:
    cmd.extend(["--target", target_col])
cmd.extend(["--output_dir", output_dir])

result = subprocess.run(
    cmd,
    capture_output=True,
    text=True,
    timeout=600  # 10 minutes
)
```

**Why Subprocess?**
- Long-running ML operations don't block UI
- Isolates pipeline execution
- Allows CLI usage of same module
- Enables timeout enforcement

---

### 3.2 Compatibility Check Details

```python
def check_ml_pipeline_compatibility(csv_file_path):
    required_columns = ["funded_date"]  # Hard requirement

    optional_but_used_columns = [
        "collection_case_days",
        "collection_case_close_date",
        "last_payment_date",
        "pg_date_of_birth",
        "demand_letter_sent_date",
        "collection_case_open_date",
    ]

    # Logic:
    # - Missing required → Incompatible
    # - Missing 3+ optional → Incompatible (likely not loan data)
    # - Otherwise → Compatible
```

---

### 3.3 Model Selection Logic

The pipeline trains 3 models and selects the best:

| Model | Type | Strengths |
|-------|------|-----------|
| LinearRegression | Baseline | Fast, interpretable, simple |
| GradientBoostingRegressor | Ensemble | Handles non-linear relationships |
| RandomForestRegressor | Ensemble | Robust to outliers, feature importance |

**Selection Criteria:** Lowest RMSLE on validation set

---

## 4. LIMITATIONS & EDGE CASES

### 4.1 Current Limitations

1. **Dataset Specific:** Only works with loan collection datasets containing `funded_date`
2. **No Progress Bar:** User sees only loading spinner for up to 10 minutes
3. **Fixed Model Types:** Cannot add custom models via UI
4. **No Hyperparameter Tuning:** Uses default sklearn parameters
5. **Single Target:** Only one target column per run

### 4.2 Edge Cases

| Scenario | Behavior |
|----------|----------|
| Empty CSV | Error: "Please upload a CSV file first!" |
| CSV with no `funded_date` | Compatibility error with detailed message |
| Very large dataset | May timeout after 10 minutes |
| Target column has all same value | Model training may fail with poor metrics |
| Non-numeric target | Treated as classification (may fail) |

---

## 5. FILE SYSTEM INTERACTIONS

### 5.1 Read Locations
- User-uploaded CSV (Gradio temp storage)
- `temp_ml_data.csv` (temporary copy)

### 5.2 Write Locations
- `ml_results/` (or custom `output_directory`)
  - `perm_importance.png`
  - `pdp_ice.png`
- `trained_models/model_YYYYMMDD_HHMMSS/`
  - `best_model.pkl`
  - `preprocessor.pkl`
  - `model_metadata.json`
  - `feature_columns.txt`
- `ml_pipeline_results_YYYYMMDD_HHMMSS.txt` (project root)

---

**Document Version:** 1.0
**Last Updated:** 2025-11-28
**Author:** Development Team
