# Tab 1: Basic EDA Analysis - Complete Interaction Map

**Generated:** 2025-11-28
**Location:** `src/gradio_interface.py` lines 605-643
**Backend Function:** `run_eda_analysis()` lines 28-79
**Core Module:** `src/eda.py` (311 lines)

---

## 1. ALL INTERACTIONS ON THIS TAB

### UI Components Inventory

| Component | Type | Purpose | Configuration | Lines |
|-----------|------|---------|---------------|-------|
| **Header** | `gr.HTML()` | Tab title | "Exploratory Data Analysis" | 606 |
| **Compatibility Notice** | `gr.HTML()` | Info banner | Blue notice about universal dataset compatibility | 609-620 |
| **eda_file** | `gr.File()` | CSV upload input | `file_types=[".csv"]` | 624-626 |
| **eda_run_btn** | `gr.Button()` | Trigger analysis | `variant="primary"` | 628 |
| **eda_output** | `gr.Textbox()` | Display results | `lines=20`, read-only | 630-634 |
| **eda_download** | `gr.File()` | Download results | `.txt` file output | 636 |

---

### 1.1 User Actions Available

#### Action 1: Upload CSV File
- **Component:** `eda_file` (File upload)
- **User Interaction:** Click "Upload CSV Dataset" → Browse → Select .csv file
- **File Restrictions:** Only `.csv` files accepted
- **No validation on upload** - File is accepted but not processed until "Run EDA Analysis" is clicked
- **Storage:** Temporary Gradio storage (auto-cleaned after session)

#### Action 2: Run EDA Analysis
- **Component:** `eda_run_btn` (Button)
- **User Interaction:** Click "Run EDA Analysis" button
- **Trigger Condition:** Button clickable regardless of file upload status
- **Processing:** Calls `run_eda_analysis(csv_file_path)` backend function
- **Execution Time:** Varies by dataset size (no timeout on this operation)
- **Visual Feedback:** Gradio shows loading spinner during processing

#### Action 3: View Results
- **Component:** `eda_output` (Textbox)
- **User Interaction:** Passive - results appear automatically after analysis
- **Display Limit:** Shows up to 8000 characters (truncated if longer)
- **Read-only:** User cannot edit content
- **Scrollable:** 20 lines visible, scrolls for more content

#### Action 4: Download Full Results
- **Component:** `eda_download` (File download)
- **User Interaction:** Click filename to download `.txt` file
- **File Format:** Plain text file with timestamp (`eda_results_YYYYMMDD_HHMMSS.txt`)
- **Content:** Complete untruncated results + metadata header
- **Storage:** Saved to project root directory (persists after session)

---

### 1.2 Inputs This Tab Can Use

#### Primary Input: CSV File

**Accepted Formats:**
- ✅ Any `.csv` file (universal compatibility)
- ✅ Mixed data types (numerical, categorical, dates, text)
- ✅ Any number of columns
- ✅ Any number of rows (no hard limits, but performance degrades with large datasets)

**Dataset Types Explicitly Supported:**
- Financial data (loan records, transactions, budgets)
- Sales records (products, customers, regions)
- Customer analytics (demographics, behavior, segments)
- Scientific research (experiments, observations, measurements)
- Survey responses (ratings, free text, multiple choice)
- Sensor data (time series, readings, events)
- **Any tabular data in CSV format**

**No Required Columns:**
- Unlike Tab 2 (ML Pipeline), this tab has **zero column requirements**
- No need for `funded_date` or any specific column names
- Works with completely arbitrary column structures

**Encoding Compatibility:**
- Defaults to pandas CSV reader (typically UTF-8)
- May have issues with non-standard encodings (no explicit encoding handling)

---

### 1.3 Dropdowns Available on This Screen

**NONE** - This tab has no dropdown components.

**Comparison to Other Tabs:**
- Tab 2 (ML Pipeline): Has target column dropdown (dynamically populated)
- Tab 3 (Model Prediction): Has model selection dropdown
- Tab 1 (EDA): **Intentionally simple** - just upload and run

---

## 2. GENERAL USE CASE FOR THIS TAB

### Primary Purpose

**First-time data exploration and comprehensive dataset understanding.**

This tab serves as the **entry point** for users who need to:
1. Understand what's in their dataset before analysis
2. Identify data quality issues (missing values, duplicates, outliers)
3. Discover relationships and patterns in the data
4. Assess dataset readiness for machine learning

---

### What Users Can Accomplish

#### ✅ Dataset Overview & Structure

**Users can learn:**
- How many rows and columns are in the dataset
- What the first 5 rows look like (preview)
- Data types of each column (numerical vs. categorical)
- Basic statistical summary (mean, std, min, max, quartiles)

**Output Location:**
- Text display: Dataset shape, first 5 rows, info, statistical summary
- `eda_output` textbox lines 41-54 in eda.py

---

#### ✅ Data Quality Assessment

**Users can identify:**
- **Missing values** - Which columns have missing data and how much
- **Duplicate rows** - How many duplicate entries exist
- **Outliers** - Via box plots for each numerical column
- **Data type distribution** - How many columns of each type

**Automated Handling:**
- Missing numerical values → Filled with median
- Missing categorical values → Filled with mode
- Duplicate rows → Automatically removed for analysis

**Output Location:**
- Text display: Missing values report, duplicate count, quality summary
- Visual: `eda_plots/box_plots.png` (outlier detection)

---

#### ✅ Distribution Analysis

**Users can visualize:**
- **Histogram + KDE** for every numerical column (distribution shape)
- **Count plots** for categorical columns (frequency distribution)
- **Box plots** for all numerical columns (outlier detection)

**Generated Plots:**
1. `eda_plots/distributions.png` - Histograms with kernel density estimates
2. `eda_plots/categorical_counts.png` - Top 10 values for each categorical column
3. `eda_plots/box_plots.png` - Box plots showing quartiles and outliers

**Use Case Example:**
- User uploads sales data → Sees revenue is right-skewed with outliers → Decides to apply log transformation

---

#### ✅ Correlation & Relationship Discovery

**Users can discover:**
- **Correlation matrix** - Heatmap showing all pairwise correlations
- **Highly correlated pairs** - Features with |correlation| > 0.7
- **Top predictive features** - Features most correlated with first numerical column (example target)
- **Scatter plots** - Visual relationships between top 5 correlated features

**Generated Plots:**
1. `eda_plots/correlation_matrix.png` - Full correlation heatmap with annotations
2. `eda_plots/scatter_plots.png` - Bivariate relationships with example target
3. `eda_plots/pairplot.png` - Pairwise relationships for first 5 numerical columns

**Categorical Encoding for Correlation:**
- Categorical columns are label-encoded temporarily
- Allows correlation analysis on mixed data types
- Original data remains unchanged

**Use Case Example:**
- User uploads customer data → Discovers "age" and "income" are highly correlated → Considers removing one to reduce multicollinearity

---

#### ✅ Multivariate Pattern Recognition

**Users can visualize:**
- **Pairplot** - Grid of scatter plots + histograms for first 5 numerical columns
- Shows relationships between multiple variables simultaneously
- Diagonal shows distribution, off-diagonal shows correlations

**Generated Plot:**
- `eda_plots/pairplot.png` - Comprehensive multivariate visualization

**Use Case Example:**
- User uploads sensor data → Sees clustering in pairplot → Identifies potential segments for classification

---

#### ✅ Predictive Feature Identification

**Users can identify:**
- Top 10 features correlated with first numerical column (example target)
- Correlation coefficient values for ranking feature importance
- Potential predictive relationships before modeling

**Methodology:**
- Uses first numerical column as "example target"
- Sorts all features by correlation with target
- Reports top predictors with correlation values

**Output Location:**
- Text display: "Top 10 features correlated with [target]"
- Visual: Scatter plots of top 5 correlated features

**Use Case Example:**
- User uploads loan data (first column: collection_days) → Sees loan_amount has 0.65 correlation → Prioritizes this feature for modeling

---

### Workflow Pattern

```
User uploads CSV
    ↓
Clicks "Run EDA Analysis"
    ↓
Backend validates file exists
    ↓
Copies file to temp location
    ↓
Calls src.eda.run_complete_eda()
    ↓
Analysis runs (no timeout):
  • Load CSV with pandas
  • Generate statistical summaries
  • Handle missing values (median/mode imputation)
  • Remove duplicates
  • Create 6 visualization plots
  • Calculate correlations
  • Identify high-correlation pairs
  • Analyze relationships with example target
  • Compile text results
    ↓
Returns (display_text, download_file)
    ↓
Display results in textbox (truncated if >8000 chars)
    ↓
Provide downloadable .txt file with complete results
    ↓
User reviews results and downloads plots from eda_plots/
```

---

## 3. DETAILED ANALYSIS OUTPUT BREAKDOWN

### Text Results Structure

**Section 1: Dataset Overview**
- Shape (rows × columns)
- First 5 rows preview
- DataFrame info (column names, non-null counts, dtypes)
- Statistical summary (describe() transposed)

**Section 2: Data Quality**
- Missing values per column
- Number of duplicate rows
- Columns with missing data count

**Section 3: Feature Analysis**
- Count of numerical columns
- Count of categorical columns
- Highly correlated pairs (threshold: 0.7)

**Section 4: Target Correlation Analysis**
- Top 10 features correlated with example target (first numerical column)
- Correlation coefficients for ranking

**Section 5: Summary**
- Dataset shape
- List of numerical columns
- List of categorical columns
- Missing values after automated handling
- Potential predictive features

**Section 6: Dataset Quality Assessment**
- Total rows (formatted with commas)
- Total columns
- Duplicate rows removed count
- Columns with missing data count
- Worst missing data percentage
- Data types distribution

---

### Visual Outputs (Saved to `eda_plots/`)

| File | Content | Purpose | Dimensions |
|------|---------|---------|------------|
| `distributions.png` | Histograms + KDE for all numerical columns | Distribution shape analysis | 18 × (6 × rows) |
| `box_plots.png` | Box plots for all numerical columns | Outlier detection | 18 × (6 × rows) |
| `categorical_counts.png` | Count plots for up to 12 categorical columns | Frequency analysis | 18 × (6 × rows) |
| `correlation_matrix.png` | Heatmap of all pairwise correlations | Relationship discovery | 12 × 8 |
| `scatter_plots.png` | Top 5 features vs. example target | Bivariate relationships | 18 × 4 |
| `pairplot.png` | Pairwise scatter + histograms (first 5 numerical cols) | Multivariate patterns | Auto-sized |

**Plot Quality:** All saved at 300 DPI with tight bounding boxes

---

## 4. COMPARISON TO OTHER TABS

### Tab 1 (EDA) vs. Tab 2 (ML Pipeline)

| Feature | Tab 1: Basic EDA | Tab 2: ML Pipeline |
|---------|------------------|-------------------|
| **Dataset Requirements** | None - any CSV works | Requires `funded_date` column |
| **User Complexity** | Simple - just upload & run | Complex - select target, configure output |
| **Processing Method** | Direct function call | Subprocess execution (10-min timeout) |
| **Outputs** | Text summary + 6 plots | Trained models + evaluation metrics + predictions |
| **Purpose** | Exploration & understanding | Model training & prediction |
| **Dropdowns** | None | Target column selection |
| **Compatibility** | Universal | Loan-specific (warns on incompatibility) |

---

### Tab 1 (EDA) vs. Tab 3 (Model Prediction)

| Feature | Tab 1: Basic EDA | Tab 3: Model Prediction |
|---------|------------------|------------------------|
| **Prerequisites** | None | Requires trained model |
| **Inputs** | 1 CSV file | 2 inputs (model directory + CSV) |
| **Outputs** | Statistical analysis | Predictions CSV |
| **Purpose** | Understanding data | Applying models |
| **Dropdowns** | None | Model selection |

---

## 5. BACKEND PROCESSING DETAILS

### Function Flow

```python
run_eda_analysis(csv_file_path)  # gradio_interface.py line 28
    ↓
Validate file exists
    ↓
Copy to temp file (temp_eda_data.csv)
    ↓
Import src.eda.run_complete_eda
    ↓
run_complete_eda(temp_file)  # eda.py line 19
    ↓
    • Load CSV with pandas
    • Create eda_plots/ directory
    • Generate statistics (shape, head, info, describe)
    • Identify missing values
    • Fill missing (median for numerical, mode for categorical)
    • Remove duplicates
    • Label-encode categorical for correlation
    • Create 6 visualization plots
    • Calculate correlations
    • Find high-correlation pairs (>0.7)
    • Analyze top 10 features vs. example target
    • Compile summary sections
    ↓
Return formatted text results
    ↓
Clean up temp file
    ↓
Create timestamped download file (eda_results_YYYYMMDD_HHMMSS.txt)
    ↓
Truncate display if >8000 characters
    ↓
Return (display_text, download_filename)
```

---

### Error Handling

**File Validation:**
```python
if not csv_file_path or not Path(csv_file_path).exists():
    return "Please upload a CSV file first!", None
```

**Exception Handling:**
- Wrapped in try/except block
- Returns error message to `eda_output` textbox
- No download file generated on error
- Error message format: `"Error running EDA: {exception_message}"`

**No Timeout:**
- Unlike ML pipeline, EDA has no subprocess timeout
- Long-running analysis (large datasets) will complete or crash
- User sees Gradio loading spinner until completion

---

## 6. LIMITATIONS & EDGE CASES

### Current Limitations

1. **No categorical limit handling:**
   - Plots only first 12 categorical columns
   - High-cardinality columns (e.g., 1000+ unique values) may create unreadable plots

2. **Large dataset performance:**
   - No row/column limits enforced
   - Memory-intensive operations (correlation matrix, pairplot) may crash on huge datasets
   - No chunking or sampling strategy

3. **Display truncation:**
   - Textbox shows only first 8000 characters
   - Users must download full results to see complete output

4. **No encoding handling:**
   - Assumes UTF-8 encoding
   - Non-standard encodings may cause CSV read errors

5. **Hardcoded example target:**
   - Uses first numerical column as "example target"
   - May not be the actual target variable user cares about
   - No way to specify custom target for correlation analysis

6. **Plot overwrites:**
   - Plots saved to same filenames in `eda_plots/`
   - Running EDA multiple times overwrites previous plots
   - No timestamp or dataset-specific naming

---

### Edge Cases

**Empty CSV:**
- Pandas will load but analysis will fail on empty dataframe
- Error message displayed

**CSV with no numerical columns:**
- Distribution plots skipped
- Correlation analysis skipped
- Still generates categorical plots and summary

**CSV with no categorical columns:**
- Categorical plots skipped
- Still generates numerical analysis and correlations

**Single column CSV:**
- Correlation matrix will be 1×1
- Pairplot skipped (requires 2+ columns)
- Basic statistics still generated

**CSV with all missing values:**
- Imputation may fail if mode/median cannot be calculated
- Likely results in error

---

## 7. USER DECISION POINTS

### What Users Learn That Informs Next Steps

After running EDA analysis, users can decide:

**Decision 1: Is the dataset ready for ML?**
- ✅ If missing values are <20%, quality is good → Proceed to Tab 2 (ML Pipeline)
- ❌ If missing values are >50%, quality is poor → Clean data externally first

**Decision 2: Which column should be the target?**
- Review correlation analysis to identify predictive relationships
- Use highly correlated features as potential targets
- Informed target selection for Tab 2

**Decision 3: Do I need feature engineering?**
- If high-cardinality categorical columns exist → May need encoding
- If skewed distributions detected → May need transformations
- If correlated pairs found → May need to remove redundant features

**Decision 4: Is this dataset compatible with ML Pipeline?**
- Check if `funded_date` column exists
- Verify loan-specific columns present
- Decide if Tab 2 is appropriate or if custom ML needed

**Decision 5: What features should I focus on?**
- Top 10 correlated features identify high-value predictors
- Highly correlated pairs suggest redundancy
- Distribution plots reveal data quality issues

---

## 8. INTEGRATION WITH OTHER TABS

### Workflow Progression

**Typical User Journey:**
```
Tab 1: Basic EDA Analysis
  ↓ (User learns dataset structure and quality)
Tab 2: Machine Learning Pipeline
  ↓ (User trains models using insights from EDA)
Tab 3: Model Prediction
  ↓ (User applies trained models to new data)
```

**Information Flow:**
- Tab 1 provides **insights** (correlation, distributions, quality)
- Tab 2 uses **same CSV format** but requires specific columns
- Tab 3 uses **models trained** in Tab 2 on **new data with same structure**

**No Automated Handoff:**
- Users must manually re-upload CSV to Tab 2
- No shared state between tabs
- Each tab is independent

---

## 9. SUCCESS CRITERIA

### User Successfully Completes EDA When:

✅ **Upload succeeds** - CSV file accepted without errors
✅ **Analysis completes** - "EDA Analysis Complete!" message appears
✅ **Results display** - Text summary visible in `eda_output` textbox
✅ **Download available** - `.txt` file link appears in `eda_download` component
✅ **Plots generated** - 6 PNG files created in `eda_plots/` directory

### Outputs Confirm Success:

- `eda_output` shows "EDA Analysis Complete!" followed by results
- `eda_download` shows filename `eda_results_YYYYMMDD_HHMMSS.txt`
- Directory `eda_plots/` contains 6 PNG files (may vary if dataset lacks numerical/categorical columns)

---

## 10. TECHNICAL SPECIFICATIONS

### Dependencies Required

```python
import pandas as pd           # CSV loading and manipulation
import numpy as np            # Numerical operations
import matplotlib.pyplot as plt  # Plotting backend
import seaborn as sns         # Statistical visualizations
from sklearn.preprocessing import LabelEncoder  # Categorical encoding
```

### File System Interactions

**Read Locations:**
- User-uploaded CSV (Gradio temp storage)

**Write Locations:**
- `temp_eda_data.csv` (temporary, deleted after processing)
- `eda_results_YYYYMMDD_HHMMSS.txt` (project root, persists)
- `eda_plots/*.png` (6 plot files, overwrites on re-run)

**Permissions Required:**
- Read: Uploaded CSV file path
- Write: Project root directory (for .txt results)
- Write: `eda_plots/` directory (create if doesn't exist)

---

## 11. SUMMARY

### Tab 1: Basic EDA Analysis at a Glance

**Purpose:** Comprehensive exploratory data analysis for any CSV dataset

**Interactions:** 4 total
1. Upload CSV file
2. Click "Run EDA Analysis" button
3. View results in textbox (passive)
4. Download full results .txt file

**Inputs:** 1 - Any CSV file (no column requirements)

**Dropdowns:** None

**Outputs:**
- Text summary (dataset stats, quality assessment, correlations)
- 6 visualization plots (distributions, box plots, categorical counts, correlation matrix, scatter plots, pairplot)
- Downloadable .txt file with complete results

**Use Case:** First-time data exploration before ML modeling - understand structure, quality, distributions, and relationships in any tabular dataset.

**Key Strength:** Universal compatibility - works with ANY CSV format without column requirements.

**Intended Users:**
- Data scientists exploring new datasets
- Analysts assessing data quality
- ML engineers identifying predictive features
- Researchers understanding experimental data
- Anyone needing quick statistical overview of CSV data

---

**Document Version:** 1.0
**Last Updated:** 2025-11-28
**Maintained By:** Development Team
