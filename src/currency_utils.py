#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Currency Data Processing Utilities for Financial Datasets

This module provides robust handling of currency-formatted data that commonly
appears in financial datasets, preventing common preprocessing errors.
"""

import pandas as pd
import numpy as np
import re
from typing import Union, List, Tuple, Dict


def detect_currency_columns(df: pd.DataFrame) -> List[str]:
    """
    Automatically detect columns that contain currency-formatted data.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of column names likely containing currency data
    """
    currency_columns = []
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Sample a few non-null values to check for currency patterns
            sample_values = df[col].dropna().head(10).astype(str)
            
            currency_patterns = [
                r'^\$',              # Starts with $
                r'€|£|¥',           # Other currency symbols
                r'\$[\s]*\d',       # $ followed by space then digit  
                r'^\(.*\)$',        # Parentheses (negative amounts)
                r'\d{1,3}(,\d{3})*', # Numbers with commas
                r'\d+\.\d{2}',      # Decimal amounts
            ]
            
            for pattern in currency_patterns:
                if any(re.search(pattern, str(val)) for val in sample_values):
                    currency_columns.append(col)
                    break
                    
            # Also check column names for hints
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['amount', 'balance', 'payment', 'principal']):
                if df[col].dtype == 'object':
                    currency_columns.append(col)
    
    return list(set(currency_columns))


def clean_currency_value(value: Union[str, float, int]) -> float:
    """
    Clean a single currency value and convert to numeric.
    
    Args:
        value: Input value (string, float, or int)
        
    Returns:
        Cleaned numeric value
    """
    if pd.isna(value) or value == '' or str(value).lower() in ['nan', 'null', 'none']:
        return np.nan
    
    # Convert to string for processing
    str_value = str(value).strip()
    
    # Handle empty or whitespace-only strings
    if not str_value or str_value.isspace():
        return np.nan
    
    # Handle negative values in parentheses: ($1,234.56)
    is_negative = False
    if str_value.startswith('(') and str_value.endswith(')'):
        is_negative = True
        str_value = str_value[1:-1]
    
    # Remove currency symbols and whitespace
    currency_symbols = ['$', '€', '£', '¥']
    for symbol in currency_symbols:
        str_value = str_value.replace(symbol, '')
    
    # Remove spaces
    str_value = re.sub(r'\s+', '', str_value)
    
    # Remove thousands separators (commas)
    str_value = str_value.replace(',', '')
    
    try:
        numeric_value = float(str_value)
        
        # Apply negative sign if it was in parentheses
        if is_negative:
            numeric_value = -numeric_value
            
        return numeric_value
        
    except (ValueError, TypeError):
        # If we can't convert to float, return NaN
        return np.nan


def clean_currency_columns(df: pd.DataFrame, currency_columns: List[str] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Clean currency-formatted columns in the DataFrame.
    
    Args:
        df: Input DataFrame
        currency_columns: List of columns to clean (auto-detected if None)
        
    Returns:
        Tuple of (cleaned DataFrame, cleaning report dictionary)
    """
    df_cleaned = df.copy()
    
    # Auto-detect currency columns if not provided
    if currency_columns is None:
        currency_columns = detect_currency_columns(df)
    
    cleaning_report = {
        'columns_processed': [],
        'total_values_converted': 0,
        'conversion_errors': {},
    }
    
    for col in currency_columns:
        if col not in df_cleaned.columns:
            continue
            
        print(f"🔧 Cleaning currency column: {col}")
        
        original_dtype = df_cleaned[col].dtype
        non_null_count = df_cleaned[col].notna().sum()
        
        # Apply cleaning function
        cleaned_values = df_cleaned[col].apply(clean_currency_value)
        
        # Count successful conversions
        original_values = df_cleaned[col].fillna('').astype(str)
        valid_conversions = 0
        
        for orig, cleaned in zip(original_values, cleaned_values):
            if pd.notna(cleaned) and str(orig).strip() not in ['', 'nan', 'NaN']:
                valid_conversions += 1
        
        # Update column with cleaned values
        df_cleaned[col] = cleaned_values
        
        # Report cleaning results
        cleaning_report['columns_processed'].append({
            'column_name': col,
            'original_dtype': str(original_dtype),
            'new_dtype': str(df_cleaned[col].dtype),
            'non_null_values': non_null_count,
            'values_converted': valid_conversions,
        })
        
        cleaning_report['total_values_converted'] += valid_conversions
        
        print(f"   ✅ Converted {valid_conversions}/{non_null_count} values in '{col}'")
    
    return df_cleaned, cleaning_report


def smart_impute_numeric_columns(df: pd.DataFrame, exclude_columns: List[str] = None) -> pd.DataFrame:
    """
    Intelligently impute missing values in numeric columns.
    
    Args:
        df: Input DataFrame
        exclude_columns: Columns to skip imputation for
        
    Returns:
        DataFrame with imputed values
    """
    df_imputed = df.copy()
    
    if exclude_columns is None:
        exclude_columns = []
    
    # Get numeric columns (including cleaned currency columns)
    numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        if col in exclude_columns or df_imputed[col].isnull().sum() == 0:
            continue
            
        missing_count = df_imputed[col].isnull().sum()
        
        # Determine appropriate imputation strategy
        if 'age' in col.lower():
            # For age columns, use median of non-zero values or default to 35
            non_zero_values = df_imputed[col][df_imputed[col] > 0]
            if len(non_zero_values) > 0:
                impute_value = non_zero_values.median()
            else:
                impute_value = 35.0
                
        elif 'days' in col.lower():
            # For days columns, use median or default to 30
            non_zero_values = df_imputed[col][df_imputed[col] > 0]
            if len(non_zero_values) > 0:
                impute_value = non_zero_values.median()
            else:
                impute_value = 30.0
                
        elif 'percent' in col.lower() or '%' in col:
            # For percentage columns, use median
            impute_value = df_imputed[col].median()
            
        elif 'amount' in col.lower() or 'balance' in col.lower():
            # For financial amounts, use median
            impute_value = df_imputed[col].median()
            
        else:
            # Default to median for other numeric columns
            impute_value = df_imputed[col].median()
        
        # Handle NaN from median calculation
        if pd.isna(impute_value):
            impute_value = 0.0
            
        # Apply imputation
        df_imputed[col].fillna(impute_value, inplace=True)
        
        print(f"📊 Imputed {missing_count} missing values in '{col}' with value: {impute_value:.2f}")
    
    return df_imputed


def validate_data_for_modeling(df: pd.DataFrame, model_columns: List[str] = None) -> Dict:
    """
    Validate that data is properly formatted for machine learning.
    
    Args:
        df: Input DataFrame
        model_columns: Expected columns for the model
        
    Returns:
        Validation report dictionary
    """
    validation_report = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'info': {},
    }
    
    # Check data types
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Identify ID columns that should be excluded
    id_columns = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['id', 'loan_id', 'contract_number']):
            if df[col].dtype == 'object' or col.lower().endswith('_id'):
                id_columns.append(col)
    
    if non_numeric_cols:
        remaining_non_numeric = [col for col in non_numeric_cols if col not in id_columns]
        if remaining_non_numeric:
            validation_report['errors'].append(
                f"Non-numeric columns found (excluding IDs): {remaining_non_numeric}"
            )
            validation_report['is_valid'] = False
    
    # Check for missing values in required columns
    if model_columns:
        missing_model_cols = [col for col in model_columns if col not in df.columns]
        if missing_model_cols:
            validation_report['errors'].append(
                f"Required model columns missing: {missing_model_cols}"
            )
            validation_report['is_valid'] = False
    
    # Check for extreme outliers
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].notna().sum() > 0:
            q99 = df[col].quantile(0.99)
            extreme_count = (df[col] > q99 * 10).sum()
            
            if extreme_count > len(df) * 0.05:  # More than 5% extreme values
                validation_report['warnings'].append(
                    f"Column '{col}' has {extreme_count} extreme outliers (>10x 99th percentile)"
                )
    
    validation_report['info'] = {
        'total_columns': len(df.columns),
        'numeric_columns': len(numeric_cols),
        'categorical_columns': len(non_numeric_cols) - len(id_columns),
        'id_columns_identified': id_columns,
        'rows_with_missing_values': df.isnull().any(axis=1).sum(),
    }
    
    return validation_report


def enhanced_preprocess_for_prediction(df: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Enhanced preprocessing pipeline for financial datasets.
    
    Args:
        df: Input DataFrame
        target_column: Target column to exclude from processing
        
    Returns:
        Tuple of (processed DataFrame, preprocessing report)
    """
    print("🚀 Starting enhanced data preprocessing...")
    
    # Initialize report
    preprocessor_report = {
        'steps_completed': [],
        'currency_columns_detected': [],
        'data_cleaning_summary': {},
        'validation_results': {},
    }
    
    # Step 1: Detect and clean currency columns
    print("\n🔍 Step 1: Detecting currency-formatted columns...")
    currency_columns = detect_currency_columns(df)
    preprocessor_report['currency_columns_detected'] = currency_columns
    
    if currency_columns:
        print(f"   Found {len(currency_columns)} currency columns: {currency_columns}")
        df_cleaned, cleaning_report = clean_currency_columns(df, currency_columns)
        
        preprocessor_report['data_cleaning_summary'] = cleaning_report
        print(f"   ✅ Currency cleaning completed")
        
    else:
        df_cleaned = df.copy()
        print("   ℹ️  No currency columns detected")
    
    preprocessor_report['steps_completed'].append('currency_detection_and_cleaning')
    
    # Step 2: Smart imputation
    print("\n📊 Step 2: Applying intelligent imputation...")
    exclude_cols = [target_column] if target_column else []
    
    df_imputed = smart_impute_numeric_columns(df_cleaned, exclude_cols)
    print("   ✅ Smart imputation completed")
    
    preprocessor_report['steps_completed'].append('smart_imputation')
    
    # Step 3: Validation
    print("\n✅ Step 3: Validating data quality...")
    
    # Get model columns for validation
    model_columns = None
    if target_column:
        model_columns = [col for col in df_imputed.columns if col != target_column]
    
    validation_results = validate_data_for_modeling(df_imputed, model_columns)
    preprocessor_report['validation_results'] = validation_results
    
    if not validation_results['is_valid']:
        print("   ❌ Data validation failed!")
        for error in validation_results['errors']:
            print(f"      Error: {error}")
    else:
        print("   ✅ Data validation passed")
    
    preprocessor_report['steps_completed'].append('data_validation')
    
    print("\n🎉 Enhanced preprocessing completed!")
    
    return df_imputed, preprocessor_report


# Example usage and testing functions
def create_sample_currency_data() -> pd.DataFrame:
    """Create sample data with currency formatting for testing."""
    
    # Create base loan data
    np.random.seed(42)
    n_samples = 100
    
    sample_data = {
        'loan_id': [f'LOAN_{i:05d}' for i in range(1, n_samples + 1)],
        'original_funded_amount': [
            f'$ {np.random.uniform(5000, 50000):,.2f}' for _ in range(n_samples)
        ],
        'outstanding_balance': [
            f'${np.random.uniform(1000, 45000):,.2f}' for _ in range(n_samples)
        ],
        'percent_paid': np.random.uniform(0, 100, n_samples),
        'borrower_age': np.random.normal(35, 10, n_samples),
    }
    
    # Add some missing values
    for i in range(0, n_samples, 10):
        sample_data['borrower_age'][i] = np.nan
    for i in range(0, n_samples, 15):
        sample_data['outstanding_balance'][i] = None
    
    return pd.DataFrame(sample_data)


if __name__ == "__main__":
    # Test the currency utilities
    print("Testing Currency Utilities...")
    
    # Create sample data with currency formatting issues
    test_df = create_sample_currency_data()
    print(f"\nOriginal DataFrame shape: {test_df.shape}")
    print(f"Data types:\n{test_df.dtypes}")
    
    # Test enhanced preprocessing
    processed_df, report = enhanced_preprocess_for_prediction(test_df)
    
    print(f"\nProcessed DataFrame shape: {processed_df.shape}")
    print(f"Final data types:\n{processed_df.dtypes}")
    
    # Print summary
    print(f"\nPreprocessing Summary:")
    print(f"- Currency columns detected: {len(report['currency_columns_detected'])}")
    print(f"- Steps completed: {report['steps_completed']}")
    print(f"- Values converted: {report['data_cleaning_summary'].get('total_values_converted', 0)}")
