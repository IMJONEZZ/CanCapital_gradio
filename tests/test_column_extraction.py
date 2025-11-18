#!/usr/bin/env python3
"""
Test script to verify column extraction functionality for Gradio interface

This tests the get_csv_columns function logic with various file input types
to debug why target column dropdown isn't populating.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd


def test_csv_column_extraction():
    """Test different methods of reading CSV and extracting columns"""

    print("🧪 Testing CSV Column Extraction Logic")
    print("=" * 50)

    # Test with the existing dataset.csv in the project
    test_file = "dataset.csv"

    print(f"\n📁 Test file: {test_file}")
    print(f"📍 Full path: {os.path.abspath(test_file)}")

    # Check if file exists
    if not os.path.exists(test_file):
        print(f"❌ Test file {test_file} not found!")
        return False

    try:
        # Method 1: Direct pandas read (baseline)
        print("\n🔍 Test 1: Direct pandas read_csv")
        df_direct = pd.read_csv(test_file)
        columns_direct = list(df_direct.columns)
        print(f"✅ Success! Found {len(columns_direct)} columns")
        print(f"Sample columns: {columns_direct[:5]}")

    except Exception as e:
        print(f"❌ Failed: {e}")
        columns_direct = []

    try:
        # Method 2: Read via file handle
        print("\n🔍 Test 2: Reading via file handle")
        with open(test_file, "r") as f:
            df_handle = pd.read_csv(f)
        columns_handle = list(df_handle.columns)
        print(f"✅ Success! Found {len(columns_handle)} columns")

    except Exception as e:
        print(f"❌ Failed: {e}")
        columns_handle = []

    try:
        # Method 3: Simulate Gradio file upload (temp file)
        print("\n🔍 Test 3: Simulating Gradio file upload")

        # Create a temporary copy of the dataset
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            with open(test_file, "r") as src:
                content = src.read()
                tmp.write(content)
            temp_path = tmp.name

        print(f"📝 Created temporary file: {temp_path}")

        # Test reading the temp file
        df_temp = pd.read_csv(temp_path)
        columns_temp = list(df_temp.columns)
        print(f"✅ Success! Found {len(columns_temp)} columns")

        # Simulate Gradio file object
        class MockGradioFile:
            def __init__(self, path):
                self.name = path

        mock_file = MockGradioFile(temp_path)

        # Test get_csv_columns logic with Gradio file
        def get_csv_columns(csv_file):
            """Extract column names from uploaded CSV file (from gradio_interface.py)"""
            try:
                if csv_file is None:
                    return []

                # Handle Gradio file upload object
                try:
                    if hasattr(csv_file, "name") and os.path.exists(csv_file.name):
                        # Direct file path from Gradio
                        df = pd.read_csv(csv_file.name)
                    elif hasattr(csv_file, "read"):
                        # File-like object - try to read directly
                        if hasattr(csv_file, "seek"):
                            csv_file.seek(0)  # Reset file pointer if possible
                        df = pd.read_csv(csv_file)
                    else:
                        # Try as direct path or URL
                        df = pd.read_csv(csv_file)
                except Exception as inner_e:
                    print(f"Primary method failed, trying alternative: {inner_e}")
                    # Alternative: use the original file if it exists in temp location
                    if hasattr(csv_file, "name") and os.path.exists(csv_file.name):
                        df = pd.read_csv(csv_file.name)
                    else:
                        raise inner_e

                return list(df.columns)
            except Exception as e:
                print(f"Error reading CSV columns: {e}")
                import traceback

                traceback.print_exc()
                return []

        columns_mock = get_csv_columns(mock_file)
        print(f"✅ Mock Gradio file test: Found {len(columns_mock)} columns")

        # Clean up
        os.unlink(temp_path)
        print("🧹 Temporary file cleaned up")

    except Exception as e:
        print(f"❌ Failed: {e}")
        columns_mock = []

    # Compare all methods
    print("\n📊 Results Summary:")
    print("-" * 30)

    methods = [
        ("Direct read_csv", columns_direct),
        ("File handle", columns_handle),
        ("Mock Gradio file", columns_mock),
    ]

    for method_name, columns in methods:
        if columns:
            print(f"✅ {method_name}: {len(columns)} columns - SUCCESS")
        else:
            print(f"❌ {method_name}: FAILED")

    # Check if all methods agree
    success_methods = [cols for _, cols in methods if cols]

    if len(success_methods) > 1:
        first_result = success_methods[0]
        all_agree = all(cols == first_result for cols in success_methods)

        if all_agree and len(first_result) > 0:
            print(f"\n🎉 All methods agree! Column extraction should work in Gradio")
            print("📋 First 10 columns:", first_result[:10])
            return True
        else:
            print(f"\n⚠️ Methods disagree! Something is inconsistent")
            return False
    elif len(success_methods) == 1:
        print(f"\n⚠️ Only one method worked. This might explain Gradio issues")
        return False
    else:
        print(f"\n❌ All methods failed. There's a fundamental problem")
        return False


def test_with_different_file_formats():
    """Test with other CSV files if they exist"""

    print("\n🔍 Testing with other available CSV files:")

    csv_files = [
        "sample_dataset.csv",
        "core-loan-performance-metrics.csv",
        "features-related-to-length-of-case.csv",
        "missing-data-patterns.csv",
    ]

    for csv_file in csv_files:
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                print(f"✅ {csv_file}: {len(df.columns)} columns")
            except Exception as e:
                print(f"❌ {csv_file}: Error - {e}")
        else:
            print(f"⚪ {csv_file}: Not found")


if __name__ == "__main__":
    print("🚀 Starting Column Extraction Test")

    # Change to project directory
    os.chdir(Path(__file__).parent)
    print(f"📁 Working directory: {os.getcwd()}")

    # Run main test
    success = test_csv_column_extraction()

    # Test other files if available
    test_with_different_file_formats()

    print("\n" + "=" * 50)
    if success:
        print("🎯 CONCLUSION: Column extraction should work in Gradio interface")
        print("🔧 If dropdown still doesn't populate, the issue might be:")
        print("   - Gradio event handling")
        print("   - UI state management")
        print("   - JavaScript/frontend issues")
    else:
        print("🔍 CONCLUSION: Found issue with column extraction logic")
        print("💡 Need to fix get_csv_columns function in gradio_interface.py")

    print("\n✨ Test completed!")
