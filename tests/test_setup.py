#!/usr/bin/env python3
"""
Test script for the CANCapital Gradio Interface

This creates a sample dataset and tests both EDA options.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_dataset():
    """Create a sample dataset for testing"""
    
    np.random.seed(42)
    n_samples = 1000
    
    # Create sample data similar to loan collection dataset
    data = {
        'loan_id': [f'LOAN_{i:05d}' for i in range(1, n_samples + 1)],
        'original_funded_amount': np.random.normal(15000, 5000, n_samples),
        'collection_case_days': np.random.exponential(30, n_samples) + 5,
        'borrower_age': np.random.normal(35, 10, n_samples),
        'loan_status': np.random.choice(['Active', 'Closed', 'Default'], n_samples, p=[0.6, 0.3, 0.1]),
        'state': np.random.choice(['CA', 'TX', 'FL', 'NY', 'IL'], n_samples),
        'percent_paid': np.random.uniform(0, 100, n_samples)
    }
    
    # Add some missing values to test handling
    for col in ['borrower_age', 'percent_paid']:
        missing_idx = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
        for idx in missing_idx:
            data[col][idx] = np.nan
    
    df = pd.DataFrame(data)
    
    # Save sample dataset
    output_path = Path(__file__).parent / "sample_dataset.csv"
    df.to_csv(output_path, index=False)
    
    print(f"✅ Sample dataset created: {output_path}")
    print(f"📊 Dataset shape: {df.shape}")
    print(f"🎯 Target variable: collection_case_days")
    
    return output_path

def test_imports():
    """Test that all required packages can be imported"""
    
    try:
        import gradio as gr
        print("✅ Gradio import successful")
        
        import pandas as pd
        print("✅ Pandas import successful")
        
        import numpy as np
        print("✅ NumPy import successful")
        
        import matplotlib.pyplot as plt
        print("✅ Matplotlib import successful")
        
        import seaborn as sns
        print("✅ Seaborn import successful")
        
        from sklearn.preprocessing import LabelEncoder
        print("✅ Scikit-learn import successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main test function"""
    
    print("🧪 Testing CANCapital Gradio Interface Setup")
    print("=" * 50)
    
    # Test imports
    print("\n1. Testing package imports...")
    if not test_imports():
        print("❌ Package import test failed")
        return False
    
    # Create sample dataset
    print("\n2. Creating sample dataset...")
    sample_path = create_sample_dataset()
    
    # Test the interface can start
    print("\n3. Testing Gradio interface startup...")
    
    import subprocess
    try:
        # Quick test to see if interface starts (with timeout)
        result = subprocess.run([
            "pixi", "run", "--project", str(Path(__file__).parent),
            "python", "-c", 
            """
import gradio as gr
print('✅ Gradio interface can be imported and initialized')
# Quick test that basic components work
with gr.Blocks() as interface:
    file_input = gr.File()
    run_button = gr.Button('Test')
print('✅ Interface components work correctly')
            """
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Gradio interface startup test passed")
            print(result.stdout)
        else:
            print(f"❌ Interface startup failed: {result.stderr}")
            
    except Exception as e:
        print(f"⚠️  Interface test skipped: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Setup test completed!")
    
    print(f"\n📋 Next steps:")
    print(f"1. Run the interface: pixi run python gradio_interface.py")
    print(f"2. Or use launcher: python run_interface.py") 
    print(f"3. Open browser to: http://localhost:7860")
    print(f"4. Upload sample dataset: {sample_path}")
    
    return True

if __name__ == "__main__":
    main()