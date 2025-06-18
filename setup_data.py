#!/usr/bin/env python3
"""
Quick setup script for generating sample data.
Run this if you encounter file not found errors in the notebooks.
"""

import os
import sys
from pathlib import Path

def main():
    print("🚀 Setting up sample data for Data Processing Comparison...")
    
    # Check if we're in the right directory
    if not os.path.exists("scenario2.ipynb"):
        print("❌ Error: Please run this script from the repository root directory")
        print("   (where scenario2.ipynb is located)")
        sys.exit(1)
    
    # Check if data already exists
    data_dir = Path("data/transactions/2025/01")
    if data_dir.exists() and len(list(data_dir.glob("*.csv"))) >= 5:
        print("✅ Sample data already exists!")
        print(f"📁 Found data in: {data_dir}")
        return
    
    # Generate data
    print("📊 Generating sample data...")
    try:
        from generate_sample_data import main as generate_data
        generate_data()
    except ImportError:
        print("❌ Error: Could not import data generation module")
        print("   Please ensure generate_sample_data.py is in the current directory")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error generating data: {e}")
        sys.exit(1)
    
    print("✅ Setup complete! You can now run all scenario notebooks.")

if __name__ == "__main__":
    main()