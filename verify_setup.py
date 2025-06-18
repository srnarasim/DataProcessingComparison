#!/usr/bin/env python3
"""
Setup Verification Script for DataProcessingComparison
======================================================

This script verifies that all required dependencies are properly installed
and working correctly for all scenarios in the comparison project.

Usage:
    python verify_setup.py

The script will:
1. Check all required Python packages
2. Verify import functionality
3. Test basic operations for each tool
4. Report any missing dependencies or issues
"""

import sys
import subprocess
from typing import Dict, List, Tuple, Optional

def check_package_import(package_name: str, import_name: Optional[str] = None) -> Tuple[bool, str]:
    """
    Check if a package can be imported successfully.
    
    Args:
        package_name: Name of the package for display
        import_name: Actual import name (if different from package_name)
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True, f"✅ {package_name}: OK"
    except ImportError as e:
        return False, f"❌ {package_name}: MISSING ({str(e)})"
    except Exception as e:
        return False, f"⚠️ {package_name}: ERROR ({str(e)})"

def check_package_version(package_name: str, import_name: Optional[str] = None) -> Tuple[bool, str]:
    """
    Check package version and import functionality.
    
    Args:
        package_name: Name of the package for display
        import_name: Actual import name (if different from package_name)
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'Unknown')
        return True, f"✅ {package_name}: v{version}"
    except ImportError as e:
        return False, f"❌ {package_name}: MISSING ({str(e)})"
    except Exception as e:
        return False, f"⚠️ {package_name}: ERROR ({str(e)})"

def test_basic_functionality() -> List[Tuple[bool, str]]:
    """Test basic functionality of key libraries."""
    results = []
    
    # Test Pandas
    try:
        import pandas as pd
        df = pd.DataFrame({'test': [1, 2, 3]})
        assert len(df) == 3
        results.append((True, "✅ Pandas: Basic operations work"))
    except Exception as e:
        results.append((False, f"❌ Pandas: Basic test failed ({str(e)})"))
    
    # Test Polars
    try:
        import polars as pl
        df = pl.DataFrame({'test': [1, 2, 3]})
        assert len(df) == 3
        results.append((True, "✅ Polars: Basic operations work"))
    except Exception as e:
        results.append((False, f"❌ Polars: Basic test failed ({str(e)})"))
    
    # Test DuckDB
    try:
        import duckdb
        conn = duckdb.connect(':memory:')
        result = conn.execute("SELECT 1 as test").fetchone()
        assert result[0] == 1
        conn.close()
        results.append((True, "✅ DuckDB: Basic operations work"))
    except Exception as e:
        results.append((False, f"❌ DuckDB: Basic test failed ({str(e)})"))
    
    # Test NumPy
    try:
        import numpy as np
        arr = np.array([1, 2, 3])
        assert len(arr) == 3
        results.append((True, "✅ NumPy: Basic operations work"))
    except Exception as e:
        results.append((False, f"❌ NumPy: Basic test failed ({str(e)})"))
    
    # Test TabsData (import only, as it requires server setup)
    try:
        import tabsdata as td
        results.append((True, "✅ TabsData: Import successful"))
    except Exception as e:
        results.append((False, f"❌ TabsData: Import failed ({str(e)})"))
    
    return results

def install_missing_packages(missing_packages: List[str]) -> bool:
    """
    Attempt to install missing packages using pip.
    
    Args:
        missing_packages: List of package names to install
    
    Returns:
        True if installation was successful, False otherwise
    """
    if not missing_packages:
        return True
    
    print(f"\n🔧 Attempting to install missing packages: {', '.join(missing_packages)}")
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ])
        print("✅ Installation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ requirements.txt not found. Please install packages manually.")
        return False

def main():
    """Main verification function."""
    print("🚀 DataProcessingComparison Setup Verification")
    print("=" * 50)
    
    # Core data processing libraries
    core_packages = [
        ('Pandas', 'pandas'),
        ('Polars', 'polars'),
        ('DuckDB', 'duckdb'),
        ('NumPy', 'numpy'),
        ('PyArrow', 'pyarrow'),
    ]
    
    # Spark libraries
    spark_packages = [
        ('PySpark', 'pyspark'),
        ('FindSpark', 'findspark'),
    ]
    
    # TabsData
    tabsdata_packages = [
        ('TabsData', 'tabsdata'),
    ]
    
    # Jupyter and notebook support
    jupyter_packages = [
        ('Jupyter', 'jupyter'),
        ('JupyterLab', 'jupyterlab'),
        ('IPython Widgets', 'ipywidgets'),
    ]
    
    # Visualization libraries
    viz_packages = [
        ('Matplotlib', 'matplotlib'),
        ('Seaborn', 'seaborn'),
        ('Plotly', 'plotly'),
    ]
    
    # Scientific computing
    sci_packages = [
        ('SciPy', 'scipy'),
        ('Scikit-learn', 'sklearn'),
    ]
    
    # Utility libraries
    util_packages = [
        ('Requests', 'requests'),
        ('Faker', 'faker'),
        ('TQDM', 'tqdm'),
    ]
    
    all_packages = [
        ("Core Data Processing", core_packages),
        ("Apache Spark", spark_packages),
        ("TabsData", tabsdata_packages),
        ("Jupyter Environment", jupyter_packages),
        ("Visualization", viz_packages),
        ("Scientific Computing", sci_packages),
        ("Utilities", util_packages),
    ]
    
    missing_packages = []
    all_success = True
    
    # Check all packages
    for category, packages in all_packages:
        print(f"\n📦 {category}:")
        for display_name, import_name in packages:
            success, message = check_package_version(display_name, import_name)
            print(f"  {message}")
            if not success:
                missing_packages.append(import_name)
                all_success = False
    
    # Test basic functionality
    print(f"\n🧪 Basic Functionality Tests:")
    func_results = test_basic_functionality()
    for success, message in func_results:
        print(f"  {message}")
        if not success:
            all_success = False
    
    # Summary
    print(f"\n" + "=" * 50)
    if all_success:
        print("🎉 All dependencies are properly installed and working!")
        print("✅ You can now run all scenarios in the comparison project.")
    else:
        print("⚠️ Some dependencies are missing or not working properly.")
        
        if missing_packages:
            print(f"\n📋 Missing packages: {', '.join(set(missing_packages))}")
            
            # Offer to install missing packages
            response = input("\n🤔 Would you like to install missing packages? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                if install_missing_packages(missing_packages):
                    print("\n🔄 Re-running verification...")
                    main()  # Re-run verification
                    return
        
        print("\n💡 To fix issues:")
        print("   1. Run: pip install -r requirements.txt")
        print("   2. Or install individual packages: pip install <package_name>")
        print("   3. Re-run this script to verify: python verify_setup.py")
    
    print(f"\n📚 For more information, see the project README.md")

if __name__ == "__main__":
    main()