#!/usr/bin/env python3
"""
Script to test all Jupyter notebooks in the repository for errors.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def test_notebook(notebook_path):
    """
    Test a single notebook by executing it with nbconvert.
    Returns (success, error_message)
    """
    try:
        # Execute the notebook using nbconvert
        result = subprocess.run([
            'jupyter', 'nbconvert', 
            '--to', 'notebook',
            '--execute',
            '--ExecutePreprocessor.timeout=300',  # 5 minute timeout
            '--output', '/tmp/test_output.ipynb',
            str(notebook_path)
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            return True, None
        else:
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        return False, "Notebook execution timed out"
    except Exception as e:
        return False, f"Error executing notebook: {str(e)}"

def main():
    """Main function to test all notebooks."""
    # Find all notebook files
    notebook_files = list(Path('.').glob('*.ipynb'))
    
    if not notebook_files:
        print("No notebook files found in the current directory.")
        return 1
    
    print(f"Found {len(notebook_files)} notebook(s) to test:")
    for nb in notebook_files:
        print(f"  - {nb}")
    print()
    
    # Test each notebook
    results = {}
    for notebook_path in sorted(notebook_files):
        print(f"Testing {notebook_path}...")
        success, error = test_notebook(notebook_path)
        results[str(notebook_path)] = (success, error)
        
        if success:
            print(f"  ✅ {notebook_path} - PASSED")
        else:
            print(f"  ❌ {notebook_path} - FAILED")
            print(f"     Error: {error}")
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for success, _ in results.values() if success)
    failed = len(results) - passed
    
    print(f"Total notebooks: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed notebooks:")
        for notebook, (success, error) in results.items():
            if not success:
                print(f"  - {notebook}: {error}")
        return 1
    else:
        print("\n🎉 All notebooks passed!")
        return 0

if __name__ == "__main__":
    sys.exit(main())