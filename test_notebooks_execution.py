#!/usr/bin/env python3
"""
Test notebook execution with timeout and error handling.
"""

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys
import traceback
from pathlib import Path
import tempfile
import os

def test_notebook_execution(notebook_path, timeout=120):
    """Test notebook execution with timeout."""
    try:
        print(f"🔍 Testing {notebook_path}...")
        
        # Load notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Create a copy for testing (don't modify original)
        test_nb = nb.copy()
        
        # Skip cells that might cause issues in automated testing
        for i, cell in enumerate(test_nb.cells):
            if cell.cell_type == 'code':
                source = ''.join(cell.source) if isinstance(cell.source, list) else cell.source
                
                # Skip cells with problematic content for automated testing
                if any(skip_pattern in source for skip_pattern in [
                    '!pip install',  # Skip pip installs
                    'input(',        # Skip interactive input
                    'plt.show()',    # Skip matplotlib show
                    'display(HTML',  # Skip HTML display
                    'from IPython.display import HTML',  # Skip IPython display
                ]):
                    print(f"   ⏭️  Skipping cell {i} (contains interactive/install commands)")
                    cell.source = "# Skipped for automated testing"
        
        # Execute with timeout
        ep = ExecutePreprocessor(
            timeout=timeout, 
            kernel_name='python3',
            allow_errors=True  # Continue execution even if some cells fail
        )
        
        # Execute in a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            ep.preprocess(test_nb, {'metadata': {'path': temp_dir}})
        
        # Check for execution errors
        errors = []
        for i, cell in enumerate(test_nb.cells):
            if cell.cell_type == 'code' and 'outputs' in cell:
                for output in cell.outputs:
                    if output.get('output_type') == 'error':
                        errors.append(f"Cell {i}: {output.get('ename', 'Unknown error')}")
        
        if errors:
            print(f"   ⚠️  Execution completed with errors:")
            for error in errors[:3]:  # Show first 3 errors
                print(f"      - {error}")
            if len(errors) > 3:
                print(f"      ... and {len(errors) - 3} more errors")
            return False
        else:
            print(f"   ✅ Execution completed successfully")
            return True
            
    except Exception as e:
        print(f"   ❌ Execution failed: {e}")
        return False

def main():
    """Test all notebooks."""
    notebooks = [
        'overview.ipynb',
        'scenario1.ipynb', 
        'scenario2.ipynb',
        'scenario3.ipynb',
        'scenario4.ipynb'
    ]
    
    print("🚀 Testing notebook execution (with skipped interactive cells)...\n")
    
    results = {}
    for notebook in notebooks:
        if Path(notebook).exists():
            results[notebook] = test_notebook_execution(notebook)
        else:
            print(f"❌ {notebook}: File not found")
            results[notebook] = False
        print()
    
    # Summary
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for notebook, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {notebook}")
    
    print(f"\nOverall: {passed}/{total} notebooks passed")
    
    if passed == total:
        print("🎉 All notebooks are executable!")
        return 0
    else:
        print("⚠️  Some notebooks have execution issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())