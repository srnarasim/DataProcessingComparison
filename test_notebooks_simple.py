#!/usr/bin/env python3
"""
Simple notebook validation script to test all notebooks for syntax and basic execution.
"""

import nbformat
import sys
import traceback
from pathlib import Path

def validate_notebook_syntax(notebook_path):
    """Validate notebook syntax without executing it."""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        print(f"✅ {notebook_path}: Valid JSON structure")
        
        # Check for common issues
        issues = []
        
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                source = ''.join(cell.source) if isinstance(cell.source, list) else cell.source
                
                # Check for magic commands that might cause issues
                if 'magic' in source.lower():
                    issues.append(f"Cell {i}: Contains 'magic' reference")
                
                # Check for problematic imports
                if 'from jupyter_styling import' in source and 'except ImportError' not in source:
                    issues.append(f"Cell {i}: Unhandled jupyter_styling import")
                
                # Check for syntax issues
                try:
                    # Basic syntax check (won't catch runtime issues)
                    compile(source, f'<cell {i}>', 'exec')
                except SyntaxError as e:
                    issues.append(f"Cell {i}: Syntax error - {e}")
        
        if issues:
            print(f"⚠️  {notebook_path}: Found potential issues:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print(f"✅ {notebook_path}: No syntax issues found")
            
        return len(issues) == 0
        
    except Exception as e:
        print(f"❌ {notebook_path}: Failed to validate - {e}")
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
    
    print("🔍 Testing notebook syntax and structure...\n")
    
    all_valid = True
    for notebook in notebooks:
        if Path(notebook).exists():
            valid = validate_notebook_syntax(notebook)
            all_valid = all_valid and valid
        else:
            print(f"❌ {notebook}: File not found")
            all_valid = False
        print()
    
    if all_valid:
        print("🎉 All notebooks passed validation!")
        return 0
    else:
        print("❌ Some notebooks have issues that need fixing.")
        return 1

if __name__ == "__main__":
    sys.exit(main())