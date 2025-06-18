#!/usr/bin/env python3
"""
Quick notebook validation focusing on JSON structure and critical issues.
"""

import nbformat
import json
import sys
from pathlib import Path

def validate_notebook(notebook_path):
    """Validate notebook structure and check for critical issues."""
    try:
        print(f"🔍 Validating {notebook_path}...")
        
        # Test JSON structure
        with open(notebook_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Load as notebook
        nb = nbformat.read(notebook_path, as_version=4)
        
        issues = []
        
        # Check for critical structural issues
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                source = ''.join(cell.source) if isinstance(cell.source, list) else cell.source
                
                # Check for unmatched quotes/brackets that could break JSON
                if source.count('"') % 2 != 0:
                    issues.append(f"Cell {i}: Unmatched quotes")
                
                # Check for missing commas in JSON structure (common issue)
                if 'enhanced_colors' in source and not source.strip().endswith(','):
                    if i < len(nb.cells) - 1:  # Not the last cell
                        issues.append(f"Cell {i}: Missing comma after enhanced_colors")
        
        if issues:
            print(f"   ⚠️  Found issues:")
            for issue in issues:
                print(f"      - {issue}")
            return False
        else:
            print(f"   ✅ Valid structure")
            return True
            
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Validation error: {e}")
        return False

def main():
    """Validate all notebooks."""
    notebooks = [
        'overview.ipynb',
        'scenario1.ipynb', 
        'scenario2.ipynb',
        'scenario3.ipynb',
        'scenario4.ipynb'
    ]
    
    print("📋 Quick notebook validation...\n")
    
    results = {}
    for notebook in notebooks:
        if Path(notebook).exists():
            results[notebook] = validate_notebook(notebook)
        else:
            print(f"❌ {notebook}: File not found")
            results[notebook] = False
        print()
    
    # Summary
    print("📊 Validation Results:")
    print("=" * 40)
    
    passed = sum(results.values())
    total = len(results)
    
    for notebook, success in results.items():
        status = "✅ VALID" if success else "❌ INVALID"
        print(f"{status} {notebook}")
    
    print(f"\nOverall: {passed}/{total} notebooks are valid")
    
    if passed == total:
        print("🎉 All notebooks have valid structure!")
        return 0
    else:
        print("⚠️  Some notebooks need fixes.")
        return 1

if __name__ == "__main__":
    sys.exit(main())