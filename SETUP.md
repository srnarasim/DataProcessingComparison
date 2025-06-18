# Setup Guide for DataProcessingComparison

This guide helps you set up the environment for running all scenarios in the DataProcessingComparison project.

## Quick Setup

### 1. Install All Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python verify_setup.py
```

If all checks pass, you're ready to run the scenarios! 🎉

## Manual Installation (if needed)

If the automatic installation fails, you can install packages individually:

### Core Data Processing Libraries
```bash
pip install pandas>=2.0.0 polars>=0.20.0 duckdb>=0.9.0 pyarrow>=10.0.0 numpy>=1.24.0
```

### Apache Spark
```bash
pip install pyspark>=3.5.0 findspark>=2.0.0
```

### TabsData (Enterprise Governance)
```bash
pip install tabsdata>=0.9.6
```

### Jupyter Environment
```bash
pip install jupyter>=1.0.0 jupyterlab>=4.0.0 notebook>=7.0.0 ipywidgets>=8.0.0
```

### Visualization Libraries
```bash
pip install matplotlib>=3.7.0 seaborn>=0.12.0 plotly>=5.15.0
```

### Scientific Computing
```bash
pip install scipy>=1.10.0 scikit-learn>=1.3.0
```

## Common Issues and Solutions

### Issue: ModuleNotFoundError for pandas/polars/duckdb

**Problem**: Missing core data processing libraries
```
ModuleNotFoundError: No module named 'pandas'
```

**Solution**: 
```bash
pip install pandas polars duckdb numpy
python verify_setup.py
```

### Issue: TabsData import warnings

**Problem**: Deprecation warnings when importing TabsData
```
UserWarning: pkg_resources is deprecated...
```

**Solution**: This is a known warning and doesn't affect functionality. TabsData works correctly despite the warning.

### Issue: PySpark not found

**Problem**: Missing Apache Spark libraries
```
ModuleNotFoundError: No module named 'pyspark'
```

**Solution**:
```bash
pip install pyspark findspark
```

### Issue: Jupyter notebooks won't start

**Problem**: Missing Jupyter environment
```
Command 'jupyter' not found
```

**Solution**:
```bash
pip install jupyter jupyterlab notebook ipywidgets
```

## Environment Requirements

- **Python**: 3.8 or higher
- **Memory**: At least 4GB RAM (8GB recommended for large datasets)
- **Storage**: At least 2GB free space for data files and dependencies

## Verification

After installation, run the verification script to ensure everything is working:

```bash
python verify_setup.py
```

The script will:
- ✅ Check all required packages are installed
- ✅ Verify import functionality  
- ✅ Test basic operations for each tool
- ✅ Report any issues with suggested fixes

## Scenarios Overview

Once setup is complete, you can run:

- **Scenario 1**: Basic data processing comparison
- **Scenario 2**: ETL pipeline performance comparison (Spark, Polars, Pandas, TabsData)
- **Scenario 3**: Real-time streaming and concurrent processing
- **Scenario 4**: Machine learning pipeline comparison

## Getting Help

If you encounter issues:

1. **Run verification**: `python verify_setup.py`
2. **Check requirements**: Ensure all packages in `requirements.txt` are installed
3. **Update packages**: `pip install --upgrade -r requirements.txt`
4. **Check Python version**: Ensure you're using Python 3.8+

## Next Steps

After successful setup:
1. Open JupyterLab: `jupyter lab`
2. Navigate to the scenario notebooks
3. Run the cells to see the comparisons in action!

Happy data processing! 🚀