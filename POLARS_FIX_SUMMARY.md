# Polars Feature Engineering Fix - Scenario 4

## Problem Description

The Polars feature engineering function in `scenario4.ipynb` was failing with the following error:

```
SchemaError: invalid series dtype: expected `String`, got `datetime[ns]` for series with name `transaction_date`
```

## Root Cause

The issue occurred in the Polars feature engineering function at lines 418-424 of the notebook. The problematic code was:

```python
# Ensure datetime types
transactions_pl = transactions_pl.with_columns([
    pl.col("transaction_date").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
])

customers_pl = customers_pl.with_columns([
    pl.col("registration_date").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
])
```

**The problem:** When pandas DataFrames with datetime columns are converted to Polars using `pl.from_pandas()`, the datetime columns maintain their datetime type (`Datetime(time_unit='ns', time_zone=None)`). However, the code was trying to use `str.strptime()` which expects a String type, not a datetime type.

## Solution

The fix was to replace the string parsing approach with a direct datetime cast:

```python
# Ensure datetime types (cast to datetime if not already)
transactions_pl = transactions_pl.with_columns([
    pl.col("transaction_date").cast(pl.Datetime)
])

customers_pl = customers_pl.with_columns([
    pl.col("registration_date").cast(pl.Datetime)
])
```

## Why This Fix Works

1. **Type Safety**: `cast(pl.Datetime)` works regardless of whether the input is already a datetime type or a string that needs parsing
2. **Efficiency**: No unnecessary string parsing when the data is already in datetime format
3. **Robustness**: Handles both scenarios - data coming from pandas (already datetime) or data that might be strings

## Files Modified

- `scenario4.ipynb`: Fixed the Polars feature engineering function

## Testing

The fix was verified with:
1. A standalone test script (`test_polars_fix.py`) that reproduces the issue and confirms the fix
2. Direct testing of the problematic code pattern
3. Verification that the notebook now contains the corrected code

## Key Takeaway

When working with Polars and pandas interoperability, be aware that:
- `pl.from_pandas()` preserves pandas datetime types as Polars datetime types
- Use `cast(pl.Datetime)` instead of `str.strptime()` when the data might already be in datetime format
- Always check the actual data types when debugging schema errors