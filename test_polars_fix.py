#!/usr/bin/env python3
"""
Test script to verify the Polars feature engineering fix
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import polars as pl
import numpy as np
import time
from datetime import datetime, timedelta

def generate_test_dataset(n_customers=100, n_transactions_per_customer=5):
    """Generate a small test dataset"""
    np.random.seed(42)
    
    print(f"Generating test dataset: {n_customers} customers, ~{n_transactions_per_customer} transactions each")
    
    # Customer demographics
    customers = []
    for i in range(n_customers):
        customer = {
            'customer_id': f"CUST_{i:06d}",
            'age': np.random.randint(18, 80),
            'gender': np.random.choice(['M', 'F'], p=[0.48, 0.52]),
            'income_bracket': np.random.choice(['Low', 'Medium', 'High'], p=[0.3, 0.5, 0.2]),
            'registration_date': datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 1460)),
            'city': np.random.choice(['NYC', 'LA', 'Chicago'], p=[0.4, 0.3, 0.3]),
            'preferred_channel': np.random.choice(['Online', 'Mobile', 'Store'], p=[0.4, 0.4, 0.2]),
            'will_churn': np.random.choice([0, 1], p=[0.8, 0.2])
        }
        customers.append(customer)
    
    customers_df = pd.DataFrame(customers)
    
    # Generate transactions
    transactions = []
    products = [f"PROD_{i:04d}" for i in range(1, 101)]
    categories = ['Electronics', 'Clothing', 'Books', 'Home']
    
    for _, customer in customers_df.iterrows():
        n_txns = max(1, np.random.poisson(n_transactions_per_customer))
        
        for txn_idx in range(n_txns):
            days_since_reg = (datetime.now() - customer['registration_date']).days
            txn_date = customer['registration_date'] + timedelta(
                days=np.random.randint(0, min(days_since_reg, 365))
            )
            
            transaction = {
                'transaction_id': f"TXN_{len(transactions):08d}",
                'customer_id': customer['customer_id'],
                'product_id': np.random.choice(products),
                'product_category': np.random.choice(categories),
                'amount': round(np.random.lognormal(3, 1), 2),
                'transaction_date': txn_date,
                'channel': np.random.choice(['Online', 'Mobile', 'Store']),
                'discount_applied': np.random.choice([True, False], p=[0.3, 0.7]),
                'payment_method': np.random.choice(['Credit', 'Debit', 'Cash'], p=[0.5, 0.3, 0.2])
            }
            transactions.append(transaction)
    
    transactions_df = pd.DataFrame(transactions)
    
    print(f"Generated {len(transactions_df)} transactions for {len(customers_df)} customers")
    
    return transactions_df, customers_df

def polars_feature_engineering_fixed(transactions_df: pd.DataFrame, customers_df: pd.DataFrame) -> pd.DataFrame:
    """Fixed Polars feature engineering function"""
    print("=== Polars Feature Engineering (Fixed) ===")
    start_time = time.time()
    
    # Convert to Polars DataFrames
    transactions_pl = pl.from_pandas(transactions_df)
    customers_pl = pl.from_pandas(customers_df)
    
    print(f"Original transaction_date dtype: {transactions_pl['transaction_date'].dtype}")
    print(f"Original registration_date dtype: {customers_pl['registration_date'].dtype}")
    
    # Ensure datetime types (cast to datetime if not already) - FIXED VERSION
    transactions_pl = transactions_pl.with_columns([
        pl.col("transaction_date").cast(pl.Datetime)
    ])
    
    customers_pl = customers_pl.with_columns([
        pl.col("registration_date").cast(pl.Datetime)
    ])
    
    print("Creating basic RFM features...")
    
    # Get reference date
    reference_date = transactions_pl.select(pl.col("transaction_date").max()).item()
    
    # Basic RFM Features
    rfm_features = (
        transactions_pl
        .group_by("customer_id")
        .agg([
            pl.col("transaction_date").min().alias("first_purchase"),
            pl.col("transaction_date").max().alias("last_purchase"),
            pl.col("transaction_id").count().alias("frequency"),
            pl.col("amount").sum().alias("monetary"),
            pl.col("amount").mean().alias("avg_amount")
        ])
        .with_columns([
            (pl.lit(reference_date) - pl.col("last_purchase")).dt.total_days().alias("recency"),
            (pl.col("last_purchase") - pl.col("first_purchase")).dt.total_days().alias("customer_lifetime")
        ])
    )
    
    print("Creating behavioral features...")
    
    # Behavioral Features
    behavioral_features = (
        transactions_pl
        .group_by("customer_id")
        .agg([
            pl.col("product_category").n_unique().alias("category_diversity"),
            pl.col("channel").mode().first().alias("preferred_channel"),
            pl.col("discount_applied").mean().alias("discount_usage_rate")
        ])
    )
    
    print("Merging features...")
    
    # Merge all features
    features = (
        customers_pl
        .join(rfm_features, on="customer_id", how="left")
        .join(behavioral_features, on="customer_id", how="left")
    )
    
    # Fill null values
    features = features.fill_null(0)
    
    processing_time = time.time() - start_time
    print(f"Polars feature engineering completed in {processing_time:.2f} seconds")
    
    # Convert back to pandas for ML compatibility
    features_pd = features.to_pandas()
    print(f"Created {len(features_pd.columns)} features for {len(features_pd)} customers")
    
    return features_pd

if __name__ == "__main__":
    print("Testing Polars feature engineering fix...")
    
    # Generate test data
    transactions_df, customers_df = generate_test_dataset()
    
    print(f"\nDataset info:")
    print(f"Transactions shape: {transactions_df.shape}")
    print(f"Customers shape: {customers_df.shape}")
    print(f"Transaction date dtype: {transactions_df['transaction_date'].dtype}")
    print(f"Registration date dtype: {customers_df['registration_date'].dtype}")
    
    # Test the fixed function
    try:
        polars_features = polars_feature_engineering_fixed(transactions_df.copy(), customers_df.copy())
        print(f"\n✅ SUCCESS: Polars features shape: {polars_features.shape}")
        print(f"Feature columns: {list(polars_features.columns)}")
        print("\nSample features:")
        print(polars_features.head())
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()