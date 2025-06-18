#!/usr/bin/env python3
"""
Generate sample data for the Data Processing Comparison scenarios.

This script creates realistic sample datasets that are used across
all scenarios in the comparison notebooks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import random
from datetime import datetime, timedelta

def generate_transaction_data(num_files=5, rows_per_file=1000000):
    """Generate sample transaction data for ETL scenarios."""
    
    print("🔄 Generating sample transaction data...")
    
    # Create output directory
    output_dir = Path("data/transactions/2025/01")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample data parameters
    customer_ids = range(1, 100001)  # 100k customers
    product_ids = range(1, 10001)    # 10k products
    store_ids = range(1, 501)        # 500 stores
    
    for file_idx in range(num_files):
        print(f"  📄 Generating file {file_idx + 1}/{num_files}...")
        
        # Generate random data
        np.random.seed(42 + file_idx)  # Reproducible but different per file
        
        data = {
            'transaction_id': range(file_idx * rows_per_file, (file_idx + 1) * rows_per_file),
            'customer_id': np.random.choice(customer_ids, rows_per_file),
            'product_id': np.random.choice(product_ids, rows_per_file),
            'store_id': np.random.choice(store_ids, rows_per_file),
            'quantity': np.random.randint(1, 11, rows_per_file),
            'unit_price': np.round(np.random.uniform(5.0, 500.0, rows_per_file), 2),
            'discount_percent': np.random.choice([0, 5, 10, 15, 20], rows_per_file, p=[0.6, 0.2, 0.1, 0.05, 0.05]),
            'timestamp': [
                datetime(2025, 1, file_idx + 1) + timedelta(
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59),
                    seconds=random.randint(0, 59)
                ) for _ in range(rows_per_file)
            ]
        }
        
        df = pd.DataFrame(data)
        
        # Calculate derived fields
        df['total_amount'] = df['quantity'] * df['unit_price']
        df['discount_amount'] = df['total_amount'] * (df['discount_percent'] / 100)
        df['final_amount'] = df['total_amount'] - df['discount_amount']
        
        # Save to CSV
        filename = output_dir / f"transactions_2025-01-{file_idx+1:02d}.csv"
        df.to_csv(filename, index=False)
        
        print(f"    ✅ Created {filename} with {len(df):,} rows")
    
    print(f"✅ Generated {num_files} transaction files in {output_dir}")
    return output_dir

def generate_customer_data():
    """Generate sample customer data."""
    
    print("🔄 Generating customer data...")
    
    output_dir = Path("data/customers")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)
    
    # Generate customer data
    num_customers = 100000
    
    first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Lisa', 'Robert', 'Emily', 'James', 'Jessica']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
    states = ['NY', 'CA', 'IL', 'TX', 'AZ', 'PA', 'TX', 'CA', 'TX', 'CA']
    
    data = {
        'customer_id': range(1, num_customers + 1),
        'first_name': np.random.choice(first_names, num_customers),
        'last_name': np.random.choice(last_names, num_customers),
        'email': [f"customer{i}@email.com" for i in range(1, num_customers + 1)],
        'city': np.random.choice(cities, num_customers),
        'state': np.random.choice(states, num_customers),
        'age': np.random.randint(18, 80, num_customers),
        'signup_date': [
            datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1825))
            for _ in range(num_customers)
        ]
    }
    
    df = pd.DataFrame(data)
    filename = output_dir / "customers.csv"
    df.to_csv(filename, index=False)
    
    print(f"✅ Created {filename} with {len(df):,} rows")
    return filename

def generate_product_data():
    """Generate sample product data."""
    
    print("🔄 Generating product data...")
    
    output_dir = Path("data/products")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)
    
    # Generate product data
    num_products = 10000
    
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Toys', 'Health', 'Automotive']
    brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE', 'BrandF', 'BrandG', 'BrandH']
    
    data = {
        'product_id': range(1, num_products + 1),
        'product_name': [f"Product {i}" for i in range(1, num_products + 1)],
        'category': np.random.choice(categories, num_products),
        'brand': np.random.choice(brands, num_products),
        'cost': np.round(np.random.uniform(1.0, 200.0, num_products), 2),
        'retail_price': np.round(np.random.uniform(5.0, 500.0, num_products), 2),
        'weight_kg': np.round(np.random.uniform(0.1, 50.0, num_products), 2),
        'in_stock': np.random.choice([True, False], num_products, p=[0.85, 0.15])
    }
    
    df = pd.DataFrame(data)
    filename = output_dir / "products.csv"
    df.to_csv(filename, index=False)
    
    print(f"✅ Created {filename} with {len(df):,} rows")
    return filename

def main():
    """Generate all sample data."""
    print("🚀 Starting sample data generation...")
    print("📊 This will create realistic datasets for all comparison scenarios")
    print()
    
    # Generate all datasets
    transaction_dir = generate_transaction_data(num_files=5, rows_per_file=200000)  # Smaller for local dev
    customer_file = generate_customer_data()
    product_file = generate_product_data()
    
    print()
    print("✅ Sample data generation complete!")
    print()
    print("📁 Generated files:")
    print(f"  📄 Transactions: {transaction_dir}/*.csv (5 files)")
    print(f"  👥 Customers: {customer_file}")
    print(f"  🛍️ Products: {product_file}")
    print()
    print("🎯 You can now run all scenario notebooks successfully!")

if __name__ == "__main__":
    main()