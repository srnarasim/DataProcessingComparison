# Scenario 1: The Jupyter Notebook Data Scientist
# Interactive exploration, frequent iteration, rich ecosystem integration

import pandas as pd
import polars as pl
import duckdb
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Install required packages (uncomment if running in Colab)
# !pip install polars duckdb pyarrow

print("🔬 Scenario 1: Interactive Data Science Analysis")
print("="*50)

# Generate realistic e-commerce transaction data
def generate_sample_data(n_rows=2_000_000):
    """Generate realistic e-commerce transaction data for analysis"""
    np.random.seed(42)  # Reproducible results
    
    print(f"Generating {n_rows:,} realistic e-commerce transactions...")
    
    # Realistic customer and product distributions
    n_customers = 50_000
    n_products = 2_000
    
    customers = [f"CUST_{i:06d}" for i in range(1, n_customers + 1)]
    products = [f"PROD_{i:04d}" for i in range(1, n_products + 1)]
    categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports', 'Beauty']
    
    # Create realistic transaction data
    data = {
        'customer_id': np.random.choice(customers, n_rows),
        'product_id': np.random.choice(products, n_rows),
        'product_category': np.random.choice(categories, n_rows),
        'order_total': np.random.lognormal(3, 1, n_rows).round(2),
        'order_date': [
            datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 730))
            for _ in range(n_rows)
        ],
        'order_id': [f"ORD_{i:08d}" for i in range(1, n_rows + 1)]
    }
    
    df = pd.DataFrame(data)
    print(f"✅ Generated dataset: {len(df):,} rows, ~{df.memory_usage(deep=True).sum() / 1024**2:.1f}MB")
    return df

# Generate the dataset
transactions_df = generate_sample_data()

# Show basic info about the dataset
print("\n📊 Dataset Overview:")
print(f"Shape: {transactions_df.shape}")
print(f"Date range: {transactions_df['order_date'].min()} to {transactions_df['order_date'].max()}")
print(f"Unique customers: {transactions_df['customer_id'].nunique():,}")
print(f"Unique products: {transactions_df['product_id'].nunique():,}")

# Save for use across different tools
transactions_df.to_csv('transactions.csv', index=False)
transactions_df.to_parquet('transactions.parquet', index=False)

print("\n" + "="*60)
print("🐼 PANDAS APPROACH - Customer Lifetime Value Analysis")
print("="*60)

def pandas_clv_analysis(df):
    """Customer Lifetime Value analysis using Pandas"""
    start_time = time.time()
    
    # Convert to datetime if not already
    df['order_date'] = pd.to_datetime(df['order_date'])
    
    # Complex aggregation with multiple metrics
    clv_data = (df.groupby('customer_id')
                .agg({
                    'order_total': ['sum', 'count', 'mean', 'std'],
                    'order_date': ['min', 'max'],
                    'product_category': 'nunique'
                })
                .round(2))
    
    # Flatten column names
    clv_data.columns = ['total_spent', 'order_count', 'avg_order', 'order_std', 
                        'first_order', 'last_order', 'categories_purchased']
    
    # Calculate days active
    clv_data['days_active'] = (clv_data['last_order'] - clv_data['first_order']).dt.days
    
    # Calculate customer lifetime value score
    clv_data['clv_score'] = (
        clv_data['total_spent'] * 0.4 + 
        clv_data['order_count'] * 10 * 0.3 + 
        clv_data['categories_purchased'] * 20 * 0.3
    )
    
    # Filter active customers (more than 30 days, multiple orders)
    active_customers = clv_data[
        (clv_data['days_active'] > 30) & 
        (clv_data['order_count'] > 1)
    ].copy()
    
    execution_time = time.time() - start_time
    
    return active_customers, execution_time

# Execute Pandas analysis
pandas_result, pandas_time = pandas_clv_analysis(transactions_df)

print(f"⏱️  Execution time: {pandas_time:.2f} seconds")
print(f"📈 Active customers identified: {len(pandas_result):,}")
print(f"💾 Memory usage: ~{pandas_result.memory_usage(deep=True).sum() / 1024**2:.1f}MB")

print("\nTop 10 customers by CLV score:")
print(pandas_result.nlargest(10, 'clv_score')[['total_spent', 'order_count', 'clv_score']])

print("\n" + "="*60)
print("⚡ POLARS APPROACH - Customer Lifetime Value Analysis")
print("="*60)

def polars_clv_analysis(df_path):
    """Customer Lifetime Value analysis using Polars"""
    start_time = time.time()
    
    # Read data with Polars
    df = pl.read_parquet(df_path)
    
    # Polars-style aggregation with method chaining
    clv_data = (df.group_by('customer_id')
                .agg([
                    pl.col('order_total').sum().alias('total_spent'),
                    pl.col('order_total').count().alias('order_count'),
                    pl.col('order_total').mean().alias('avg_order'),
                    pl.col('order_total').std().alias('order_std'),
                    pl.col('order_date').min().alias('first_order'),
                    pl.col('order_date').max().alias('last_order'),
                    pl.col('product_category').n_unique().alias('categories_purchased')
                ])
                .with_columns([
                    (pl.col('last_order') - pl.col('first_order')).dt.total_days().alias('days_active')
                ])
                .with_columns([
                    (pl.col('total_spent') * 0.4 + 
                     pl.col('order_count') * 10 * 0.3 + 
                     pl.col('categories_purchased') * 20 * 0.3).alias('clv_score')
                ])
                .filter(
                    (pl.col('days_active') > 30) & 
                    (pl.col('order_count') > 1)
                ))
    
    execution_time = time.time() - start_time
    
    return clv_data, execution_time

# Execute Polars analysis
polars_result, polars_time = polars_clv_analysis('transactions.parquet')

print(f"⏱️  Execution time: {polars_time:.2f} seconds")
print(f"📈 Active customers identified: {polars_result.height:,}")
print(f"⚡ Performance improvement: {pandas_time/polars_time:.1f}x faster than Pandas")

print("\nTop 10 customers by CLV score:")
print(polars_result.sort('clv_score', descending=True).head(10).select(['total_spent', 'order_count', 'clv_score']))

print("\n" + "="*60)
print("🦆 DUCKDB APPROACH - Customer Lifetime Value Analysis")
print("="*60)

def duckdb_clv_analysis():
    """Customer Lifetime Value analysis using DuckDB SQL"""
    start_time = time.time()
    
    # Connect to DuckDB and register the parquet file
    conn = duckdb.connect()
    
    # DuckDB can directly query parquet files
    query = """
    SELECT 
        customer_id,
        SUM(order_total) as total_spent,
        COUNT(*) as order_count,
        AVG(order_total) as avg_order,
        STDDEV(order_total) as order_std,
        MIN(order_date) as first_order,
        MAX(order_date) as last_order,
        COUNT(DISTINCT product_category) as categories_purchased,
        DATE_DIFF('day', MIN(order_date), MAX(order_date)) as days_active,
        (SUM(order_total) * 0.4 + 
         COUNT(*) * 10 * 0.3 + 
         COUNT(DISTINCT product_category) * 20 * 0.3) as clv_score
    FROM 'transactions.parquet' 
    GROUP BY customer_id
    HAVING days_active > 30 AND order_count > 1
    ORDER BY clv_score DESC
    """
    
    result = conn.execute(query).fetchdf()
    execution_time = time.time() - start_time
    
    return result, execution_time

# Execute DuckDB analysis
duckdb_result, duckdb_time = duckdb_clv_analysis()

print(f"⏱️  Execution time: {duckdb_time:.2f} seconds")
print(f"📈 Active customers identified: {len(duckdb_result):,}")
print(f"🚀 Performance improvement: {pandas_time/duckdb_time:.1f}x faster than Pandas")

print("\nTop 10 customers by CLV score:")
print(duckdb_result.head(10)[['total_spent', 'order_count', 'clv_score']])

print("\n" + "="*60)
print("📊 PERFORMANCE COMPARISON")
print("="*60)

# Create performance comparison
performance_data = {
    'Tool': ['Pandas', 'Polars', 'DuckDB'],
    'Execution Time (s)': [pandas_time, polars_time, duckdb_time],
    'Speed vs Pandas': [1.0, pandas_time/polars_time, pandas_time/duckdb_time],
    'Memory Efficiency': ['Baseline', 'Better', 'Good'],
    'Learning Curve': ['Easy', 'Moderate', 'Easy (SQL)']
}

performance_df = pd.DataFrame(performance_data)
print(performance_df.to_string(index=False))

# Visualize performance comparison
plt.figure(figsize=(12, 8))

# Execution time comparison
plt.subplot(2, 2, 1)
bars = plt.bar(performance_data['Tool'], performance_data['Execution Time (s)'], 
               color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.title('Execution Time Comparison')
plt.ylabel('Seconds')
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'{performance_data["Execution Time (s)"][i]:.2f}s', 
             ha='center', va='bottom')

# Speed improvement comparison
plt.subplot(2, 2, 2)
bars = plt.bar(performance_data['Tool'], performance_data['Speed vs Pandas'], 
               color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.title('Speed Improvement vs Pandas')
plt.ylabel('Times Faster')
plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Pandas Baseline')
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'{performance_data["Speed vs Pandas"][i]:.1f}x', 
             ha='center', va='bottom')

# Customer distribution analysis
plt.subplot(2, 2, 3)
bins = np.logspace(0, 4, 30)
plt.hist(pandas_result['total_spent'], bins=bins, alpha=0.7, label='Customer Spend Distribution')
plt.xscale('log')
plt.xlabel('Total Spent ($)')
plt.ylabel('Number of Customers')
plt.title('Customer Lifetime Value Distribution')
plt.grid(True, alpha=0.3)

# Category analysis
plt.subplot(2, 2, 4)
category_counts = transactions_df['product_category'].value_counts()
plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
plt.title('Product Category Distribution')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("🎯 SCENARIO 1 CONCLUSIONS")
print("="*60)

print(f"""
For the Jupyter Notebook Data Scientist scenario with {len(transactions_df):,} transactions:

🐼 PANDAS:
   ✅ Familiar API, rich ecosystem
   ✅ Excellent for visualization and exploration
   ❌ Highest memory usage (~{pandas_result.memory_usage(deep=True).sum() / 1024**2:.0f}MB)
   ❌ Slowest execution ({pandas_time:.2f}s)
   
⚡ POLARS:
   ✅ {pandas_time/polars_time:.1f}x faster than Pandas
   ✅ Lower memory footprint
   ✅ Expressive API similar to Pandas
   ❌ Smaller ecosystem, some learning curve
   
🦆 DUCKDB:
   ✅ {pandas_time/duckdb_time:.1f}x faster than Pandas
   ✅ SQL familiarity
   ✅ Excellent for analytical queries
   ✅ Direct parquet file querying
   ❌ Less flexibility for complex transformations

WINNER FOR THIS SCENARIO: DuckDB for SQL-comfortable teams, Polars for performance-focused Python teams
""")

print("\n🚀 Next Steps:")
print("- Try modifying the CLV calculation formula")
print("- Experiment with different customer segmentation criteria")
print("- Add time-based analysis (monthly trends, seasonality)")
print("- Integrate with your favorite visualization library")
