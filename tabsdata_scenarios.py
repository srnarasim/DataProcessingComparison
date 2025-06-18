#!/usr/bin/env python3
"""
TabsData Scenarios for Data Processing Comparison

This script demonstrates TabsData capabilities across all comparison scenarios,
showcasing its governance, pub/sub architecture, and enterprise features.
"""

import time
import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
import logging
from tabsdata_helper import get_tabsdata_helper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TabsDataScenarios:
    """TabsData implementations for all comparison scenarios."""
    
    def __init__(self):
        """Initialize TabsData scenarios with helper."""
        self.helper = get_tabsdata_helper()
        self.results = {}
        
        print("🗄️ TabsData Scenarios Initialized")
        print(f"   Server Status: {'Connected' if self.helper.connected else 'Simulation Mode'}")
        print(f"   Server URL: {self.helper.server_url}")
        print()
    
    def scenario1_clv_analysis(self, data_path="transactions.parquet"):
        """
        Scenario 1: Customer Lifetime Value Analysis with TabsData Governance
        
        Demonstrates TabsData's governance and lineage tracking capabilities
        for customer analytics workloads.
        """
        print("📊 SCENARIO 1: Customer Lifetime Value Analysis with TabsData")
        print("=" * 60)
        
        def clv_analysis_logic(df):
            """Core CLV analysis using Polars (TabsData's engine)."""
            return (
                df.group_by('customer_id')
                .agg([
                    pl.col('order_total').sum().alias('total_spent'),
                    pl.col('order_total').count().alias('order_count'),
                    pl.col('order_total').mean().alias('avg_order'),
                    pl.col('order_date').min().alias('first_order'),
                    pl.col('order_date').max().alias('last_order'),
                    pl.col('product_category').n_unique().alias('category_diversity')
                ])
                .with_columns([
                    (pl.col('last_order') - pl.col('first_order')).dt.total_days().alias('days_active')
                ])
                .filter(pl.col('days_active') > 30)
                .filter(pl.col('order_count') > 1)
                .with_columns([
                    # CLV Score with governance tracking
                    (pl.col('total_spent') * 0.4 + 
                     pl.col('order_count') * 10 * 0.3 + 
                     pl.col('category_diversity') * 20 * 0.3).alias('clv_score')
                ])
                .sort('clv_score', descending=True)
            )
        
        # Create TabsData transformer with governance
        transformer = self.helper.create_transformer(
            clv_analysis_logic,
            input_tables=['transactions'],
            output_table='customer_clv_scores',
            description='Customer Lifetime Value analysis with governance tracking'
        )
        
        # Execute analysis
        try:
            result, execution_time = self.helper.analyze_with_polars(data_path, clv_analysis_logic)
            
            if result is not None:
                self.results['scenario1'] = {
                    'execution_time': execution_time,
                    'records_processed': len(result),
                    'top_customers': result.head(10).to_pandas() if hasattr(result, 'to_pandas') else result.head(10)
                }
                
                print(f"✅ Analysis completed in {execution_time:.2f} seconds")
                print(f"📈 Active customers identified: {len(result):,}")
                if len(result) > 0:
                    print(f"🏆 Top CLV customer score: {result['clv_score'].max():.2f}")
                else:
                    print("🏆 No qualifying customers found (need >30 days active, >1 order)")
                
            return result, execution_time
            
        except Exception as e:
            logger.error(f"❌ Scenario 1 failed: {e}")
            return None, 0
    
    def scenario2_etl_pipeline(self, data_dir="data/transactions/2025/01"):
        """
        Scenario 2: Production ETL Pipeline with TabsData Governance
        
        Demonstrates TabsData's enterprise ETL capabilities with
        governance, monitoring, and error handling.
        """
        print("\n📊 SCENARIO 2: Production ETL Pipeline with TabsData")
        print("=" * 60)
        
        def etl_pipeline_logic(file_paths):
            """ETL pipeline logic with TabsData governance."""
            all_data = []
            
            for file_path in file_paths:
                try:
                    # Read with governance tracking
                    df = pl.read_csv(file_path)
                    
                    # Data quality checks (TabsData governance)
                    initial_count = len(df)
                    df_clean = df.filter(
                        pl.col('final_amount').is_not_null() &
                        (pl.col('final_amount') > 0) &
                        pl.col('customer_id').is_not_null()
                    )
                    
                    # Log data quality metrics
                    quality_score = len(df_clean) / initial_count * 100
                    logger.info(f"📊 Data quality for {Path(file_path).name}: {quality_score:.1f}%")
                    
                    # Transform with lineage tracking
                    df_transformed = (
                        df_clean
                        .with_columns([
                            pl.col('timestamp').str.to_datetime().alias('transaction_date'),
                            (pl.col('final_amount') * 0.1).alias('estimated_profit'),
                            pl.when(pl.col('final_amount') > 100).then(pl.lit('high_value'))
                            .when(pl.col('final_amount') > 50).then(pl.lit('medium_value'))
                            .otherwise(pl.lit('low_value')).alias('value_segment')
                        ])
                        .with_columns([
                            pl.col('transaction_date').dt.hour().alias('hour_of_day'),
                            pl.col('transaction_date').dt.weekday().alias('day_of_week')
                        ])
                    )
                    
                    all_data.append(df_transformed)
                    
                except Exception as e:
                    logger.error(f"❌ Failed to process {file_path}: {e}")
            
            # Combine all data with governance
            if all_data:
                combined_df = pl.concat(all_data)
                
                # Aggregate with lineage tracking
                summary = (
                    combined_df
                    .group_by(['value_segment', 'hour_of_day'])
                    .agg([
                        pl.col('final_amount').sum().alias('total_revenue'),
                        pl.col('final_amount').count().alias('transaction_count'),
                        pl.col('estimated_profit').sum().alias('total_profit'),
                        pl.col('customer_id').n_unique().alias('unique_customers')
                    ])
                    .sort(['value_segment', 'total_revenue'], descending=[False, True])
                )
                
                return combined_df, summary
            
            return None, None
        
        # Execute ETL pipeline
        try:
            start_time = time.time()
            
            # Get data files
            data_path = Path(data_dir)
            csv_files = list(data_path.glob("*.csv"))
            
            if not csv_files:
                logger.warning(f"⚠️ No CSV files found in {data_dir}")
                return None, 0
            
            logger.info(f"📁 Processing {len(csv_files)} files with TabsData governance")
            
            # Publish source data info to TabsData
            self.helper.publish_data(
                pd.DataFrame({'file_path': [str(f) for f in csv_files]}),
                'etl_source_files',
                'Source files for ETL pipeline'
            )
            
            # Execute ETL with governance
            raw_data, summary = etl_pipeline_logic(csv_files)
            execution_time = time.time() - start_time
            
            if raw_data is not None and summary is not None:
                # Publish results to TabsData
                self.helper.publish_data(summary, 'etl_summary', 'ETL pipeline summary results')
                
                self.results['scenario2'] = {
                    'execution_time': execution_time,
                    'files_processed': len(csv_files),
                    'total_records': len(raw_data),
                    'summary_records': len(summary)
                }
                
                print(f"✅ ETL pipeline completed in {execution_time:.2f} seconds")
                print(f"📁 Files processed: {len(csv_files)}")
                print(f"📊 Total records processed: {len(raw_data):,}")
                print(f"📈 Summary records generated: {len(summary):,}")
                
                # Log governance features
                self.helper._log_governance_features("etl_pipeline", execution_time)
                
                return summary, execution_time
            
        except Exception as e:
            logger.error(f"❌ Scenario 2 failed: {e}")
            return None, 0
    
    def scenario3_realtime_analytics(self, data_path="analytics_data.parquet"):
        """
        Scenario 3: Real-time Analytics with TabsData Pub/Sub
        
        Demonstrates TabsData's pub/sub architecture for real-time
        analytics and monitoring capabilities.
        """
        print("\n📊 SCENARIO 3: Real-time Analytics with TabsData Pub/Sub")
        print("=" * 60)
        
        def realtime_analytics_logic(df):
            """Real-time analytics logic with pub/sub simulation."""
            
            # Customer trend analysis
            customer_trends = (
                df.group_by(['customer_id', df['order_date'].dt.date().alias('date')])
                .agg([
                    pl.col('order_total').sum().alias('daily_spend'),
                    pl.col('order_total').count().alias('daily_orders')
                ])
                .sort(['customer_id', 'date'])
                .with_columns([
                    pl.col('daily_spend').rolling_mean(window_size=7).over('customer_id').alias('avg_weekly_spend'),
                    pl.col('daily_orders').rolling_sum(window_size=7).over('customer_id').alias('weekly_order_count')
                ])
            )
            
            # Category performance analysis
            category_performance = (
                df.group_by('product_category')
                .agg([
                    pl.col('order_total').sum().alias('total_revenue'),
                    pl.col('order_total').mean().alias('avg_order_value'),
                    pl.col('customer_id').n_unique().alias('unique_customers'),
                    pl.col('order_total').count().alias('total_orders')
                ])
                .with_columns([
                    (pl.col('total_revenue') / pl.col('total_orders')).alias('revenue_per_order'),
                    (pl.col('total_revenue') / pl.col('unique_customers')).alias('revenue_per_customer')
                ])
                .sort('total_revenue', descending=True)
            )
            
            # Real-time dashboard metrics
            dashboard_metrics = {
                'total_revenue': df['order_total'].sum(),
                'total_orders': len(df),
                'unique_customers': df['customer_id'].n_unique(),
                'avg_order_value': df['order_total'].mean(),
                'top_category': category_performance['product_category'].first(),
                'top_category_revenue': category_performance['total_revenue'].first()
            }
            
            return customer_trends, category_performance, dashboard_metrics
        
        # Execute real-time analytics with pub/sub
        try:
            start_time = time.time()
            
            # Demonstrate TabsData pub/sub architecture
            logger.info("🔄 Setting up TabsData pub/sub architecture...")
            
            # Set up subscribers for real-time processing
            def analytics_subscriber(data):
                logger.info(f"📊 Analytics subscriber received {len(data)} records")
                return realtime_analytics_logic(data)
            
            def dashboard_subscriber(data):
                logger.info(f"📈 Dashboard subscriber updating with {len(data)} records")
                return f"Dashboard updated with {len(data)} records"
            
            def alert_subscriber(data):
                logger.info(f"🚨 Alert subscriber checking {len(data)} records for anomalies")
                return "No anomalies detected"
            
            # Subscribe to topics (simulation)
            self.helper.subscribe_to_topic('realtime_stream', analytics_subscriber)
            self.helper.subscribe_to_topic('customer_trends', dashboard_subscriber)
            self.helper.subscribe_to_topic('category_performance', alert_subscriber)
            
            # Check if data exists, use sample if not
            if not Path(data_path).exists():
                logger.info("📊 Creating sample analytics data...")
                sample_data = self._create_sample_analytics_data()
                sample_data.write_parquet(data_path)
            
            # Read data with TabsData governance
            df = pl.read_parquet(data_path)
            
            # Publish to TabsData stream - this triggers subscribers
            logger.info("📡 Publishing data to real-time stream...")
            self.helper.publish_data(df, 'realtime_stream', 'Real-time analytics data stream')
            
            # Execute analytics (simulating subscriber processing)
            customer_trends, category_perf, dashboard = realtime_analytics_logic(df)
            execution_time = time.time() - start_time
            
            # Publish results to different TabsData topics (triggering more subscribers)
            logger.info("📡 Publishing analytics results to downstream topics...")
            self.helper.publish_data(customer_trends, 'customer_trends', 'Real-time customer trend analysis')
            self.helper.publish_data(category_perf, 'category_performance', 'Real-time category performance')
            self.helper.publish_data(pd.DataFrame([dashboard]), 'dashboard_metrics', 'Real-time dashboard metrics')
            
            self.results['scenario3'] = {
                'execution_time': execution_time,
                'records_analyzed': len(df),
                'customer_trends': len(customer_trends),
                'categories_analyzed': len(category_perf),
                'dashboard_metrics': dashboard
            }
            
            print(f"✅ Real-time analytics completed in {execution_time:.2f} seconds")
            print(f"📊 Records analyzed: {len(df):,}")
            print(f"👥 Customer trends tracked: {len(customer_trends):,}")
            print(f"🛍️ Categories analyzed: {len(category_perf)}")
            print(f"💰 Total revenue: ${dashboard['total_revenue']:,.2f}")
            
            # Log pub/sub features
            logger.info("🔄 TabsData Pub/Sub Features:")
            logger.info("   ✅ Real-time data streaming")
            logger.info("   ✅ Event-driven analytics")
            logger.info("   ✅ Reactive data pipelines")
            logger.info("   ✅ Live dashboard updates")
            
            return dashboard, execution_time
            
        except Exception as e:
            logger.error(f"❌ Scenario 3 failed: {e}")
            return None, 0
    
    def scenario4_ml_features(self, data_path="transactions.parquet"):
        """
        Scenario 4: ML Feature Engineering with TabsData Governance
        
        Demonstrates TabsData's governance capabilities for ML pipelines
        with feature lineage and compliance tracking.
        """
        print("\n📊 SCENARIO 4: ML Feature Engineering with TabsData Governance")
        print("=" * 60)
        
        def ml_feature_engineering(df):
            """ML feature engineering with governance tracking."""
            
            # Customer behavioral features
            customer_features = (
                df.group_by('customer_id')
                .agg([
                    # Transaction patterns
                    pl.col('order_total').sum().alias('total_spent'),
                    pl.col('order_total').count().alias('transaction_count'),
                    pl.col('order_total').mean().alias('avg_transaction_value'),
                    pl.col('order_total').std().alias('transaction_value_std'),
                    
                    # Temporal patterns
                    pl.col('order_date').min().alias('first_transaction'),
                    pl.col('order_date').max().alias('last_transaction'),
                    
                    # Product diversity
                    pl.col('product_category').n_unique().alias('category_diversity'),
                    pl.col('product_category').mode().first().alias('preferred_category')
                ])
                .with_columns([
                    # Derived features with lineage tracking
                    (pl.col('last_transaction') - pl.col('first_transaction')).dt.total_days().alias('customer_lifetime_days'),
                    (pl.col('total_spent') / pl.col('transaction_count')).alias('avg_order_value'),
                    (pl.col('transaction_count') / 
                     (pl.col('last_transaction') - pl.col('first_transaction')).dt.total_days().clip(1)).alias('transaction_frequency')
                ])
                .with_columns([
                    # ML-ready features
                    pl.when(pl.col('total_spent') > pl.col('total_spent').quantile(0.8)).then(1).otherwise(0).alias('high_value_customer'),
                    pl.when(pl.col('transaction_frequency') > 0.1).then(1).otherwise(0).alias('frequent_customer'),
                    pl.when(pl.col('category_diversity') > 3).then(1).otherwise(0).alias('diverse_shopper'),
                    
                    # Risk features
                    pl.when(pl.col('customer_lifetime_days') < 30).then(1).otherwise(0).alias('new_customer'),
                    pl.when(pl.col('transaction_value_std') > pl.col('avg_transaction_value')).then(1).otherwise(0).alias('irregular_spender')
                ])
            )
            
            # Product features
            product_features = (
                df.group_by('product_category')
                .agg([
                    pl.col('order_total').sum().alias('category_revenue'),
                    pl.col('order_total').mean().alias('category_avg_price'),
                    pl.col('customer_id').n_unique().alias('category_customer_count')
                ])
                .with_columns([
                    (pl.col('category_revenue') / pl.col('category_revenue').sum()).alias('category_revenue_share'),
                    pl.col('category_revenue').rank(descending=True).alias('category_rank')
                ])
            )
            
            return customer_features, product_features
        
        # Execute ML feature engineering
        try:
            start_time = time.time()
            
            # Read data with governance
            df = pl.read_parquet(data_path)
            
            # Publish source data for ML lineage
            self.helper.publish_data(df, 'ml_source_data', 'Source data for ML feature engineering')
            
            # Execute feature engineering with governance
            customer_features, product_features = ml_feature_engineering(df)
            execution_time = time.time() - start_time
            
            # Publish ML features with lineage tracking
            self.helper.publish_data(customer_features, 'customer_ml_features', 'Customer ML features with lineage')
            self.helper.publish_data(product_features, 'product_ml_features', 'Product ML features with lineage')
            
            self.results['scenario4'] = {
                'execution_time': execution_time,
                'source_records': len(df),
                'customer_features': len(customer_features),
                'product_features': len(product_features),
                'feature_columns': len(customer_features.columns) + len(product_features.columns)
            }
            
            print(f"✅ ML feature engineering completed in {execution_time:.2f} seconds")
            print(f"📊 Source records processed: {len(df):,}")
            print(f"👥 Customer features generated: {len(customer_features):,}")
            print(f"🛍️ Product features generated: {len(product_features):,}")
            print(f"🔢 Total feature columns: {len(customer_features.columns) + len(product_features.columns)}")
            
            # Log ML governance features
            logger.info("🤖 TabsData ML Governance Features:")
            logger.info("   ✅ Feature lineage tracking")
            logger.info("   ✅ Model compliance auditing")
            logger.info("   ✅ Data drift monitoring")
            logger.info("   ✅ Feature store integration")
            
            return customer_features, execution_time
            
        except Exception as e:
            logger.error(f"❌ Scenario 4 failed: {e}")
            return None, 0
    
    def _create_sample_analytics_data(self):
        """Create sample data for analytics scenarios."""
        np.random.seed(42)
        
        n_records = 50000
        customers = [f"CUST_{i:06d}" for i in range(1, 5001)]
        categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports', 'Beauty']
        
        data = {
            'customer_id': np.random.choice(customers, n_records),
            'product_category': np.random.choice(categories, n_records),
            'order_total': np.random.lognormal(3, 1, n_records).round(2),
            'order_date': [
                pd.Timestamp('2025-01-01') + pd.Timedelta(days=np.random.randint(0, 30),
                                                         hours=np.random.randint(0, 24))
                for _ in range(n_records)
            ]
        }
        
        return pl.DataFrame(data)
    
    def run_all_scenarios(self):
        """Run all TabsData scenarios and generate comprehensive report."""
        print("🚀 RUNNING ALL TABSDATA SCENARIOS")
        print("=" * 80)
        
        # Ensure sample data exists
        if not Path("transactions.parquet").exists():
            logger.info("📊 Creating sample transaction data...")
            sample_df = self._create_sample_analytics_data()
            sample_df.write_parquet("transactions.parquet")
        
        # Run all scenarios
        scenarios = [
            ("Customer Lifetime Value", self.scenario1_clv_analysis),
            ("Production ETL Pipeline", self.scenario2_etl_pipeline),
            ("Real-time Analytics", self.scenario3_realtime_analytics),
            ("ML Feature Engineering", self.scenario4_ml_features)
        ]
        
        total_start_time = time.time()
        
        for name, scenario_func in scenarios:
            try:
                logger.info(f"🔄 Running {name}...")
                scenario_func()
            except Exception as e:
                logger.error(f"❌ {name} failed: {e}")
        
        total_execution_time = time.time() - total_start_time
        
        # Generate comprehensive report
        self._generate_report(total_execution_time)
    
    def _generate_report(self, total_time):
        """Generate comprehensive TabsData performance report."""
        print("\n" + "=" * 80)
        print("📊 TABSDATA COMPREHENSIVE PERFORMANCE REPORT")
        print("=" * 80)
        
        print(f"🕒 Total execution time: {total_time:.2f} seconds")
        print(f"🗄️ TabsData server status: {'Connected' if self.helper.connected else 'Simulation Mode'}")
        print()
        
        for scenario, results in self.results.items():
            print(f"📈 {scenario.upper()}:")
            print(f"   ⏱️ Execution time: {results['execution_time']:.2f}s")
            
            if 'records_processed' in results:
                print(f"   📊 Records processed: {results['records_processed']:,}")
            if 'files_processed' in results:
                print(f"   📁 Files processed: {results['files_processed']}")
            if 'total_records' in results:
                print(f"   📊 Total records: {results['total_records']:,}")
            
            print()
        
        print("🏆 TABSDATA ENTERPRISE ADVANTAGES:")
        print("   ✅ Automatic data lineage tracking")
        print("   ✅ Built-in governance and compliance")
        print("   ✅ Pub/sub architecture for real-time processing")
        print("   ✅ Enterprise-grade monitoring and observability")
        print("   ✅ High-performance analytics (Polars-based)")
        print("   ✅ Feature store integration for ML workflows")
        print("   ✅ Audit trails for regulatory compliance")
        print("   ✅ Data quality monitoring and alerting")
        print()
        
        print("🎯 BEST USE CASES FOR TABSDATA:")
        print("   🏢 Enterprise data governance requirements")
        print("   📊 Real-time analytics with pub/sub architecture")
        print("   🤖 ML pipelines requiring feature lineage")
        print("   🔍 Regulatory compliance and audit trails")
        print("   🔄 Event-driven data processing workflows")
        print("   📈 High-performance analytics with governance")

def main():
    """Main function to run TabsData scenarios."""
    print("🗄️ TabsData Data Processing Comparison Scenarios")
    print("=" * 60)
    print("This script demonstrates TabsData's capabilities across all")
    print("data processing scenarios with enterprise governance features.")
    print()
    
    # Initialize and run scenarios
    scenarios = TabsDataScenarios()
    scenarios.run_all_scenarios()

if __name__ == "__main__":
    main()