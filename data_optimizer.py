#!/usr/bin/env python3
"""
Data optimizer for handling large UC datasets efficiently
Combines, optimizes, and prepares data for dashboard
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import logging
from typing import Dict, List, Optional
import pyarrow as pa
import pyarrow.parquet as pq

class UCDataOptimizer:
    def __init__(self):
        """Initialize data optimizer"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def combine_all_collected_data(self, data_dirs: List[str] = None) -> pd.DataFrame:
        """Combine all collected UC data from various sources"""

        if data_dirs is None:
            data_dirs = [
                "uc_dashboard_data",
                "granular_uc_data",
                "comprehensive_uc_data",
                "real_uc_data",
                "test_uc_data"
            ]

        all_files = []

        # Find all parquet files
        for data_dir in data_dirs:
            if os.path.exists(data_dir):
                parquet_files = glob.glob(f"{data_dir}/*.parquet")
                all_files.extend(parquet_files)
                self.logger.info(f"Found {len(parquet_files)} files in {data_dir}")

        self.logger.info(f"Total files found: {len(all_files)}")

        if not all_files:
            self.logger.warning("No data files found")
            return pd.DataFrame()

        # Load and combine all data
        dataframes = []
        total_records = 0

        for file_path in all_files:
            try:
                df = pd.read_parquet(file_path)
                if not df.empty:
                    # Add file source metadata
                    df['Source_File'] = os.path.basename(file_path)
                    df['Collection_Time'] = datetime.fromtimestamp(os.path.getctime(file_path))

                    dataframes.append(df)
                    total_records += len(df)
                    self.logger.info(f"Loaded {file_path}: {len(df)} records")

            except Exception as e:
                self.logger.error(f"Error loading {file_path}: {e}")

        if not dataframes:
            return pd.DataFrame()

        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
        self.logger.info(f"Combined dataset: {len(combined_df)} total records")

        return combined_df

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and data types"""

        if df.empty:
            return df

        # Standard column mapping
        column_mapping = {
            'LSOA_Code': 'Geography_Code',
            'LSOA_Name': 'Geography_Name',
            'UC_Households': 'Value',
            'UC_People': 'Value',
            'Households_Count': 'Value',
            'People_Count': 'Value'
        }

        # Apply mappings
        df = df.rename(columns=column_mapping)

        # Ensure standard columns exist
        required_columns = [
            'Geography_Code', 'Geography_Name', 'Time_Period',
            'Value', 'Dataset', 'Date'
        ]

        for col in required_columns:
            if col not in df.columns:
                if col == 'Geography_Code':
                    df['Geography_Code'] = 'UK_TOTAL'
                elif col == 'Geography_Name':
                    df['Geography_Name'] = 'United Kingdom'
                elif col == 'Dataset':
                    df['Dataset'] = 'unknown'
                else:
                    df[col] = None

        # Standardize data types
        try:
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

            if 'Year' not in df.columns and 'Date' in df.columns:
                df['Year'] = df['Date'].dt.year
            if 'Month' not in df.columns and 'Date' in df.columns:
                df['Month'] = df['Date'].dt.month
            if 'Quarter' not in df.columns and 'Date' in df.columns:
                df['Quarter'] = df['Date'].dt.quarter

        except Exception as e:
            self.logger.warning(f"Error standardizing data types: {e}")

        return df

    def deduplicate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates while preserving the most recent/complete data"""

        if df.empty:
            return df

        # Define key columns for deduplication
        key_columns = ['Geography_Code', 'Time_Period', 'Dataset']

        # Add Category if it exists (for demographic breakdowns)
        if 'Category' in df.columns:
            key_columns.append('Category')

        # Sort by collection time to keep most recent
        if 'Collection_Time' in df.columns:
            df = df.sort_values('Collection_Time', ascending=False)

        # Remove duplicates, keeping first (most recent)
        before_count = len(df)
        df = df.drop_duplicates(subset=key_columns, keep='first')
        after_count = len(df)

        self.logger.info(f"Deduplication: {before_count} -> {after_count} records ({before_count - after_count} duplicates removed)")

        return df

    def optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency"""

        if df.empty:
            return df

        # Optimize string columns
        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')

        # Optimize numeric columns
        for col in df.select_dtypes(include=['float64']).columns:
            if col in ['Value', 'Growth_Rate', 'Previous_Value']:
                # Keep full precision for important measures
                continue
            else:
                # Downcast other floats
                df[col] = pd.to_numeric(df[col], downcast='float')

        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')

        return df

    def create_optimized_dataset(self, output_dir: str = "optimized_uc_data") -> Dict[str, str]:
        """Create optimized, combined dataset for dashboard"""

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.logger.info("🔄 Starting data optimization process...")

        # 1. Combine all data
        self.logger.info("1️⃣ Combining all collected data...")
        combined_df = self.combine_all_collected_data()

        if combined_df.empty:
            self.logger.error("No data to optimize")
            return {}

        # 2. Standardize columns
        self.logger.info("2️⃣ Standardizing columns...")
        combined_df = self.standardize_columns(combined_df)

        # 3. Deduplicate
        self.logger.info("3️⃣ Removing duplicates...")
        combined_df = self.deduplicate_data(combined_df)

        # 4. Optimize data types
        self.logger.info("4️⃣ Optimizing data types...")
        combined_df = self.optimize_data_types(combined_df)

        # 5. Create analytics-ready datasets
        saved_files = {}

        # Main combined dataset
        main_file = f"{output_dir}/uc_optimized_combined_{timestamp}.parquet"
        combined_df.to_parquet(main_file, index=False, compression='snappy')
        saved_files['optimized_combined'] = main_file

        # CSV version for inspection
        csv_file = f"{output_dir}/uc_optimized_combined_{timestamp}.csv"
        combined_df.to_csv(csv_file, index=False)
        saved_files['optimized_csv'] = csv_file

        # Create specialized datasets
        self.logger.info("5️⃣ Creating specialized datasets...")

        # Time series dataset (aggregated)
        if 'Date' in combined_df.columns:
            timeseries_df = combined_df.groupby(['Date', 'Dataset']).agg({
                'Value': 'sum',
                'Geography_Code': 'count'
            }).reset_index()
            timeseries_df.rename(columns={'Geography_Code': 'Area_Count'}, inplace=True)

            timeseries_file = f"{output_dir}/uc_timeseries_{timestamp}.parquet"
            timeseries_df.to_parquet(timeseries_file, index=False)
            saved_files['timeseries'] = timeseries_file

        # Geographic dataset (latest values only)
        if not combined_df.empty:
            latest_date = combined_df['Date'].max()
            geographic_df = combined_df[combined_df['Date'] == latest_date].copy()

            geographic_file = f"{output_dir}/uc_geographic_latest_{timestamp}.parquet"
            geographic_df.to_parquet(geographic_file, index=False)
            saved_files['geographic_latest'] = geographic_file

        # Demographics dataset (if available)
        demo_df = combined_df[combined_df['Category'].notna()] if 'Category' in combined_df.columns else pd.DataFrame()
        if not demo_df.empty:
            demo_file = f"{output_dir}/uc_demographics_{timestamp}.parquet"
            demo_df.to_parquet(demo_file, index=False)
            saved_files['demographics'] = demo_file

        # Log optimization results
        self.logger.info("✅ Data optimization complete!")
        self.logger.info(f"📊 Final dataset statistics:")
        self.logger.info(f"  - Total records: {len(combined_df):,}")
        self.logger.info(f"  - Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
        self.logger.info(f"  - Unique geographies: {combined_df['Geography_Code'].nunique():,}")
        self.logger.info(f"  - Datasets included: {combined_df['Dataset'].nunique()}")
        self.logger.info(f"  - Memory usage: {combined_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

        if 'Category' in combined_df.columns:
            self.logger.info(f"  - Demographic categories: {combined_df['Category'].nunique()}")

        return saved_files

    def create_dashboard_ready_data(self, optimized_file: str, output_dir: str = "dashboard_ready") -> str:
        """Create final dashboard-ready dataset"""

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.logger.info("🎯 Creating dashboard-ready dataset...")

        df = pd.read_parquet(optimized_file)

        # Final processing for dashboard
        # 1. Ensure all required columns
        required_dashboard_columns = [
            'Geography_Code', 'Geography_Name', 'Time_Period', 'Date',
            'Value', 'Dataset', 'Year', 'Month', 'Quarter', 'Growth_Rate'
        ]

        for col in required_dashboard_columns:
            if col not in df.columns:
                if col == 'Growth_Rate':
                    # Calculate growth rate if missing
                    df = df.sort_values(['Geography_Code', 'Dataset', 'Date'])
                    df['Previous_Value'] = df.groupby(['Geography_Code', 'Dataset'])['Value'].shift(1)
                    df['Growth_Rate'] = ((df['Value'] - df['Previous_Value']) / df['Previous_Value'] * 100).round(2)
                    df['Growth_Rate'] = df['Growth_Rate'].replace([np.inf, -np.inf], np.nan)
                    df.drop('Previous_Value', axis=1, inplace=True)

        # 2. Add dashboard-specific computed columns
        df['Value_Millions'] = (df['Value'] / 1_000_000).round(2)
        df['Value_Thousands'] = (df['Value'] / 1_000).round(1)

        # 3. Create period labels
        df['Period_Label'] = df['Date'].dt.strftime('%Y-%m')
        df['Year_Quarter'] = df['Year'].astype(str) + '-Q' + df['Quarter'].astype(str)

        # 4. Final optimization
        df = self.optimize_data_types(df)

        # Save dashboard-ready file
        dashboard_file = f"{output_dir}/uc_dashboard_ready_{timestamp}.parquet"
        df.to_parquet(dashboard_file, index=False, compression='snappy')

        self.logger.info(f"✅ Dashboard-ready dataset created: {dashboard_file}")
        self.logger.info(f"📊 Records: {len(df):,}")
        self.logger.info(f"💾 File size: {os.path.getsize(dashboard_file) / 1024**2:.1f} MB")

        return dashboard_file

def main():
    """Run data optimization"""
    optimizer = UCDataOptimizer()

    # Create optimized dataset
    saved_files = optimizer.create_optimized_dataset()

    if saved_files:
        print("\n🎯 Data Optimization Complete!")
        print("📂 Optimized files created:")
        for file_type, filepath in saved_files.items():
            print(f"  - {file_type}: {filepath}")

        # Create dashboard-ready version
        if 'optimized_combined' in saved_files:
            dashboard_file = optimizer.create_dashboard_ready_data(saved_files['optimized_combined'])
            print(f"\n🚀 Dashboard-ready file: {dashboard_file}")
    else:
        print("❌ No data was optimized")

if __name__ == '__main__':
    main()