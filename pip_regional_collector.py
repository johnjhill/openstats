#!/usr/bin/env python3
"""
PIP Regional Data Collector
Collect Personal Independence Payment data with regional breakdowns for time-series analysis
"""

import pandas as pd
import requests
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PIPRegionalCollector:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://stat-xplore.dwp.gov.uk/webapi/rest/v1"
        self.headers = {
            'APIKey': api_key,
            'Content-Type': 'application/json'
        }
        self.rate_limit_delay = 2.0

    def rate_limit(self):
        """Apply rate limiting between requests"""
        time.sleep(self.rate_limit_delay)

    def collect_pip_regional_data(self, database_id: str, measure_id: str,
                                 geographic_field_id: str, time_field_id: str,
                                 collection_name: str) -> pd.DataFrame:
        """Collect PIP data with regional and time breakdown"""

        logger.info(f"🗺️ Collecting {collection_name}...")

        url = f"{self.base_url}/table"
        payload = {
            "database": database_id,
            "measures": [measure_id],
            "dimensions": [geographic_field_id, time_field_id],
            "recodes": {}
        }

        try:
            self.rate_limit()
            logger.info(f"🔄 Querying {database_id} with regional breakdown...")
            response = requests.post(url, json=payload, headers=self.headers)

            if response.status_code == 200:
                data = response.json()
                return self.convert_pip_response_to_dataframe(data, collection_name)
            else:
                try:
                    error_data = response.json()
                    logger.error(f"API error {response.status_code}: {error_data.get('message', response.text)}")
                except:
                    logger.error(f"API error {response.status_code}: {response.text}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error collecting {collection_name}: {e}")
            return pd.DataFrame()

    def convert_pip_response_to_dataframe(self, response: Dict, collection_name: str) -> pd.DataFrame:
        """Convert PIP API response to pandas DataFrame"""

        records = []

        if 'cubes' not in response:
            logger.warning(f"No cubes in response for {collection_name}")
            return pd.DataFrame()

        for cube_id, cube_data in response['cubes'].items():
            if 'values' not in cube_data:
                continue

            # Get dimension info
            dimensions = cube_data.get('dimensions', [])
            geo_dimension = None
            time_dimension = None

            for dim in dimensions:
                field_id = dim.get('field', '')
                if any(geo_term in field_id.lower() for geo_term in ['coa', 'ward', 'pcon', 'geography']):
                    geo_dimension = dim
                elif any(time_term in field_id.lower() for time_term in ['date', 'time']):
                    time_dimension = dim

            if not geo_dimension or not time_dimension:
                logger.warning(f"Missing geo or time dimension in {collection_name}")
                continue

            # Extract data values
            values = cube_data['values']
            geo_items = geo_dimension.get('items', [])
            time_items = time_dimension.get('items', [])

            # Build records from the data matrix
            for i, geo_item in enumerate(geo_items):
                # Extract geographic info
                geo_codes = geo_item.get('codes', [])
                geo_labels = geo_item.get('labels', [])
                geo_code = geo_codes[0] if geo_codes else f'GEO_{i}'
                geo_name = geo_labels[0] if geo_labels else f'Area_{i}'

                for j, time_item in enumerate(time_items):
                    # Extract time info
                    time_codes = time_item.get('codes', [])
                    time_labels = time_item.get('labels', [])
                    time_code = time_codes[0] if time_codes else f'TIME_{j}'
                    time_period = time_labels[0] if time_labels else f'Period_{j}'

                    # Calculate index in values array (row-major order)
                    value_index = i * len(time_items) + j
                    value = values[value_index] if value_index < len(values) else None

                    if value is not None and value > 0:  # Only include non-zero values
                        records.append({
                            'Geography_Code': geo_code,
                            'Geography_Name': geo_name,
                            'Time_Period': time_period,
                            'Time_Code': time_code,
                            'Value': value,
                            'Dataset': collection_name,
                            'Measure': cube_id,
                            'Benefit_Type': 'PIP'
                        })

        df = pd.DataFrame(records)

        if not df.empty:
            # Process dates
            df['Date'] = pd.to_datetime(df['Time_Period'], format='%Y %b', errors='coerce')

            # Add helpful columns
            if 'Date' in df.columns:
                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month
                df['Quarter'] = df['Date'].dt.quarter
                df['Month_Name'] = df['Date'].dt.month_name()

            logger.info(f"✅ {collection_name}: {len(df)} records collected")

            # Log summary stats
            if len(df) > 0:
                logger.info(f"   📍 Geographic areas: {df['Geography_Code'].nunique()}")
                logger.info(f"   📅 Time periods: {df['Date'].nunique()}")
                logger.info(f"   📊 Total PIP cases: {df['Value'].sum():,.0f}")
                logger.info(f"   📅 Date range: {df['Date'].min()} to {df['Date'].max()}")

        return df

    def collect_comprehensive_pip_regional_data(self) -> Dict[str, pd.DataFrame]:
        """Collect comprehensive regional PIP data for time-series analysis"""

        logger.info("🚀 Starting comprehensive PIP regional data collection...")

        # Define collection configurations for different geographic levels and datasets
        pip_configs = [
            {
                'database': 'str:database:PIP_Monthly_new',
                'measure': 'str:count:PIP_Monthly_new:V_F_PIP_MONTHLY',
                'geographic_field': 'str:field:PIP_Monthly_new:V_F_PIP_MONTHLY:COA_2011',
                'time_field': 'str:field:PIP_Monthly_new:F_PIP_DATE:DATE2',
                'name': 'pip_output_areas_2019_onwards',
                'description': 'PIP cases by Output Areas from 2019'
            },
            {
                'database': 'str:database:PIP_Monthly_new',
                'measure': 'str:count:PIP_Monthly_new:V_F_PIP_MONTHLY',
                'geographic_field': 'str:field:PIP_Monthly_new:V_F_PIP_MONTHLY:WARD_CODE',
                'time_field': 'str:field:PIP_Monthly_new:F_PIP_DATE:DATE2',
                'name': 'pip_wards_2019_onwards',
                'description': 'PIP cases by Wards from 2019'
            },
            {
                'database': 'str:database:PIP_Monthly_new',
                'measure': 'str:count:PIP_Monthly_new:V_F_PIP_MONTHLY',
                'geographic_field': 'str:field:PIP_Monthly_new:V_F_PIP_MONTHLY:PCON24',
                'time_field': 'str:field:PIP_Monthly_new:F_PIP_DATE:DATE2',
                'name': 'pip_constituencies_2024_2019_onwards',
                'description': 'PIP cases by Parliamentary Constituencies (2024 boundaries) from 2019'
            },
            {
                'database': 'str:database:PIP_Monthly_new',
                'measure': 'str:measure:PIP_Monthly_new:V_F_PIP_MONTHLY:PIP_AWARD_AMOUNT',
                'geographic_field': 'str:field:PIP_Monthly_new:V_F_PIP_MONTHLY:WARD_CODE',
                'time_field': 'str:field:PIP_Monthly_new:F_PIP_DATE:DATE2',
                'name': 'pip_financial_awards_wards_2019_onwards',
                'description': 'PIP financial awards by Wards from 2019'
            },
            # Try the older dataset for historical comparison
            {
                'database': 'str:database:PIP_Monthly',
                'measure': 'str:count:PIP_Monthly:V_F_PIP_MONTHLY',
                'geographic_field': 'str:field:PIP_Monthly:V_F_PIP_MONTHLY:WARD_CODE',
                'time_field': 'str:field:PIP_Monthly:F_PIP_DATE:DATE2',
                'name': 'pip_wards_historical_to_2019',
                'description': 'PIP cases by Wards (historical to 2019)'
            }
        ]

        results = {}

        for config in pip_configs:
            logger.info(f"📊 Collecting: {config['description']}")

            df = self.collect_pip_regional_data(
                config['database'],
                config['measure'],
                config['geographic_field'],
                config['time_field'],
                config['name']
            )

            if not df.empty:
                results[config['name']] = df
            else:
                logger.warning(f"❌ No data collected for {config['name']}")

        return results

def main():
    """Run PIP regional data collection"""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv('STAT_XPLORE_API_KEY')
    if not api_key:
        logger.error("API key not found in environment variables")
        return

    collector = PIPRegionalCollector(api_key)

    # Collect all regional PIP data
    pip_datasets = collector.collect_comprehensive_pip_regional_data()

    if not pip_datasets:
        logger.error("No PIP regional data was collected")
        return

    # Save results
    output_dir = "pip_regional_data"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    saved_files = {}

    for name, df in pip_datasets.items():
        if not df.empty:
            # Save as parquet
            filename = f"{output_dir}/pip_{name}_{timestamp}.parquet"
            df.to_parquet(filename, index=False)
            saved_files[name] = filename

            # Save as CSV for inspection
            csv_filename = f"{output_dir}/pip_{name}_{timestamp}.csv"
            df.to_csv(csv_filename, index=False)

            logger.info(f"💾 Saved {name}: {len(df)} records")

    # Create combined regional dataset
    if pip_datasets:
        combined_df = pd.concat(pip_datasets.values(), ignore_index=True)

        # Add growth rate calculations
        if not combined_df.empty and 'Date' in combined_df.columns:
            combined_df = combined_df.sort_values(['Geography_Code', 'Dataset', 'Date'])
            combined_df['Previous_Value'] = combined_df.groupby(['Geography_Code', 'Dataset'])['Value'].shift(1)
            combined_df['Growth_Rate'] = ((combined_df['Value'] - combined_df['Previous_Value']) / combined_df['Previous_Value'] * 100).round(2)
            combined_df['Growth_Rate'] = combined_df['Growth_Rate'].replace([float('inf'), float('-inf')], None)

        combined_filename = f"{output_dir}/pip_regional_combined_{timestamp}.parquet"
        combined_df.to_parquet(combined_filename, index=False)

        csv_filename = f"{output_dir}/pip_regional_combined_{timestamp}.csv"
        combined_df.to_csv(csv_filename, index=False)

        # Calculate summary statistics
        total_records = len(combined_df)
        total_areas = combined_df['Geography_Code'].nunique()
        total_cases = combined_df['Value'].sum()
        date_range = f"{combined_df['Date'].min().strftime('%Y-%m')} to {combined_df['Date'].max().strftime('%Y-%m')}"
        datasets = combined_df['Dataset'].nunique()

        print(f"\n🎉 PIP REGIONAL DATA COLLECTION COMPLETE!")
        print(f"📂 Files saved to: {output_dir}/")
        print(f"📊 Total records: {total_records:,}")
        print(f"🗺️ Geographic areas: {total_areas:,}")
        print(f"📅 Date range: {date_range}")
        print(f"📈 Datasets: {datasets}")
        print(f"💯 Total PIP cases: {total_cases:,.0f}")

        print(f"\n📋 Individual datasets collected:")
        for name, df in pip_datasets.items():
            print(f"  - {name}: {len(df):,} records")

        print(f"\n🎯 Use this data for:")
        print(f"  - Regional PIP trend analysis")
        print(f"  - Geographic inequality studies")
        print(f"  - Policy impact assessment")
        print(f"  - Time-series forecasting")

        return combined_filename
    else:
        logger.warning("No PIP regional data to combine")
        return None

if __name__ == "__main__":
    main()