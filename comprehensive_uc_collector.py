#!/usr/bin/env python3
"""
Comprehensive UC collector for all datasets, measures, and maximum granularity
This collects from ALL 4 UC databases with ALL available measures
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

class ComprehensiveUCCollector:
    def __init__(self, api_key: str):
        """Initialize comprehensive collector"""
        self.base_url = "https://stat-xplore.dwp.gov.uk/webapi/rest/v1"
        self.headers = {
            "APIKey": api_key,
            "Content-Type": "application/json"
        }
        self.session = requests.Session()

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load schema
        try:
            with open('comprehensive_uc_schema_with_geo.json', 'r') as f:
                self.schema = json.load(f)
        except:
            self.logger.error("Schema file not found")
            self.schema = {}

        # Rate limiting
        self.base_delay = 2.0
        self.max_retries = 5
        self.api_calls = 0

    def _make_request(self, method: str, url: str, **kwargs) -> Optional[Dict]:
        """Make request with rate limiting"""
        time.sleep(self.base_delay)

        for attempt in range(self.max_retries):
            try:
                self.api_calls += 1
                response = self.session.request(method, url, headers=self.headers, **kwargs)

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    wait_time = (2 ** attempt) * self.base_delay * 2
                    self.logger.warning(f"Rate limited. Waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                else:
                    self.logger.error(f"API error {response.status_code}: {response.text}")
                    return None

            except Exception as e:
                self.logger.error(f"Request failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep((2 ** attempt) * self.base_delay)

        return None

    def collect_all_measures_all_databases(self) -> Dict[str, pd.DataFrame]:
        """Collect ALL measures from ALL UC databases"""

        all_data = {}

        for db_id, db_config in self.schema.items():
            db_label = db_config['label']
            self.logger.info(f"🔄 Processing database: {db_label}")

            # Get time field for this database
            time_fields = db_config.get('time_fields', [])
            if not time_fields:
                # Use the 'other_fields' that might be time
                other_fields = db_config.get('other_fields', [])
                time_field_id = None
                for field in other_fields:
                    if 'TIME' in field['label'].upper() or 'PERIOD' in field['label'].upper():
                        time_field_id = field['id']
                        break
                if not time_field_id and other_fields:
                    time_field_id = other_fields[0]['id']  # Use first available
            else:
                time_field_id = time_fields[0]['id']

            if not time_field_id:
                self.logger.warning(f"No time field found for {db_label}")
                continue

            # Collect ALL measures for this database
            measures = db_config.get('measures', [])
            for measure in measures:
                measure_id = measure['id']
                measure_label = measure['label']

                self.logger.info(f"📊 Collecting measure: {measure_label}")

                # Basic query for this measure
                query = {
                    "database": db_id,
                    "measures": [measure_id],
                    "dimensions": [[time_field_id]]
                }

                result = self._make_request('POST', f"{self.base_url}/table", json=query)

                if result:
                    df = self._convert_to_dataframe(result, f"{db_label}_{measure_label}")
                    if df is not None and not df.empty:
                        dataset_key = f"{db_id}_{measure_id}".replace(':', '_').replace(' ', '_')
                        all_data[dataset_key] = df
                        self.logger.info(f"✅ Collected {len(df)} records for {measure_label}")
                    else:
                        self.logger.warning(f"❌ No data for {measure_label}")
                else:
                    self.logger.error(f"❌ Failed to collect {measure_label}")

        return all_data

    def collect_payment_amounts_and_special_measures(self) -> Dict[str, pd.DataFrame]:
        """Collect special measures like payment amounts, RSRS reductions, etc."""

        special_data = {}

        # Households database has payment amounts
        households_config = self.schema.get('str:database:UC_Households', {})
        if households_config:
            measures = households_config.get('measures', [])

            for measure in measures:
                if 'Payment' in measure['label'] or 'RSRS' in measure['label']:
                    measure_id = measure['id']
                    measure_label = measure['label']

                    self.logger.info(f"💰 Collecting payment measure: {measure_label}")

                    query = {
                        "database": "str:database:UC_Households",
                        "measures": [measure_id],
                        "dimensions": [["str:field:UC_Households:F_UC_HH_DATE:DATE_NAME"]]
                    }

                    result = self._make_request('POST', f"{self.base_url}/table", json=query)

                    if result:
                        df = self._convert_to_dataframe(result, f"payment_{measure_label}")
                        if df is not None and not df.empty:
                            special_data[f"payment_{measure_label}"] = df
                            self.logger.info(f"✅ Collected payment data: {len(df)} records")

        return special_data

    def collect_claims_and_starts_data(self) -> Dict[str, pd.DataFrame]:
        """Collect UC Claims and UC Starts data"""

        claims_starts_data = {}

        # UC Claims
        claims_config = self.schema.get('str:database:UC_Claims', {})
        if claims_config:
            self.logger.info("📈 Collecting UC Claims data...")

            query = {
                "database": "str:database:UC_Claims",
                "measures": ["str:count:UC_Claims:V_F_UC_CLAIMS"],
                "dimensions": [["str:field:UC_Claims:F_UC_WEEKS:WEEK_NAME"]]
            }

            result = self._make_request('POST', f"{self.base_url}/table", json=query)

            if result:
                df = self._convert_to_dataframe(result, "uc_claims")
                if df is not None and not df.empty:
                    claims_starts_data["uc_claims"] = df
                    self.logger.info(f"✅ UC Claims: {len(df)} records")

        # UC Starts
        starts_config = self.schema.get('str:database:UC_Starts', {})
        if starts_config:
            self.logger.info("🚀 Collecting UC Starts data...")

            query = {
                "database": "str:database:UC_Starts",
                "measures": ["str:count:UC_Starts:V_F_UC_STARTS"],
                "dimensions": [["str:field:UC_Starts:F_UC_DATE:DATE_NAME"]]
            }

            result = self._make_request('POST', f"{self.base_url}/table", json=query)

            if result:
                df = self._convert_to_dataframe(result, "uc_starts")
                if df is not None and not df.empty:
                    claims_starts_data["uc_starts"] = df
                    self.logger.info(f"✅ UC Starts: {len(df)} records")

        return claims_starts_data

    def collect_demographic_breakdowns(self) -> Dict[str, pd.DataFrame]:
        """Collect demographic breakdowns for people and households"""

        demo_data = {}

        # People demographics
        people_config = self.schema.get('str:database:UC_Monthly', {})
        if people_config:
            demo_fields = people_config.get('demographic_fields', [])

            for demo_field in demo_fields[:3]:  # Limit to avoid overwhelming API
                demo_id = demo_field['id']
                demo_label = demo_field['label']

                self.logger.info(f"👥 Collecting people demographic: {demo_label}")

                query = {
                    "database": "str:database:UC_Monthly",
                    "measures": ["str:count:UC_Monthly:V_F_UC_CASELOAD_FULL"],
                    "dimensions": [
                        ["str:field:UC_Monthly:F_UC_DATE:DATE_NAME"],
                        [demo_id]
                    ]
                }

                result = self._make_request('POST', f"{self.base_url}/table", json=query)

                if result:
                    df = self._convert_to_dataframe(result, f"people_demo_{demo_label}")
                    if df is not None and not df.empty:
                        demo_data[f"people_demo_{demo_label}"] = df
                        self.logger.info(f"✅ People demographic {demo_label}: {len(df)} records")

        # Households demographics
        households_config = self.schema.get('str:database:UC_Households', {})
        if households_config:
            demo_fields = households_config.get('demographic_fields', [])

            for demo_field in demo_fields:
                demo_id = demo_field['id']
                demo_label = demo_field['label']

                self.logger.info(f"🏠 Collecting households demographic: {demo_label}")

                query = {
                    "database": "str:database:UC_Households",
                    "measures": ["str:count:UC_Households:V_F_UC_HOUSEHOLDS"],
                    "dimensions": [
                        ["str:field:UC_Households:F_UC_HH_DATE:DATE_NAME"],
                        [demo_id]
                    ]
                }

                result = self._make_request('POST', f"{self.base_url}/table", json=query)

                if result:
                    df = self._convert_to_dataframe(result, f"households_demo_{demo_label}")
                    if df is not None and not df.empty:
                        demo_data[f"households_demo_{demo_label}"] = df
                        self.logger.info(f"✅ Households demographic {demo_label}: {len(df)} records")

        return demo_data

    def _convert_to_dataframe(self, api_response: Dict, data_type: str) -> Optional[pd.DataFrame]:
        """Convert API response to DataFrame"""

        if not api_response or 'cubes' not in api_response:
            return None

        try:
            cubes = api_response['cubes']
            if not cubes:
                return None

            measure_id = list(cubes.keys())[0]
            cube_data = cubes[measure_id]
            values = cube_data['values']

            # Extract time periods
            recodes = api_response.get('query', {}).get('recodes', {})
            time_periods = []
            categories = []

            for field_id, recode_info in recodes.items():
                if any(time_word in field_id.upper() for time_word in ['DATE', 'TIME', 'WEEK']):
                    for time_map in recode_info.get('map', []):
                        if time_map:
                            time_value_id = time_map[0]
                            # Extract date part
                            if 'WEEK' in field_id:
                                # Handle weekly data differently
                                week_part = time_value_id.split(':')[-1]
                                time_periods.append(f"Week {week_part}")
                            else:
                                date_part = time_value_id.split(':')[-1]
                                if len(date_part) == 6:  # YYYYMM format
                                    year = date_part[:4]
                                    month = date_part[4:6]
                                    time_periods.append(f"{year} {month}")
                                else:
                                    time_periods.append(date_part)
                else:
                    # Other dimension (demographic, etc.)
                    for cat_map in recode_info.get('map', []):
                        if cat_map:
                            categories.append(cat_map[0])

            data_rows = []

            if isinstance(values, list):
                if not categories:  # Simple time series
                    for i, value in enumerate(values):
                        if i < len(time_periods) and value is not None:
                            time_str = time_periods[i]

                            # Convert to date if possible
                            date_obj = None
                            if 'Week' not in time_str:
                                try:
                                    if ' ' in time_str:
                                        year, month = time_str.split()
                                        date_obj = pd.to_datetime(f"{year}-{month.zfill(2)}-01")
                                        time_label = date_obj.strftime('%Y %b')
                                    else:
                                        time_label = time_str
                                except:
                                    time_label = time_str
                            else:
                                time_label = time_str

                            row = {
                                'Geography_Code': 'UK_TOTAL',
                                'Geography_Name': 'United Kingdom',
                                'Time_Period': time_label,
                                'Date': date_obj,
                                'Value': float(value),
                                'Dataset': data_type,
                                'Measure': measure_id.split(':')[-1] if ':' in measure_id else measure_id
                            }
                            data_rows.append(row)

                else:  # Multi-dimensional data
                    # Handle 2D structure
                    if isinstance(values[0], list):
                        for i, time_values in enumerate(values):
                            time_period = time_periods[i] if i < len(time_periods) else f"Period_{i}"

                            for j, value in enumerate(time_values):
                                if value is not None:
                                    category = categories[j] if j < len(categories) else f"Category_{j}"

                                    # Convert time
                                    date_obj = None
                                    if 'Week' not in time_period:
                                        try:
                                            if ' ' in time_period:
                                                year, month = time_period.split()
                                                date_obj = pd.to_datetime(f"{year}-{month.zfill(2)}-01")
                                                time_label = date_obj.strftime('%Y %b')
                                            else:
                                                time_label = time_period
                                        except:
                                            time_label = time_period
                                    else:
                                        time_label = time_period

                                    row = {
                                        'Geography_Code': 'UK_TOTAL',
                                        'Geography_Name': 'United Kingdom',
                                        'Time_Period': time_label,
                                        'Date': date_obj,
                                        'Value': float(value),
                                        'Dataset': data_type,
                                        'Category': category,
                                        'Measure': measure_id.split(':')[-1] if ':' in measure_id else measure_id
                                    }
                                    data_rows.append(row)

            df = pd.DataFrame(data_rows)

            if not df.empty and 'Date' in df.columns:
                # Add derived fields
                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month
                df['Quarter'] = df['Date'].dt.quarter
                df['Month_Name'] = df['Date'].dt.strftime('%B')

                # Calculate growth rates
                if 'Value' in df.columns:
                    df = df.sort_values(['Geography_Code', 'Category', 'Date'] if 'Category' in df.columns else ['Geography_Code', 'Date'])
                    groupby_cols = ['Geography_Code', 'Category'] if 'Category' in df.columns else ['Geography_Code']
                    df['Previous_Value'] = df.groupby(groupby_cols)['Value'].shift(1)
                    df['Growth_Rate'] = ((df['Value'] - df['Previous_Value']) / df['Previous_Value'] * 100).round(2)
                    df['Growth_Rate'] = df['Growth_Rate'].replace([np.inf, -np.inf], np.nan)

            return df

        except Exception as e:
            self.logger.error(f"Error converting to DataFrame: {e}")
            return None

    def collect_everything(self, output_dir: str = "comprehensive_uc_data") -> Dict[str, str]:
        """Collect ALL available UC data"""

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.logger.info("🚀 Starting COMPREHENSIVE UC data collection...")
        self.logger.info("📊 Collecting ALL measures from ALL databases...")

        all_datasets = {}

        # 1. Collect all measures from all databases
        self.logger.info("1️⃣ Collecting all measures from all databases...")
        basic_data = self.collect_all_measures_all_databases()
        all_datasets.update(basic_data)

        # 2. Collect payment amounts and special measures
        self.logger.info("2️⃣ Collecting payment amounts and special measures...")
        payment_data = self.collect_payment_amounts_and_special_measures()
        all_datasets.update(payment_data)

        # 3. Collect claims and starts data
        self.logger.info("3️⃣ Collecting claims and starts data...")
        claims_starts = self.collect_claims_and_starts_data()
        all_datasets.update(claims_starts)

        # 4. Collect demographic breakdowns
        self.logger.info("4️⃣ Collecting demographic breakdowns...")
        demo_data = self.collect_demographic_breakdowns()
        all_datasets.update(demo_data)

        # Save all datasets
        saved_files = {}

        for dataset_name, df in all_datasets.items():
            if not df.empty:
                # Save as parquet
                parquet_file = f"{output_dir}/{dataset_name}_{timestamp}.parquet"
                df.to_parquet(parquet_file, index=False)
                saved_files[f"{dataset_name}_parquet"] = parquet_file

                self.logger.info(f"✅ Saved {dataset_name}: {len(df)} records")

        # Create mega combined dataset
        if all_datasets:
            all_dfs = [df for df in all_datasets.values() if not df.empty]
            if all_dfs:
                mega_combined = pd.concat(all_dfs, ignore_index=True, sort=False)

                mega_file = f"{output_dir}/uc_mega_comprehensive_{timestamp}.parquet"
                mega_combined.to_parquet(mega_file, index=False)
                saved_files['mega_comprehensive'] = mega_file

                # Also save as CSV
                mega_csv = f"{output_dir}/uc_mega_comprehensive_{timestamp}.csv"
                mega_combined.to_csv(mega_csv, index=False)
                saved_files['mega_comprehensive_csv'] = mega_csv

                self.logger.info(f"🎯 MEGA DATASET CREATED: {len(mega_combined)} total records")
                self.logger.info(f"📊 Datasets included: {len(all_datasets)}")
                self.logger.info(f"📅 Date range: {mega_combined['Date'].min()} to {mega_combined['Date'].max()}")
                self.logger.info(f"📈 Total API calls: {self.api_calls}")

        return saved_files

def main():
    api_key = os.getenv('STAT_XPLORE_API_KEY')
    if not api_key:
        print("❌ API key not found")
        return

    collector = ComprehensiveUCCollector(api_key)
    saved_files = collector.collect_everything()

    if saved_files:
        print("\n🎉 COMPREHENSIVE UC DATA COLLECTION COMPLETE!")
        print("📂 Files created:")
        for file_type, filepath in saved_files.items():
            print(f"  - {file_type}: {filepath}")
    else:
        print("❌ No data collected")

if __name__ == '__main__':
    main()