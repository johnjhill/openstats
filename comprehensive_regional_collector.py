#!/usr/bin/env python3
"""
Comprehensive Regional Data Collector
Uses the breakthrough recodes pattern to collect UC, PIP, and Housing Benefit regional data
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

class ComprehensiveRegionalCollector:
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

    def get_regional_values(self, database: str, field_id: str) -> List[str]:
        """Get regional values for a specific geographic field"""
        try:
            self.rate_limit()
            url = f"{self.base_url}/schema/{field_id}"
            response = requests.get(url, headers=self.headers)

            if response.status_code == 200:
                field_data = response.json()

                # Look for regional valueset (Region level)
                if 'children' in field_data:
                    for child in field_data['children']:
                        child_id = child.get('id', '')
                        child_label = child.get('label', '')

                        # Look for Region level data
                        if 'REGION' in child_id.upper():
                            logger.info(f"Found regional valueset: {child_label}")
                            return self.get_values_from_valueset(child_id)

                # If no regional level found, try the first geographic valueset
                if 'children' in field_data and field_data['children']:
                    return self.get_values_from_valueset(field_data['children'][0]['id'])

            return []
        except Exception as e:
            logger.error(f"Error getting regional values: {e}")
            return []

    def get_values_from_valueset(self, valueset_id: str) -> List[str]:
        """Get individual values from a valueset"""
        try:
            self.rate_limit()
            url = f"{self.base_url}/schema/{valueset_id}"
            response = requests.get(url, headers=self.headers)

            if response.status_code == 200:
                valueset_data = response.json()

                if 'children' in valueset_data:
                    values = []
                    for child in valueset_data['children']:
                        value_id = child.get('id', '')
                        if value_id:
                            values.append(value_id)
                    return values[:10]  # Limit to first 10 for testing

            return []
        except Exception as e:
            logger.error(f"Error getting values from valueset: {e}")
            return []

    def get_time_values(self, database: str, time_field: str) -> List[str]:
        """Get recent time values for a time field"""
        try:
            self.rate_limit()
            url = f"{self.base_url}/schema/{time_field}"
            response = requests.get(url, headers=self.headers)

            if response.status_code == 200:
                field_data = response.json()

                # Look for time valueset
                if 'children' in field_data:
                    for child in field_data['children']:
                        valueset_id = child.get('id', '')
                        if valueset_id:
                            return self.get_recent_time_values(valueset_id)

            return []
        except Exception as e:
            logger.error(f"Error getting time values: {e}")
            return []

    def get_recent_time_values(self, time_valueset_id: str) -> List[str]:
        """Get recent time values from time valueset"""
        try:
            self.rate_limit()
            url = f"{self.base_url}/schema/{time_valueset_id}"
            response = requests.get(url, headers=self.headers)

            if response.status_code == 200:
                valueset_data = response.json()

                if 'children' in valueset_data:
                    recent_values = []
                    for child in valueset_data['children']:
                        time_id = child.get('id', '')
                        if time_id and ('2024' in time_id or '2023' in time_id):
                            recent_values.append(time_id)

                    # Return last 6 months
                    return recent_values[-6:] if recent_values else []

            return []
        except Exception as e:
            logger.error(f"Error getting recent time values: {e}")
            return []

    def collect_regional_data(self, config: Dict[str, str]) -> pd.DataFrame:
        """Collect regional data using the breakthrough recodes pattern"""

        logger.info(f"🗺️ Collecting {config['name']}...")

        # Get regional and time values
        regional_values = self.get_regional_values(config['database'], config['geographic_field'])
        time_values = self.get_time_values(config['database'], config['time_field'])

        if not regional_values:
            logger.warning(f"No regional values found for {config['name']}")
            return pd.DataFrame()

        if not time_values:
            logger.warning(f"No time values found for {config['name']}")
            return pd.DataFrame()

        logger.info(f"Using {len(regional_values)} regions and {len(time_values)} time periods")

        # Build the breakthrough recodes payload
        payload = {
            "database": config['database'],
            "measures": [config['measure']],
            "recodes": {
                config['geographic_field']: {
                    "map": [[value] for value in regional_values],
                    "total": True
                },
                config['time_field']: {
                    "map": [[value] for value in time_values],
                    "total": False
                }
            },
            "dimensions": [
                [config['geographic_field']],
                [config['time_field']]
            ]
        }

        try:
            self.rate_limit()
            url = f"{self.base_url}/table"
            logger.info(f"🔄 Querying {config['database']} with regional breakdown...")

            response = requests.post(url, json=payload, headers=self.headers)

            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ SUCCESS! Collected regional data for {config['name']}")
                return self.convert_response_to_dataframe(data, config['name'])
            else:
                try:
                    error_data = response.json()
                    logger.error(f"API error {response.status_code}: {error_data.get('message', response.text)}")
                except:
                    logger.error(f"API error {response.status_code}: {response.text}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error collecting {config['name']}: {e}")
            return pd.DataFrame()

    def convert_response_to_dataframe(self, response: Dict, collection_name: str) -> pd.DataFrame:
        """Convert regional API response to DataFrame with correct nested structure handling"""
        records = []

        if 'cubes' not in response:
            return pd.DataFrame()

        # Get field information from the response
        fields_info = {}
        if 'fields' in response:
            for field in response['fields']:
                field_uri = field.get('uri', '')
                field_info = {
                    'label': field.get('label', ''),
                    'items': field.get('items', [])
                }
                fields_info[field_uri] = field_info

        for cube_id, cube_data in response['cubes'].items():
            if 'values' not in cube_data:
                continue

            values = cube_data['values']

            # Extract geographic and time information from fields
            geographic_field = None
            time_field = None

            for field_uri, field_info in fields_info.items():
                if 'COA_CODE' in field_uri or 'geography' in field_info['label'].lower():
                    geographic_field = field_info
                elif 'DATE' in field_uri or 'time' in field_info['label'].lower() or 'month' in field_info['label'].lower():
                    time_field = field_info

            if not geographic_field or not time_field:
                continue

            # Extract geographic areas (excluding Total)
            geo_items = [item for item in geographic_field['items'] if item.get('type') != 'Total']
            time_items = time_field['items']

            # Process the nested values structure
            for i, geo_item in enumerate(geo_items):
                geo_labels = geo_item.get('labels', [])
                geo_uris = geo_item.get('uris', [])
                geo_name = geo_labels[0] if geo_labels else f'Area_{i}'
                geo_code = geo_uris[0].split(':')[-1] if geo_uris else f'GEO_{i}'

                # Get the values for this geographic area
                if i < len(values):
                    area_values = values[i]

                    # Handle both nested arrays and flat arrays
                    if isinstance(area_values, list):
                        value_list = area_values
                    else:
                        value_list = [area_values]

                    for j, time_item in enumerate(time_items):
                        time_labels = time_item.get('labels', [])
                        time_uris = time_item.get('uris', [])
                        time_period = time_labels[0] if time_labels else f'Period_{j}'

                        # Get the value for this time period
                        if j < len(value_list):
                            value = value_list[j]

                            if value is not None and value > 0:
                                records.append({
                                    'Geography_Code': geo_code,
                                    'Geography_Name': geo_name,
                                    'Time_Period': time_period,
                                    'Value': value,
                                    'Dataset': collection_name,
                                    'Measure': cube_id,
                                    'Benefit_Type': self.get_benefit_type(collection_name)
                                })

        df = pd.DataFrame(records)

        if not df.empty:
            # Process dates
            try:
                # Handle different date formats
                if df['Time_Period'].str.contains(r'\d{6}').any():
                    # Format like "202410 (Oct-24)"
                    df['Date'] = pd.to_datetime(df['Time_Period'].str.extract(r'(\d{6})')[0], format='%Y%m', errors='coerce')
                else:
                    df['Date'] = pd.to_datetime(df['Time_Period'], format='%Y %b', errors='coerce')
            except:
                df['Date'] = pd.to_datetime(df['Time_Period'], errors='coerce')

            # Add helpful columns
            if 'Date' in df.columns and df['Date'].notna().any():
                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month
                df['Quarter'] = df['Date'].dt.quarter

            logger.info(f"✅ {collection_name}: {len(df)} records processed")
            if len(df) > 0:
                logger.info(f"   📍 Geographic areas: {df['Geography_Code'].nunique()}")
                logger.info(f"   📅 Time periods: {df['Time_Period'].nunique()}")
                logger.info(f"   📊 Total cases: {df['Value'].sum():,.0f}")

        return df

    def get_benefit_type(self, collection_name: str) -> str:
        """Determine benefit type from collection name"""
        if 'uc' in collection_name.lower() or 'universal' in collection_name.lower():
            return 'Universal Credit'
        elif 'pip' in collection_name.lower():
            return 'Personal Independence Payment'
        elif 'hb' in collection_name.lower() or 'housing' in collection_name.lower():
            return 'Housing Benefit'
        else:
            return 'Unknown'

    def collect_all_regional_data(self) -> Dict[str, pd.DataFrame]:
        """Collect comprehensive regional data for all benefit types"""

        logger.info("🚀 Starting comprehensive regional data collection with breakthrough pattern...")

        # Define working configurations based on the breakthrough
        configs = [
            # Housing Benefit (confirmed working)
            {
                'database': 'str:database:hb_new',
                'measure': 'str:count:hb_new:V_F_HB_NEW',
                'geographic_field': 'str:field:hb_new:V_F_HB_NEW:COA_CODE',
                'time_field': 'str:field:hb_new:F_HB_NEW_DATE:NEW_DATE_NAME',
                'name': 'housing_benefit_regional'
            },
            # Universal Credit configurations
            {
                'database': 'str:database:UC_Households',
                'measure': 'str:count:UC_Households:V_F_UC_HOUSEHOLDS',
                'geographic_field': 'str:field:UC_Households:V_F_UC_HOUSEHOLDS:COA_CODE',
                'time_field': 'str:field:UC_Households:F_UC_HH_DATE:DATE_NAME',
                'name': 'uc_households_regional'
            },
            {
                'database': 'str:database:UC_Monthly',
                'measure': 'str:count:UC_Monthly:V_F_UC_CASELOAD_FULL',
                'geographic_field': 'str:field:UC_Monthly:V_F_UC_CASELOAD_FULL:COA_CODE',
                'time_field': 'str:field:UC_Monthly:F_UC_DATE:DATE_NAME',
                'name': 'uc_people_regional'
            },
            # PIP configurations
            {
                'database': 'str:database:PIP_Monthly_new',
                'measure': 'str:count:PIP_Monthly_new:V_F_PIP_MONTHLY',
                'geographic_field': 'str:field:PIP_Monthly_new:V_F_PIP_MONTHLY:COA_2011',
                'time_field': 'str:field:PIP_Monthly_new:F_PIP_DATE:DATE2',
                'name': 'pip_regional'
            }
        ]

        results = {}

        for config in configs:
            logger.info(f"📊 Collecting: {config['name']}")

            df = self.collect_regional_data(config)

            if not df.empty:
                results[config['name']] = df
                logger.info(f"✅ {config['name']}: {len(df)} records collected")
            else:
                logger.warning(f"❌ No data collected for {config['name']}")

        return results

def main():
    """Run comprehensive regional data collection"""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv('STAT_XPLORE_API_KEY')
    if not api_key:
        logger.error("API key not found in environment variables")
        return

    collector = ComprehensiveRegionalCollector(api_key)

    # Collect all regional data
    regional_datasets = collector.collect_all_regional_data()

    if not regional_datasets:
        logger.error("No regional data was collected")
        return

    # Save results
    output_dir = "regional_benefit_data"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    saved_files = {}

    for name, df in regional_datasets.items():
        if not df.empty:
            # Save as parquet
            filename = f"{output_dir}/{name}_{timestamp}.parquet"
            df.to_parquet(filename, index=False)
            saved_files[name] = filename

            # Save as CSV for inspection
            csv_filename = f"{output_dir}/{name}_{timestamp}.csv"
            df.to_csv(csv_filename, index=False)

            logger.info(f"💾 Saved {name}: {len(df)} records")

    # Create combined regional dataset
    if regional_datasets:
        combined_df = pd.concat(regional_datasets.values(), ignore_index=True)

        # Add growth rate calculations
        if not combined_df.empty and 'Date' in combined_df.columns:
            combined_df = combined_df.sort_values(['Geography_Code', 'Dataset', 'Date'])
            combined_df['Previous_Value'] = combined_df.groupby(['Geography_Code', 'Dataset'])['Value'].shift(1)
            combined_df['Growth_Rate'] = ((combined_df['Value'] - combined_df['Previous_Value']) / combined_df['Previous_Value'] * 100).round(2)
            combined_df['Growth_Rate'] = combined_df['Growth_Rate'].replace([float('inf'), float('-inf')], None)

        combined_filename = f"{output_dir}/all_benefits_regional_{timestamp}.parquet"
        combined_df.to_parquet(combined_filename, index=False)

        csv_filename = f"{output_dir}/all_benefits_regional_{timestamp}.csv"
        combined_df.to_csv(csv_filename, index=False)

        # Calculate summary statistics
        total_records = len(combined_df)
        total_areas = combined_df['Geography_Code'].nunique()
        benefit_types = combined_df['Benefit_Type'].unique()
        datasets = combined_df['Dataset'].nunique()
        date_range = f"{combined_df['Date'].min().strftime('%Y-%m')} to {combined_df['Date'].max().strftime('%Y-%m')}" if 'Date' in combined_df.columns else "Unknown"

        print(f"\n🎉 COMPREHENSIVE REGIONAL BENEFIT DATA COLLECTION COMPLETE!")
        print(f"📂 Files saved to: {output_dir}/")
        print(f"📊 Total records: {total_records:,}")
        print(f"🗺️ Geographic areas: {total_areas:,}")
        print(f"📅 Date range: {date_range}")
        print(f"📈 Datasets: {datasets}")
        print(f"💼 Benefit types: {', '.join(benefit_types)}")

        print(f"\n📋 Individual datasets collected:")
        for name, df in regional_datasets.items():
            print(f"  - {name}: {len(df):,} records")

        print(f"\n🎯 This breakthrough enables:")
        print(f"  - Regional benefit trend analysis")
        print(f"  - Geographic inequality studies")
        print(f"  - Cross-benefit regional comparisons")
        print(f"  - Policy impact assessment by region")
        print(f"  - Time-series forecasting by area")

        return combined_filename
    else:
        logger.warning("No regional data to combine")
        return None

if __name__ == "__main__":
    main()