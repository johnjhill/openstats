#!/usr/bin/env python3
"""
Regional Universal Credit Data Collector
Specifically designed to extract regional and local authority breakdowns
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

class RegionalUCCollector:
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

    def explore_table_for_regions(self, table_id: str) -> Dict[str, Any]:
        """Explore a table to find all available geographic dimensions"""
        url = f"{self.base_url}/table"

        payload = {
            "database": table_id,
            "measures": [],
            "dimensions": [],
            "recodes": {}
        }

        try:
            self.rate_limit()
            response = requests.post(url, json=payload, headers=self.headers)

            if response.status_code == 200:
                data = response.json()
                return data
            else:
                logger.error(f"Failed to explore table {table_id}: {response.status_code} - {response.text}")
                return {}

        except Exception as e:
            logger.error(f"Error exploring table {table_id}: {e}")
            return {}

    def get_dimension_values(self, dimension_id: str) -> List[Dict[str, str]]:
        """Get all available values for a dimension (like geographic areas)"""
        url = f"{self.base_url}/table"

        # First, try to get the values by exploring the dimension
        payload = {
            "database": dimension_id.split(':')[2],  # Extract database from dimension ID
            "measures": [],
            "dimensions": [dimension_id],
            "recodes": {}
        }

        try:
            self.rate_limit()
            response = requests.post(url, json=payload, headers=self.headers)

            if response.status_code == 200:
                data = response.json()

                # Extract geographic values from the response
                if 'cubes' in data:
                    for cube_id, cube_data in data['cubes'].items():
                        if 'dimensions' in cube_data:
                            for dim in cube_data['dimensions']:
                                if dim['field'] == dimension_id:
                                    return dim.get('items', [])

                # Alternative: check if dimension info is in fields
                if 'fields' in data:
                    for field in data['fields']:
                        if field['uri'] == dimension_id:
                            return field.get('items', [])

            else:
                logger.error(f"Failed to get dimension values for {dimension_id}: {response.status_code}")

        except Exception as e:
            logger.error(f"Error getting dimension values for {dimension_id}: {e}")

        return []

    def collect_regional_data(self, database_id: str, measure_id: str,
                            geographic_dimension: str, time_dimension: str = None) -> pd.DataFrame:
        """Collect UC data with regional breakdown"""

        logger.info(f"🗺️ Collecting regional data for {database_id}")

        # Get available geographic areas
        logger.info(f"🔍 Exploring geographic dimension: {geographic_dimension}")
        geo_values = self.get_dimension_values(geographic_dimension)

        if not geo_values:
            logger.warning(f"No geographic values found for {geographic_dimension}")
            # Try without geographic breakdown (national only)
            return self.collect_national_data(database_id, measure_id, time_dimension)

        logger.info(f"📍 Found {len(geo_values)} geographic areas")

        # Prepare query with geographic breakdown
        dimensions = [geographic_dimension]
        if time_dimension:
            dimensions.append(time_dimension)

        url = f"{self.base_url}/table"
        payload = {
            "database": database_id,
            "measures": [measure_id],
            "dimensions": dimensions,
            "recodes": {}
        }

        try:
            self.rate_limit()
            logger.info(f"🔄 Querying {database_id} with geographic breakdown...")
            response = requests.post(url, json=payload, headers=self.headers)

            if response.status_code == 200:
                data = response.json()
                return self.convert_regional_response_to_dataframe(data, geographic_dimension)
            else:
                logger.error(f"API error {response.status_code}: {response.text}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error collecting regional data: {e}")
            return pd.DataFrame()

    def collect_national_data(self, database_id: str, measure_id: str, time_dimension: str = None) -> pd.DataFrame:
        """Fallback to collect national-level data when regional isn't available"""

        logger.info(f"📊 Collecting national data for {database_id}")

        dimensions = []
        if time_dimension:
            dimensions.append(time_dimension)

        url = f"{self.base_url}/table"
        payload = {
            "database": database_id,
            "measures": [measure_id],
            "dimensions": dimensions,
            "recodes": {}
        }

        try:
            self.rate_limit()
            response = requests.post(url, json=payload, headers=self.headers)

            if response.status_code == 200:
                data = response.json()
                return self.convert_national_response_to_dataframe(data)
            else:
                logger.error(f"API error {response.status_code}: {response.text}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error collecting national data: {e}")
            return pd.DataFrame()

    def convert_regional_response_to_dataframe(self, response: Dict, geographic_dimension: str) -> pd.DataFrame:
        """Convert API response with regional data to pandas DataFrame"""

        records = []

        if 'cubes' not in response:
            logger.warning("No cubes in response")
            return pd.DataFrame()

        for cube_id, cube_data in response['cubes'].items():
            if 'values' not in cube_data:
                continue

            # Get dimension info
            dimensions = cube_data.get('dimensions', [])
            geo_dimension = None
            time_dimension = None

            for dim in dimensions:
                if dim['field'] == geographic_dimension:
                    geo_dimension = dim
                elif 'DATE' in dim['field'] or 'TIME' in dim['field']:
                    time_dimension = dim

            # Extract data values
            values = cube_data['values']

            if geo_dimension:
                # Regional data available
                for i, geo_item in enumerate(geo_dimension.get('items', [])):
                    geo_code = geo_item.get('codes', [geo_item.get('uri', 'Unknown')])[0]
                    geo_name = geo_item.get('labels', [geo_item.get('label', 'Unknown')])[0]

                    if time_dimension:
                        # Regional + time series
                        for j, time_item in enumerate(time_dimension.get('items', [])):
                            time_period = time_item.get('labels', [time_item.get('label', 'Unknown')])[0]

                            # Calculate index in values array
                            value_index = i * len(time_dimension.get('items', [])) + j
                            value = values[value_index] if value_index < len(values) else None

                            if value is not None:
                                records.append({
                                    'Geography_Code': geo_code,
                                    'Geography_Name': geo_name,
                                    'Time_Period': time_period,
                                    'Value': value,
                                    'Dataset': cube_id,
                                    'Measure': cube_id
                                })
                    else:
                        # Regional only, no time
                        value = values[i] if i < len(values) else None
                        if value is not None:
                            records.append({
                                'Geography_Code': geo_code,
                                'Geography_Name': geo_name,
                                'Time_Period': 'Latest',
                                'Value': value,
                                'Dataset': cube_id,
                                'Measure': cube_id
                            })

        return pd.DataFrame(records)

    def convert_national_response_to_dataframe(self, response: Dict) -> pd.DataFrame:
        """Convert API response with national data to pandas DataFrame"""

        records = []

        if 'cubes' not in response:
            return pd.DataFrame()

        for cube_id, cube_data in response['cubes'].items():
            if 'values' not in cube_data:
                continue

            dimensions = cube_data.get('dimensions', [])
            time_dimension = None

            for dim in dimensions:
                if 'DATE' in dim['field'] or 'TIME' in dim['field']:
                    time_dimension = dim
                    break

            values = cube_data['values']

            if time_dimension:
                # National time series
                for i, time_item in enumerate(time_dimension.get('items', [])):
                    time_period = time_item.get('labels', [time_item.get('label', 'Unknown')])[0]
                    value = values[i] if i < len(values) else None

                    if value is not None:
                        records.append({
                            'Geography_Code': 'UK_TOTAL',
                            'Geography_Name': 'United Kingdom',
                            'Time_Period': time_period,
                            'Value': value,
                            'Dataset': cube_id,
                            'Measure': cube_id
                        })
            else:
                # Single national value
                if values:
                    records.append({
                        'Geography_Code': 'UK_TOTAL',
                        'Geography_Name': 'United Kingdom',
                        'Time_Period': 'Latest',
                        'Value': values[0],
                        'Dataset': cube_id,
                        'Measure': cube_id
                    })

        return pd.DataFrame(records)

    def collect_all_regional_uc_data(self) -> Dict[str, pd.DataFrame]:
        """Collect comprehensive regional UC data"""

        logger.info("🚀 Starting comprehensive regional UC data collection...")

        # Define the datasets and their geographic dimensions
        regional_configs = [
            {
                'database': 'str:database:UC_Households',
                'measure': 'str:count:UC_Households:V_F_UC_HOUSEHOLDS',
                'geographic_dimension': 'str:field:UC_Households:V_F_UC_HOUSEHOLDS:COA_CODE',
                'time_dimension': 'str:field:UC_Households:F_UC_HH_DATE:DATE_NAME',
                'name': 'households_regional_oa'
            },
            {
                'database': 'str:database:UC_Households',
                'measure': 'str:count:UC_Households:V_F_UC_HOUSEHOLDS',
                'geographic_dimension': 'str:field:UC_Households:V_F_UC_HOUSEHOLDS:WARD_CODE',
                'time_dimension': 'str:field:UC_Households:F_UC_HH_DATE:DATE_NAME',
                'name': 'households_regional_ward'
            },
            {
                'database': 'str:database:UC_Monthly',
                'measure': 'str:count:UC_Monthly:V_F_UC_CASELOAD_FULL',
                'geographic_dimension': 'str:field:UC_Monthly:V_F_UC_CASELOAD_FULL:COA_CODE',
                'time_dimension': 'str:field:UC_Monthly:F_UC_DATE:DATE_NAME',
                'name': 'people_regional_oa'
            },
            {
                'database': 'str:database:UC_Monthly',
                'measure': 'str:count:UC_Monthly:V_F_UC_CASELOAD_FULL',
                'geographic_dimension': 'str:field:UC_Monthly:V_F_UC_CASELOAD_FULL:WARD_CODE',
                'time_dimension': 'str:field:UC_Monthly:F_UC_DATE:DATE_NAME',
                'name': 'people_regional_ward'
            },
            {
                'database': 'str:database:UC_Starts',
                'measure': 'str:count:UC_Starts:V_F_UC_STARTS',
                'geographic_dimension': 'str:field:UC_Starts:V_F_UC_STARTS:POSTCODE_DISTRICT',
                'time_dimension': 'str:field:UC_Starts:F_UC_DATE:DATE_NAME',
                'name': 'starts_postcode_district'
            }
        ]

        results = {}

        for config in regional_configs:
            logger.info(f"📊 Collecting {config['name']}...")

            df = self.collect_regional_data(
                config['database'],
                config['measure'],
                config['geographic_dimension'],
                config['time_dimension']
            )

            if not df.empty:
                logger.info(f"✅ {config['name']}: {len(df)} records")
                results[config['name']] = df
            else:
                logger.warning(f"❌ No data for {config['name']}")

        return results

def main():
    """Run regional data collection"""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv('STAT_XPLORE_API_KEY')
    if not api_key:
        logger.error("API key not found in environment variables")
        return

    collector = RegionalUCCollector(api_key)

    # Collect all regional data
    regional_datasets = collector.collect_all_regional_uc_data()

    # Save results
    output_dir = "regional_uc_data"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    saved_files = {}

    for name, df in regional_datasets.items():
        if not df.empty:
            # Add date processing
            if 'Time_Period' in df.columns:
                try:
                    df['Date'] = pd.to_datetime(df['Time_Period'], format='%Y %b', errors='coerce')
                except:
                    df['Date'] = pd.to_datetime(df['Time_Period'], errors='coerce')

            # Save as parquet
            filename = f"{output_dir}/uc_{name}_{timestamp}.parquet"
            df.to_parquet(filename, index=False)
            saved_files[name] = filename

            logger.info(f"💾 Saved {name}: {filename}")

    # Create combined regional dataset
    if regional_datasets:
        combined_df = pd.concat(regional_datasets.values(), ignore_index=True)
        combined_filename = f"{output_dir}/uc_regional_combined_{timestamp}.parquet"
        combined_df.to_parquet(combined_filename, index=False)

        csv_filename = f"{output_dir}/uc_regional_combined_{timestamp}.csv"
        combined_df.to_csv(csv_filename, index=False)

        logger.info(f"🎯 Combined regional dataset: {len(combined_df)} records")
        logger.info(f"📊 Geographic areas: {combined_df['Geography_Code'].nunique()}")
        logger.info(f"📅 Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")

        print(f"\n🎉 REGIONAL UC DATA COLLECTION COMPLETE!")
        print(f"📂 Files saved to: {output_dir}/")
        print(f"📊 Total records: {len(combined_df):,}")
        print(f"🗺️ Geographic areas: {combined_df['Geography_Code'].nunique():,}")
        print(f"📈 Datasets: {len(regional_datasets)}")

        return combined_filename
    else:
        logger.warning("No regional data collected")
        return None

if __name__ == "__main__":
    main()