#!/usr/bin/env python3
"""
Working Regional Data Collector for UC and PIP
Uses the discovered recodes pattern to access regional benefit data
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

class WorkingRegionalCollector:
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

    def get_geographic_values(self, field_id: str) -> List[str]:
        """Get available geographic values for a field"""
        try:
            self.rate_limit()
            url = f"{self.base_url}/schema/{field_id}"
            response = requests.get(url, headers=self.headers)

            if response.status_code == 200:
                data = response.json()
                if 'children' in data:
                    return [child.get('id', '') for child in data['children'][:10]]  # First 10 areas
            return []
        except Exception as e:
            logger.error(f"Error getting geographic values: {e}")
            return []

    def collect_regional_data_with_recodes(self, database_id: str, measure_id: str,
                                         geographic_field_id: str, time_field_id: str,
                                         collection_name: str, sample_areas: List[str] = None) -> pd.DataFrame:
        """Collect regional data using the recodes pattern"""

        logger.info(f"🗺️ Collecting {collection_name} with recodes pattern...")

        # If no sample areas provided, try to get some
        if not sample_areas:
            sample_areas = self.get_geographic_values(geographic_field_id)
            if not sample_areas:
                # Use database-specific fallback areas based on the database ID
                if 'hb_new' in database_id.lower():
                    # Housing Benefit areas from the JSON examples
                    sample_areas = [
                        "str:value:hb_new:V_F_HB_NEW:COA_CODE:V_C_MASTERGEOG11_GB:K03000001",  # England
                        "str:value:hb_new:V_F_HB_NEW:COA_CODE:V_C_MASTERGEOG11_GB:K04000001",  # Wales
                        "str:value:hb_new:V_F_HB_NEW:COA_CODE:V_C_MASTERGEOG11_GB:K02000001"   # Scotland
                    ]
                elif 'uc_households' in database_id.lower():
                    # UC Households areas
                    sample_areas = [
                        "str:value:UC_Households:V_F_UC_HOUSEHOLDS:COA_CODE:V_C_MASTERGEOG11_GB:K03000001",  # England
                        "str:value:UC_Households:V_F_UC_HOUSEHOLDS:COA_CODE:V_C_MASTERGEOG11_GB:K04000001",  # Wales
                        "str:value:UC_Households:V_F_UC_HOUSEHOLDS:COA_CODE:V_C_MASTERGEOG11_GB:K02000001"   # Scotland
                    ]
                elif 'uc_monthly' in database_id.lower():
                    # UC Monthly areas
                    sample_areas = [
                        "str:value:UC_Monthly:V_F_UC_CASELOAD_FULL:COA_CODE:V_C_MASTERGEOG11_GB:K03000001",  # England
                        "str:value:UC_Monthly:V_F_UC_CASELOAD_FULL:COA_CODE:V_C_MASTERGEOG11_GB:K04000001",  # Wales
                        "str:value:UC_Monthly:V_F_UC_CASELOAD_FULL:COA_CODE:V_C_MASTERGEOG11_GB:K02000001"   # Scotland
                    ]
                elif 'pip' in database_id.lower():
                    # PIP areas
                    sample_areas = [
                        "str:value:PIP_Monthly_new:V_F_PIP_MONTHLY:COA_2011:V_C_MASTERGEOG11_GB:K03000001",  # England
                        "str:value:PIP_Monthly_new:V_F_PIP_MONTHLY:COA_2011:V_C_MASTERGEOG11_GB:K04000001",  # Wales
                        "str:value:PIP_Monthly_new:V_F_PIP_MONTHLY:COA_2011:V_C_MASTERGEOG11_GB:K02000001"   # Scotland
                    ]
                else:
                    # Generic fallback
                    sample_areas = [
                        "str:value:GENERIC:FIELD:COA_CODE:V_C_MASTERGEOG11_GB:K03000001",  # England
                        "str:value:GENERIC:FIELD:COA_CODE:V_C_MASTERGEOG11_GB:K04000001",  # Wales
                        "str:value:GENERIC:FIELD:COA_CODE:V_C_MASTERGEOG11_GB:K02000001"   # Scotland
                    ]

        # Create time dimension map (last 24 months)
        time_values = []
        for year in [2023, 2024, 2025]:
            for month in range(1, 13):
                if year == 2025 and month > 8:  # Stop at Aug 2025
                    break
                time_values.append([f"str:value:{database_id.split(':')[-1]}:{time_field_id.split(':')[-1]}:C_DATE:{year:04d}{month:02d}"])

        # Build payload using the recodes pattern from Housing Benefit example
        payload = {
            "database": database_id,
            "measures": [measure_id],
            "recodes": {
                geographic_field_id: {
                    "map": [[area] for area in sample_areas],
                    "total": True
                },
                time_field_id: {
                    "map": time_values,
                    "total": False
                }
            },
            "dimensions": [[geographic_field_id], [time_field_id]]
        }

        try:
            self.rate_limit()
            url = f"{self.base_url}/table"
            logger.info(f"🔄 Querying {database_id} with recodes...")

            response = requests.post(url, json=payload, headers=self.headers)

            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ SUCCESS! Regional data accessed for {collection_name}")
                return self.convert_response_to_dataframe(data, collection_name)
            else:
                try:
                    error_data = response.json()
                    logger.error(f"API error {response.status_code}: {error_data.get('message', response.text)}")
                except:
                    logger.error(f"API error {response.status_code}: {response.text}")

                # Try simpler query without recodes
                return self.try_simple_query(database_id, measure_id, time_field_id, collection_name)

        except Exception as e:
            logger.error(f"Error collecting {collection_name}: {e}")
            return pd.DataFrame()

    def try_simple_query(self, database_id: str, measure_id: str, time_field_id: str, collection_name: str) -> pd.DataFrame:
        """Fallback to simple national query"""
        logger.info(f"🔄 Trying simple national query for {collection_name}...")

        payload = {
            "database": database_id,
            "measures": [measure_id],
            "dimensions": [time_field_id],
            "recodes": {}
        }

        try:
            self.rate_limit()
            url = f"{self.base_url}/table"
            response = requests.post(url, json=payload, headers=self.headers)

            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ National data collected for {collection_name}")
                return self.convert_response_to_dataframe(data, collection_name)
            else:
                logger.error(f"Simple query also failed for {collection_name}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error in simple query: {e}")
            return pd.DataFrame()

    def convert_response_to_dataframe(self, response: Dict, collection_name: str) -> pd.DataFrame:
        """Convert API response to DataFrame"""
        records = []

        if 'cubes' not in response:
            return pd.DataFrame()

        for cube_id, cube_data in response['cubes'].items():
            if 'values' not in cube_data:
                continue

            dimensions = cube_data.get('dimensions', [])
            values = cube_data['values']

            # Handle different dimension structures
            if len(dimensions) == 1:
                # Single dimension (time only)
                time_dimension = dimensions[0]
                time_items = time_dimension.get('items', [])

                for i, time_item in enumerate(time_items):
                    if i < len(values) and values[i] is not None:
                        time_label = time_item.get('labels', ['Unknown'])[0]

                        records.append({
                            'Geography_Code': 'UK_TOTAL',
                            'Geography_Name': 'United Kingdom',
                            'Time_Period': time_label,
                            'Value': values[i],
                            'Dataset': collection_name,
                            'Measure': cube_id,
                            'Benefit_Type': self.get_benefit_type(collection_name)
                        })

            elif len(dimensions) == 2:
                # Two dimensions (geography + time)
                geo_dimension = dimensions[0]
                time_dimension = dimensions[1]

                geo_items = geo_dimension.get('items', [])
                time_items = time_dimension.get('items', [])

                for i, geo_item in enumerate(geo_items):
                    geo_codes = geo_item.get('codes', [])
                    geo_labels = geo_item.get('labels', [])
                    geo_code = geo_codes[0] if geo_codes else f'GEO_{i}'
                    geo_name = geo_labels[0] if geo_labels else f'Area_{i}'

                    for j, time_item in enumerate(time_items):
                        time_label = time_item.get('labels', ['Unknown'])[0]

                        # Calculate index in values array
                        value_index = i * len(time_items) + j
                        if value_index < len(values) and values[value_index] is not None:
                            records.append({
                                'Geography_Code': geo_code,
                                'Geography_Name': geo_name,
                                'Time_Period': time_label,
                                'Value': values[value_index],
                                'Dataset': collection_name,
                                'Measure': cube_id,
                                'Benefit_Type': self.get_benefit_type(collection_name)
                            })

        df = pd.DataFrame(records)

        if not df.empty:
            # Process dates
            try:
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

    def collect_comprehensive_benefit_data(self) -> Dict[str, pd.DataFrame]:
        """Collect comprehensive benefit data using working patterns"""

        logger.info("🚀 Starting comprehensive benefit data collection with recodes...")

        # Define working configurations
        configs = [
            # Universal Credit
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
            # PIP
            {
                'database': 'str:database:PIP_Monthly_new',
                'measure': 'str:count:PIP_Monthly_new:V_F_PIP_MONTHLY',
                'geographic_field': 'str:field:PIP_Monthly_new:V_F_PIP_MONTHLY:COA_2011',
                'time_field': 'str:field:PIP_Monthly_new:F_PIP_DATE:DATE2',
                'name': 'pip_regional'
            },
            # Housing Benefit (known working example)
            {
                'database': 'str:database:hb_new',
                'measure': 'str:count:hb_new:V_F_HB_NEW',
                'geographic_field': 'str:field:hb_new:V_F_HB_NEW:COA_CODE',
                'time_field': 'str:field:hb_new:F_HB_NEW_DATE:NEW_DATE_NAME',
                'name': 'housing_benefit_regional'
            }
        ]

        results = {}

        for config in configs:
            logger.info(f"📊 Collecting: {config['name']}")

            df = self.collect_regional_data_with_recodes(
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
    """Run working regional data collection"""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv('STAT_XPLORE_API_KEY')
    if not api_key:
        logger.error("API key not found in environment variables")
        return

    collector = WorkingRegionalCollector(api_key)

    # Test with the working Housing Benefit pattern first
    logger.info("🧪 Testing with known working Housing Benefit query...")

    hb_df = collector.collect_regional_data_with_recodes(
        'str:database:hb_new',
        'str:count:hb_new:V_F_HB_NEW',
        'str:field:hb_new:V_F_HB_NEW:COA_CODE',
        'str:field:hb_new:F_HB_NEW_DATE:NEW_DATE_NAME',
        'housing_benefit_test'
    )

    if not hb_df.empty:
        logger.info("✅ Housing Benefit test successful - pattern works!")
        logger.info(f"   Records: {len(hb_df)}")
        logger.info(f"   Areas: {hb_df['Geography_Code'].nunique()}")

        # Now try with UC and PIP
        logger.info("\n🎯 Applying working pattern to UC and PIP...")

        all_datasets = collector.collect_comprehensive_benefit_data()

        if all_datasets:
            # Save results
            output_dir = "working_regional_data"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            for name, df in all_datasets.items():
                filename = f"{output_dir}/{name}_{timestamp}.parquet"
                df.to_parquet(filename, index=False)

                csv_filename = f"{output_dir}/{name}_{timestamp}.csv"
                df.to_csv(csv_filename, index=False)

                logger.info(f"💾 Saved {name}: {len(df)} records")

            # Create combined dataset
            combined_df = pd.concat(all_datasets.values(), ignore_index=True)
            combined_filename = f"{output_dir}/all_benefits_regional_{timestamp}.parquet"
            combined_df.to_parquet(combined_filename, index=False)

            print(f"\n🎉 REGIONAL BENEFIT DATA COLLECTION COMPLETE!")
            print(f"📊 Total records: {len(combined_df):,}")
            print(f"🗺️ Geographic areas: {combined_df['Geography_Code'].nunique():,}")
            print(f"💼 Benefit types: {combined_df['Benefit_Type'].unique()}")
            print(f"📅 Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")

            return combined_filename
        else:
            logger.error("No datasets collected")
    else:
        logger.error("Housing Benefit test failed - pattern may not work")

    return None

if __name__ == "__main__":
    main()