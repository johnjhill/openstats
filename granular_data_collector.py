#!/usr/bin/env python3
"""
Comprehensive granular Universal Credit data collector
Designed to extract maximum detail while handling rate limits
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import requests

load_dotenv()

class GranularUCCollector:
    def __init__(self, api_key: str, base_delay: float = 2.0, max_retries: int = 5):
        """
        Initialize granular UC data collector with aggressive rate limiting

        Args:
            api_key: Stat-Xplore API key
            base_delay: Base delay between requests (seconds)
            max_retries: Maximum retry attempts
        """
        self.base_url = "https://stat-xplore.dwp.gov.uk/webapi/rest/v1"
        self.headers = {
            "APIKey": api_key,
            "Content-Type": "application/json"
        }
        self.base_delay = base_delay
        self.max_retries = max_retries
        self.session = requests.Session()

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load comprehensive schema
        try:
            with open('comprehensive_uc_schema_with_geo.json', 'r') as f:
                self.schema = json.load(f)
        except:
            self.logger.error("Schema file not found. Run comprehensive_schema_explorer.py first")
            self.schema = {}

        # Track API call counts and timing
        self.api_calls = 0
        self.last_call_time = 0
        self.rate_limit_tracker = {}

    def _make_request_with_backoff(self, method: str, url: str, **kwargs) -> Optional[Dict]:
        """Enhanced request method with exponential backoff and rate limiting"""

        # Implement adaptive delay based on recent API calls
        current_time = time.time()
        time_since_last = current_time - self.last_call_time

        if time_since_last < self.base_delay:
            sleep_time = self.base_delay - time_since_last
            self.logger.info(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)

        for attempt in range(self.max_retries):
            try:
                self.api_calls += 1
                self.last_call_time = time.time()

                response = self.session.request(method, url, headers=self.headers, **kwargs)

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Rate limited - implement exponential backoff
                    wait_time = (2 ** attempt) * self.base_delay * 2
                    self.logger.warning(f"Rate limited (429). Waiting {wait_time}s before retry {attempt + 1}")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 503:
                    # Service unavailable
                    wait_time = (2 ** attempt) * self.base_delay * 3
                    self.logger.warning(f"Service unavailable (503). Waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                else:
                    self.logger.error(f"API error {response.status_code}: {response.text}")
                    if attempt == self.max_retries - 1:
                        return None
                    time.sleep((2 ** attempt) * self.base_delay)

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    return None
                time.sleep((2 ** attempt) * self.base_delay)

        return None

    def get_geographic_field_values(self, field_id: str, limit: int = 50) -> List[Dict]:
        """Get available values for a geographic field"""

        self.logger.info(f"Exploring geographic field: {field_id}")

        field_details = self._make_request_with_backoff('GET', f"{self.base_url}/schema/{field_id}")

        if not field_details or 'children' not in field_details:
            return []

        values = []
        for child in field_details['children'][:limit]:  # Limit to avoid overwhelming
            if child.get('type') == 'VALUE':
                values.append({
                    'id': child.get('id', ''),
                    'label': child.get('label', ''),
                    'code': child.get('uris', [None])[0] if child.get('uris') else None
                })

        self.logger.info(f"Found {len(values)} geographic values for {field_id}")
        return values

    def query_with_geography(self, database_id: str, measure_id: str,
                           time_field_id: str, geo_field_id: str,
                           geo_values: List[str] = None,
                           demographic_field_id: str = None,
                           demographic_values: List[str] = None) -> Optional[Dict]:
        """
        Query with specific geographic and demographic breakdowns
        """

        # Build basic query structure
        query = {
            "database": database_id,
            "measures": [measure_id],
            "dimensions": [[time_field_id]]  # Always include time
        }

        # Add geography dimension if specified
        if geo_field_id:
            if geo_values:
                # Use recodes to filter specific geographic areas
                query["recodes"] = {
                    geo_field_id: {
                        "map": [[value] for value in geo_values[:20]],  # Limit to 20 areas
                        "total": False
                    }
                }
            query["dimensions"].append([geo_field_id])

        # Add demographic dimension if specified
        if demographic_field_id:
            if demographic_values:
                if "recodes" not in query:
                    query["recodes"] = {}
                query["recodes"][demographic_field_id] = {
                    "map": [[value] for value in demographic_values[:10]],  # Limit demographics
                    "total": False
                }
            query["dimensions"].append([demographic_field_id])

        self.logger.info(f"Querying {database_id} with geography and demographics")
        return self._make_request_with_backoff('POST', f"{self.base_url}/table", json=query)

    def collect_households_granular_data(self) -> pd.DataFrame:
        """Collect detailed households data with geographic and demographic breakdowns"""

        households_config = self.schema.get('str:database:UC_Households', {})
        if not households_config:
            self.logger.error("Households database not found in schema")
            return pd.DataFrame()

        # Get basic data first
        basic_query = {
            "database": "str:database:UC_Households",
            "measures": ["str:count:UC_Households:V_F_UC_HOUSEHOLDS"],
            "dimensions": [["str:field:UC_Households:F_UC_HH_DATE:DATE_NAME"]]
        }

        self.logger.info("Collecting basic households data...")
        basic_result = self._make_request_with_backoff('POST', f"{self.base_url}/table", json=basic_query)

        all_dataframes = []

        if basic_result:
            basic_df = self._convert_to_dataframe(basic_result, 'households_total', {})
            if basic_df is not None:
                all_dataframes.append(basic_df)

        # Collect geographic breakdowns
        geo_groups = households_config.get('geography_details', {})

        for group_id, group_info in geo_groups.items():
            if 'residence-based' in group_info['group_label']:  # Focus on residence-based geography
                for level in group_info.get('levels', []):
                    geo_field_id = level['id']

                    self.logger.info(f"Collecting data for geographic level: {level['label']}")

                    # Get some geographic values
                    geo_values = self.get_geographic_field_values(geo_field_id, limit=20)
                    if geo_values:
                        geo_value_ids = [v['id'] for v in geo_values[:15]]  # Limit to 15 areas

                        # Query with geographic breakdown
                        geo_result = self.query_with_geography(
                            database_id="str:database:UC_Households",
                            measure_id="str:count:UC_Households:V_F_UC_HOUSEHOLDS",
                            time_field_id="str:field:UC_Households:F_UC_HH_DATE:DATE_NAME",
                            geo_field_id=geo_field_id,
                            geo_values=geo_value_ids
                        )

                        if geo_result:
                            geo_df = self._convert_to_dataframe(geo_result, 'households_geographic', {
                                'geographic_level': level['label'],
                                'geographic_field': geo_field_id
                            })
                            if geo_df is not None:
                                all_dataframes.append(geo_df)

                        # Add delay to avoid overwhelming the API
                        time.sleep(self.base_delay * 1.5)

        # Collect demographic breakdowns
        demo_fields = households_config.get('demographic_fields', [])
        for demo_field in demo_fields:
            demo_field_id = demo_field['id']

            self.logger.info(f"Collecting demographic data: {demo_field['label']}")

            # Query with demographic breakdown
            demo_result = self.query_with_geography(
                database_id="str:database:UC_Households",
                measure_id="str:count:UC_Households:V_F_UC_HOUSEHOLDS",
                time_field_id="str:field:UC_Households:F_UC_HH_DATE:DATE_NAME",
                geo_field_id=None,
                demographic_field_id=demo_field_id
            )

            if demo_result:
                demo_df = self._convert_to_dataframe(demo_result, 'households_demographic', {
                    'demographic_type': demo_field['label'],
                    'demographic_field': demo_field_id
                })
                if demo_df is not None:
                    all_dataframes.append(demo_df)

            time.sleep(self.base_delay)

        # Combine all data
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
            self.logger.info(f"Collected total of {len(combined_df)} granular records")
            return combined_df

        return pd.DataFrame()

    def collect_people_granular_data(self) -> pd.DataFrame:
        """Collect detailed people data with all available breakdowns"""

        people_config = self.schema.get('str:database:UC_Monthly', {})
        if not people_config:
            self.logger.error("People database not found in schema")
            return pd.DataFrame()

        all_dataframes = []

        # Basic people data
        basic_query = {
            "database": "str:database:UC_Monthly",
            "measures": ["str:count:UC_Monthly:V_F_UC_CASELOAD_FULL"],
            "dimensions": [["str:field:UC_Monthly:F_UC_DATE:DATE_NAME"]]
        }

        self.logger.info("Collecting basic people data...")
        basic_result = self._make_request_with_backoff('POST', f"{self.base_url}/table", json=basic_query)

        if basic_result:
            basic_df = self._convert_to_dataframe(basic_result, 'people_total', {})
            if basic_df is not None:
                all_dataframes.append(basic_df)

        # Collect demographic breakdowns for people
        demo_fields = people_config.get('demographic_fields', [])
        for demo_field in demo_fields[:3]:  # Limit to avoid rate limits
            demo_field_id = demo_field['id']

            self.logger.info(f"Collecting people demographic data: {demo_field['label']}")

            demo_result = self.query_with_geography(
                database_id="str:database:UC_Monthly",
                measure_id="str:count:UC_Monthly:V_F_UC_CASELOAD_FULL",
                time_field_id="str:field:UC_Monthly:F_UC_DATE:DATE_NAME",
                geo_field_id=None,
                demographic_field_id=demo_field_id
            )

            if demo_result:
                demo_df = self._convert_to_dataframe(demo_result, 'people_demographic', {
                    'demographic_type': demo_field['label'],
                    'demographic_field': demo_field_id
                })
                if demo_df is not None:
                    all_dataframes.append(demo_df)

            time.sleep(self.base_delay)

        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
            return combined_df

        return pd.DataFrame()

    def _convert_to_dataframe(self, api_response: Dict, data_type: str, metadata: Dict) -> Optional[pd.DataFrame]:
        """Convert API response to DataFrame with metadata"""

        if not api_response or 'cubes' not in api_response:
            return None

        try:
            cubes = api_response['cubes']
            if not cubes:
                return None

            # Get the first cube
            measure_id = list(cubes.keys())[0]
            cube_data = cubes[measure_id]
            values = cube_data['values']

            # Extract time periods from recodes
            recodes = api_response.get('query', {}).get('recodes', {})
            time_periods = []
            geographic_areas = []
            demographic_categories = []

            # Parse dimensions based on the query structure
            dimensions = api_response.get('query', {}).get('dimensions', [])

            # Extract time dimension
            for field_id, recode_info in recodes.items():
                if 'DATE' in field_id:
                    for time_map in recode_info.get('map', []):
                        if time_map:
                            time_value_id = time_map[0]
                            date_part = time_value_id.split(':')[-1]
                            if len(date_part) == 6:
                                year = date_part[:4]
                                month = date_part[4:6]
                                time_periods.append(f"{year} {month}")
                elif 'GEOGRAPHY' in field_id.upper() or any(geo in field_id.upper() for geo in ['POSTCODE', 'LA', 'WARD']):
                    for geo_map in recode_info.get('map', []):
                        if geo_map:
                            geographic_areas.append(geo_map[0])
                else:
                    # Demographic or other dimension
                    for demo_map in recode_info.get('map', []):
                        if demo_map:
                            demographic_categories.append(demo_map[0])

            # Create DataFrame based on data structure
            data_rows = []

            if isinstance(values, list):
                if len(dimensions) == 1:  # Time only
                    for i, value in enumerate(values):
                        if i < len(time_periods) and value is not None:
                            time_str = time_periods[i]
                            try:
                                year, month = time_str.split()
                                date_obj = pd.to_datetime(f"{year}-{month.zfill(2)}-01")
                                time_label = date_obj.strftime('%Y %b')
                            except:
                                time_label = time_str
                                date_obj = None

                            row = {
                                'Geography_Code': metadata.get('geographic_level', 'UK_TOTAL'),
                                'Geography_Name': metadata.get('geographic_level', 'United Kingdom'),
                                'Time_Period': time_label,
                                'Date': date_obj,
                                'Value': float(value),
                                'Dataset': data_type,
                                'Data_Type': data_type,
                                'Measure': measure_id.split(':')[-1] if ':' in measure_id else measure_id
                            }

                            # Add metadata
                            row.update(metadata)
                            data_rows.append(row)

                elif len(dimensions) == 2:  # Time + one other dimension
                    # Handle 2D array
                    if isinstance(values[0], list):
                        for i, time_values in enumerate(values):
                            time_period = time_periods[i] if i < len(time_periods) else f"Period_{i}"

                            for j, value in enumerate(time_values):
                                if value is not None:
                                    try:
                                        year, month = time_period.split()
                                        date_obj = pd.to_datetime(f"{year}-{month.zfill(2)}-01")
                                        time_label = date_obj.strftime('%Y %b')
                                    except:
                                        time_label = time_period
                                        date_obj = None

                                    # Determine the other dimension
                                    other_value = (geographic_areas[j] if j < len(geographic_areas)
                                                 else demographic_categories[j] if j < len(demographic_categories)
                                                 else f"Category_{j}")

                                    row = {
                                        'Geography_Code': other_value if geographic_areas else 'UK_TOTAL',
                                        'Geography_Name': other_value if geographic_areas else 'United Kingdom',
                                        'Time_Period': time_label,
                                        'Date': date_obj,
                                        'Value': float(value),
                                        'Dataset': data_type,
                                        'Category': other_value if demographic_categories else None,
                                        'Measure': measure_id.split(':')[-1] if ':' in measure_id else measure_id
                                    }

                                    row.update(metadata)
                                    data_rows.append(row)

            df = pd.DataFrame(data_rows)

            if not df.empty:
                # Add derived fields
                if 'Date' in df.columns:
                    df['Year'] = df['Date'].dt.year
                    df['Month'] = df['Date'].dt.month
                    df['Quarter'] = df['Date'].dt.quarter
                    df['Month_Name'] = df['Date'].dt.strftime('%B')

                # Calculate growth rates
                if 'Value' in df.columns and 'Date' in df.columns:
                    df = df.sort_values(['Geography_Code', 'Date'])
                    df['Previous_Value'] = df.groupby('Geography_Code')['Value'].shift(1)
                    df['Growth_Rate'] = ((df['Value'] - df['Previous_Value']) / df['Previous_Value'] * 100).round(2)
                    df['Growth_Rate'] = df['Growth_Rate'].replace([np.inf, -np.inf], np.nan)

            return df

        except Exception as e:
            self.logger.error(f"Error converting to DataFrame: {e}")
            return None

    def collect_all_granular_data(self, output_dir: str = "granular_uc_data") -> Dict[str, str]:
        """Collect all available granular UC data"""

        os.makedirs(output_dir, exist_ok=True)

        self.logger.info("🚀 Starting comprehensive granular UC data collection...")
        self.logger.info(f"📊 Target: Maximum detail with rate limit management")

        saved_files = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Collect households data
        self.logger.info("📊 Collecting granular households data...")
        households_df = self.collect_households_granular_data()

        if not households_df.empty:
            households_file = f"{output_dir}/uc_households_granular_{timestamp}.parquet"
            households_df.to_parquet(households_file, index=False)
            saved_files['households_granular'] = households_file
            self.logger.info(f"✅ Saved households data: {len(households_df)} records")

        # Add delay between major collections
        time.sleep(self.base_delay * 2)

        # Collect people data
        self.logger.info("📊 Collecting granular people data...")
        people_df = self.collect_people_granular_data()

        if not people_df.empty:
            people_file = f"{output_dir}/uc_people_granular_{timestamp}.parquet"
            people_df.to_parquet(people_file, index=False)
            saved_files['people_granular'] = people_file
            self.logger.info(f"✅ Saved people data: {len(people_df)} records")

        # Create combined dataset
        all_dfs = [df for df in [households_df, people_df] if not df.empty]
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True, sort=False)
            combined_file = f"{output_dir}/uc_granular_combined_{timestamp}.parquet"
            combined_df.to_parquet(combined_file, index=False)
            saved_files['combined_granular'] = combined_file

            # Also save as CSV for inspection
            csv_file = f"{output_dir}/uc_granular_combined_{timestamp}.csv"
            combined_df.to_csv(csv_file, index=False)
            saved_files['combined_csv'] = csv_file

            self.logger.info(f"✅ Combined dataset: {len(combined_df)} total records")

        # Log collection statistics
        self.logger.info(f"📈 Collection Statistics:")
        self.logger.info(f"  - Total API calls made: {self.api_calls}")
        self.logger.info(f"  - Files created: {len(saved_files)}")
        if all_dfs:
            self.logger.info(f"  - Total records: {sum(len(df) for df in all_dfs)}")
            self.logger.info(f"  - Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
            self.logger.info(f"  - Unique geographies: {combined_df['Geography_Code'].nunique()}")

        return saved_files

def main():
    """Run comprehensive granular data collection"""
    api_key = os.getenv('STAT_XPLORE_API_KEY')

    if not api_key:
        print("❌ API key not found in .env file")
        return

    collector = GranularUCCollector(api_key, base_delay=2.5, max_retries=5)
    saved_files = collector.collect_all_granular_data()

    if saved_files:
        print("\n🎯 Granular Data Collection Complete!")
        print("📂 Files created:")
        for file_type, filepath in saved_files.items():
            print(f"  - {file_type}: {filepath}")
    else:
        print("❌ No data was collected")

if __name__ == '__main__':
    main()