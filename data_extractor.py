import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
import logging
from typing import Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class StatXploreAPI:
    def __init__(self, api_key, rate_limit_delay=1.0, max_retries=3):
        """
        Initialize the Stat-Xplore API client

        Args:
            api_key (str): Your API key from stat-xplore.dwp.gov.uk
            rate_limit_delay (float): Delay between API calls in seconds
            max_retries (int): Maximum number of retries for failed requests
        """
        self.base_url = "https://stat-xplore.dwp.gov.uk/webapi/rest/v1"
        self.headers = {
            "APIKey": api_key,
            "Content-Type": "application/json"
        }
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.session = requests.Session()

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Dataset configurations based on working Stat-Xplore API queries
        self.datasets = {
            'households': {
                'id': 'str:database:UC_Households',
                'measures': ['str:count:UC_Households:V_F_UC_HOUSEHOLDS'],
                'time_dim': 'str:field:UC_Households:F_UC_HH_DATE:DATE_NAME'
            },
            'people': {
                'id': 'str:database:UC_Monthly',
                'measures': ['str:count:UC_Monthly:V_F_UC_CASELOAD_FULL'],
                'time_dim': 'str:field:UC_Monthly:F_UC_DATE:DATE_NAME'
            }
        }
    
    def _make_request(self, method: str, url: str, **kwargs) -> Optional[Dict]:
        """Make a request with rate limiting and retry logic"""
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.rate_limit_delay)
                response = self.session.request(method, url, headers=self.headers, **kwargs)

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Rate limited - wait longer and retry
                    wait_time = 2 ** attempt * self.rate_limit_delay
                    self.logger.warning(f"Rate limited. Waiting {wait_time}s before retry {attempt + 1}")
                    time.sleep(wait_time)
                    continue
                else:
                    self.logger.error(f"API error {response.status_code}: {response.text}")
                    if attempt == self.max_retries - 1:
                        return None

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(2 ** attempt)

        return None

    def get_schema_info(self):
        """Get information about available datasets and their structure"""
        return self._make_request('GET', f"{self.base_url}/schema")

    def get_dataset_info(self, dataset_id):
        """Get detailed information about a specific dataset"""
        return self._make_request('GET', f"{self.base_url}/schema/{dataset_id}")

    def query_dataset(self, dataset_type: str, start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> Optional[Dict]:
        """
        Query Universal Credit data using correct Stat-Xplore API format

        Args:
            dataset_type (str): Type of dataset ('households', 'people')
            start_date (str): Start date in format 'YYYY-MM' (optional)
            end_date (str): End date in format 'YYYY-MM' (optional)
        """
        if dataset_type not in self.datasets:
            self.logger.error(f"Unknown dataset type: {dataset_type}")
            return None

        config = self.datasets[dataset_type]

        # Simple query with just time dimension (working format)
        query = {
            "database": config['id'],
            "measures": config['measures'],
            "dimensions": [
                [config['time_dim']]  # Just time dimension for now
            ]
        }

        self.logger.info(f"Querying {dataset_type} data...")
        return self._make_request('POST', f"{self.base_url}/table", json=query)
    
    def query_universal_credit_lsoa(self, start_date=None, end_date=None):
        """
        Query Universal Credit data at LSOA level for full history
        
        Args:
            start_date (str): Start date in format 'YYYY-MM' (optional)
            end_date (str): End date in format 'YYYY-MM' (optional)
        """
        
        # Universal Credit dataset ID - you may need to adjust this based on current schema
        # Common UC dataset IDs include:
        dataset_id = "str:database:UC_Households"  # or "str:database:UC_People" 
        
        # Query structure for Universal Credit at LSOA level
        query = {
            "database": dataset_id,
            "measures": [
                "str:count:UC_Households:V_F_UC_HOUSEHOLDS"  # Total UC households
            ],
            "dimensions": [
                [  # Geography - LSOA level
                    {
                        "dimension": "str:field:UC_Households:F_UC_GEOGRAPHY:V_C_GEOGRAPHY_LSOA11",
                        "values": ["*"]  # All LSOAs
                    }
                ],
                [  # Time dimension
                    {
                        "dimension": "str:field:UC_Households:F_UC_DATE:V_C_UC_DATE_MONTHLY",
                        "values": self._get_time_range(start_date, end_date)
                    }
                ]
            ]
        }
        
        response = requests.post(
            f"{self.base_url}/table", 
            headers=self.headers, 
            json=query
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    
    def _get_time_range(self, start_date=None, end_date=None):
        """
        Generate time range for query. If no dates specified, returns all available periods.
        
        Args:
            start_date (str): Start date in format 'YYYY-MM'
            end_date (str): End date in format 'YYYY-MM'
        """
        if start_date is None and end_date is None:
            return ["*"]  # All available time periods
        
        # If specific dates are provided, you would need to construct the time period IDs
        # Format is typically like "str:value:UC_Households:F_UC_DATE:V_C_UC_DATE_MONTHLY:201310"
        time_periods = []
        # Implementation would depend on exact date format requirements
        
        return time_periods if time_periods else ["*"]
    
    def convert_to_dataframe(self, api_response: Dict, dataset_type: str = 'households') -> Optional[pd.DataFrame]:
        """
        Convert API response to a pandas DataFrame with enhanced structure

        Args:
            api_response (dict): Response from the API query
            dataset_type (str): Type of dataset for proper column naming
        """
        if not api_response or 'cubes' not in api_response:
            self.logger.error("Invalid API response - no cubes found")
            return None

        try:
            # Extract cube data - cubes is a dict, not a list
            cubes = api_response['cubes']
            if not cubes:
                self.logger.error("No cubes in response")
                return None

            # Get the first (and likely only) cube
            measure_id = list(cubes.keys())[0]
            cube_data = cubes[measure_id]
            values = cube_data['values']

            # Extract time dimension information from recodes
            recodes = api_response.get('query', {}).get('recodes', {})
            time_periods = []

            # Find time dimension in recodes
            for field_id, recode_info in recodes.items():
                if 'DATE' in field_id:
                    for time_map in recode_info.get('map', []):
                        if time_map:
                            time_value_id = time_map[0]
                            # Extract date from value ID like "C_UC_HH_DATE:201508"
                            date_part = time_value_id.split(':')[-1]
                            if len(date_part) == 6:  # YYYYMM format
                                year = date_part[:4]
                                month = date_part[4:6]
                                time_periods.append(f"{year} {month}")
                    break

            # Create simple time series data structure
            data_rows = []

            # For simple time-series data (one value per time period)
            if isinstance(values, list) and time_periods:
                for i, value in enumerate(values):
                    if i < len(time_periods) and value is not None:
                        # Convert YYYY MM to proper date format
                        time_str = time_periods[i]
                        try:
                            year, month = time_str.split()
                            date_obj = pd.to_datetime(f"{year}-{month.zfill(2)}-01")
                            time_label = date_obj.strftime('%Y %b')
                        except:
                            time_label = time_str
                            date_obj = None

                        data_rows.append({
                            'Geography_Code': 'UK_TOTAL',  # Total for UK
                            'Geography_Name': 'United Kingdom',
                            'Time_Period': time_label,
                            'Date': date_obj,
                            'Value': float(value),
                            'Dataset': dataset_type
                        })

            df = pd.DataFrame(data_rows)

            # Data validation and cleaning
            if not df.empty:
                df = self._validate_and_clean_dataframe(df)
                # Add derived fields
                df['Year'] = df['Date'].dt.year if 'Date' in df.columns else None
                df['Month'] = df['Date'].dt.month if 'Date' in df.columns else None
                df['Quarter'] = df['Date'].dt.quarter if 'Date' in df.columns else None
                df['Month_Name'] = df['Date'].dt.strftime('%B') if 'Date' in df.columns else None
                df['Region'] = 'United Kingdom'

                # Calculate growth rates
                df = df.sort_values('Date')
                df['Previous_Value'] = df['Value'].shift(1)
                df['Growth_Rate'] = ((df['Value'] - df['Previous_Value']) / df['Previous_Value'] * 100).round(2)
                df['Growth_Rate'] = df['Growth_Rate'].replace([np.inf, -np.inf], np.nan)

            self.logger.info(f"Converted to DataFrame: {len(df)} records")
            return df

        except Exception as e:
            self.logger.error(f"Error converting to DataFrame: {e}")
            return None

    def _generate_demo_combinations(self, additional_dims):
        """Generate all combinations of demographic dimensions"""
        if not additional_dims:
            return []

        combinations = []
        # This is a simplified version - you'd need to implement proper combinatorial logic
        # For now, just handle the first additional dimension
        if additional_dims:
            for idx, item in enumerate(additional_dims[0]):
                combinations.append({
                    'indices': [idx],
                    'labels': {'Demographic_Category': item.get('label', ''), 'Demographic_Code': item.get('uris', [None])[0]}
                })
        return combinations

    def _validate_and_clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the dataframe"""
        # Remove rows with null geography codes
        df = df.dropna(subset=['Geography_Code'])

        # Convert time periods to datetime where possible
        try:
            df['Date'] = pd.to_datetime(df['Time_Period'], format='%Y %b', errors='coerce')
        except:
            try:
                df['Date'] = pd.to_datetime(df['Time_Period'], errors='coerce')
            except:
                self.logger.warning("Could not convert time periods to datetime")

        # Ensure numeric columns are properly typed
        numeric_cols = [col for col in df.columns if col.endswith('_Count')]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df
    
    def get_comprehensive_uc_data(self,
                                 datasets: List[str] = None,
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None,
                                 parallel_execution: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Get comprehensive Universal Credit data from working API

        Args:
            datasets (List[str]): List of datasets to collect ['households', 'people']
            start_date (str): Start date in format 'YYYY-MM'
            end_date (str): End date in format 'YYYY-MM'
            parallel_execution (bool): Whether to execute queries in parallel
        """
        if datasets is None:
            datasets = ['households', 'people']

        all_data = {}

        # Generate tasks for each dataset
        query_tasks = []
        for dataset in datasets:
            task_name = dataset
            query_tasks.append({
                'name': task_name,
                'dataset_type': dataset,
                'start_date': start_date,
                'end_date': end_date
            })

        self.logger.info(f"Starting data collection for {len(query_tasks)} combinations")

        if parallel_execution and len(query_tasks) > 1:
            # Execute queries in parallel
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_task = {
                    executor.submit(self._execute_single_query, task): task
                    for task in query_tasks
                }

                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result_df = future.result()
                        if result_df is not None:
                            all_data[task['name']] = result_df
                            self.logger.info(f"Completed {task['name']}: {len(result_df)} records")
                        else:
                            self.logger.warning(f"No data returned for {task['name']}")
                    except Exception as e:
                        self.logger.error(f"Error processing {task['name']}: {e}")
        else:
            # Execute queries sequentially
            for task in query_tasks:
                result_df = self._execute_single_query(task)
                if result_df is not None:
                    all_data[task['name']] = result_df
                    self.logger.info(f"Completed {task['name']}: {len(result_df)} records")

        return all_data

    def _execute_single_query(self, task: Dict) -> Optional[pd.DataFrame]:
        """Execute a single query task"""
        try:
            result = self.query_dataset(
                dataset_type=task['dataset_type'],
                start_date=task['start_date'],
                end_date=task['end_date']
            )

            if result:
                return self.convert_to_dataframe(result, task['dataset_type'])
            return None

        except Exception as e:
            self.logger.error(f"Error in single query {task['name']}: {e}")
            return None

    def save_comprehensive_data(self, data_dict: Dict[str, pd.DataFrame],
                               output_dir: str = "uc_data",
                               file_format: str = "parquet") -> Dict[str, str]:
        """
        Save all collected data to files

        Args:
            data_dict (Dict): Dictionary of DataFrames from get_comprehensive_uc_data
            output_dir (str): Output directory
            file_format (str): File format ('parquet', 'csv', 'feather')
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        for dataset_name, df in data_dict.items():
            if df is not None and not df.empty:
                filename = f"uc_{dataset_name}_{timestamp}.{file_format}"
                filepath = os.path.join(output_dir, filename)

                try:
                    if file_format == 'parquet':
                        df.to_parquet(filepath, index=False)
                    elif file_format == 'csv':
                        df.to_csv(filepath, index=False)
                    elif file_format == 'feather':
                        df.to_feather(filepath)
                    else:
                        raise ValueError(f"Unsupported file format: {file_format}")

                    saved_files[dataset_name] = filepath
                    self.logger.info(f"Saved {dataset_name}: {filepath} ({len(df)} records)")

                except Exception as e:
                    self.logger.error(f"Error saving {dataset_name}: {e}")

        # Create a combined file for dashboard use
        if data_dict:
            try:
                combined_df = self._create_combined_dataset(data_dict)
                if combined_df is not None:
                    combined_filename = f"uc_combined_{timestamp}.{file_format}"
                    combined_filepath = os.path.join(output_dir, combined_filename)

                    if file_format == 'parquet':
                        combined_df.to_parquet(combined_filepath, index=False)
                    elif file_format == 'csv':
                        combined_df.to_csv(combined_filepath, index=False)
                    elif file_format == 'feather':
                        combined_df.to_feather(combined_filepath)

                    saved_files['combined'] = combined_filepath
                    self.logger.info(f"Saved combined dataset: {combined_filepath} ({len(combined_df)} records)")

            except Exception as e:
                self.logger.error(f"Error creating combined dataset: {e}")

        return saved_files

    def _create_combined_dataset(self, data_dict: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Create a combined dataset suitable for dashboard use"""
        try:
            combined_data = []

            for dataset_name, df in data_dict.items():
                if df is not None and not df.empty:
                    # Add dataset identifier
                    df_copy = df.copy()
                    df_copy['Dataset'] = dataset_name

                    # Standardize column names
                    df_copy = self._standardize_columns(df_copy)
                    combined_data.append(df_copy)

            if combined_data:
                return pd.concat(combined_data, ignore_index=True, sort=False)
            return None

        except Exception as e:
            self.logger.error(f"Error combining datasets: {e}")
            return None

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across datasets"""
        # Create mapping for common columns
        column_mapping = {}

        for col in df.columns:
            if col.endswith('_Count'):
                column_mapping[col] = 'Value'
            elif 'Geography' in col and 'Code' in col:
                column_mapping[col] = 'Geography_Code'
            elif 'Geography' in col and 'Name' in col:
                column_mapping[col] = 'Geography_Name'

        return df.rename(columns=column_mapping)

# Example usage functions
def collect_comprehensive_uc_data(api_key: str, output_dir: str = "uc_data"):
    """
    Comprehensive Universal Credit data collection for dashboard use

    Args:
        api_key (str): Your API key from stat-xplore.dwp.gov.uk
        output_dir (str): Directory to save the collected data
    """
    print("🔄 Starting comprehensive Universal Credit data collection...")

    # Initialize the API client with enhanced settings
    api = StatXploreAPI(api_key, rate_limit_delay=1.5, max_retries=3)

    # Define what data to collect
    datasets_to_collect = ['households', 'people']  # Core datasets

    print(f"📊 Collecting data for:")
    print(f"  - Datasets: {', '.join(datasets_to_collect)}")
    print(f"  - Output directory: {output_dir}")

    try:
        # Collect comprehensive data
        all_data = api.get_comprehensive_uc_data(
            datasets=datasets_to_collect,
            parallel_execution=True
        )

        if all_data:
            print(f"\n✅ Successfully collected {len(all_data)} datasets:")
            total_records = 0
            for name, df in all_data.items():
                records = len(df) if df is not None else 0
                total_records += records
                print(f"  - {name}: {records:,} records")

            print(f"\n💾 Saving data (Total: {total_records:,} records)...")

            # Save data in multiple formats for flexibility
            saved_files = {}

            # Save as Parquet (efficient for analytics)
            parquet_files = api.save_comprehensive_data(
                all_data,
                output_dir=output_dir,
                file_format="parquet"
            )
            saved_files.update({f"{k}_parquet": v for k, v in parquet_files.items()})

            # Save as CSV (human readable)
            csv_files = api.save_comprehensive_data(
                all_data,
                output_dir=output_dir,
                file_format="csv"
            )
            saved_files.update({f"{k}_csv": v for k, v in csv_files.items()})

            print(f"\n📂 Files saved:")
            for file_type, filepath in saved_files.items():
                print(f"  - {file_type}: {filepath}")

            # Data summary
            if 'combined' in all_data and all_data['combined'] is not None:
                df = all_data['combined']
                print(f"\n📈 Data Summary:")
                print(f"  - Total records: {len(df):,}")
                print(f"  - Geographic areas: {df['Geography_Code'].nunique():,}")
                print(f"  - Date range: {df['Time_Period'].min()} to {df['Time_Period'].max()}")
                print(f"  - Datasets included: {df['Dataset'].unique().tolist()}")

                # Latest month summary
                if 'Date' in df.columns:
                    latest_data = df[df['Date'] == df['Date'].max()]
                    if not latest_data.empty and 'Value' in latest_data.columns:
                        total_latest = latest_data['Value'].sum()
                        print(f"  - Total UC count (latest month): {total_latest:,}")

            return saved_files

        else:
            print("❌ No data was collected")
            return {}

    except Exception as e:
        print(f"❌ Error during data collection: {e}")
        return {}

def explore_schema(api_key: str):
    """Explore available datasets and their structure"""
    print("🔍 Exploring Stat-Xplore schema...")

    api = StatXploreAPI(api_key)

    try:
        schema = api.get_schema_info()
        if schema:
            print("📊 Available databases:")
            databases = schema.get('databases', [])
            for i, db in enumerate(databases, 1):
                print(f"  {i}. {db}")

            # Get detailed info for UC databases
            uc_databases = [db for db in databases if 'UC' in db or 'Universal' in db]
            if uc_databases:
                print(f"\n🎯 Universal Credit databases found: {len(uc_databases)}")
                for db in uc_databases[:3]:  # Limit to first 3 to avoid too much output
                    print(f"\n📋 Details for {db}:")
                    details = api.get_dataset_info(db)
                    if details:
                        if 'measures' in details:
                            print(f"  Measures: {len(details['measures'])}")
                        if 'dimensions' in details:
                            print(f"  Dimensions: {len(details['dimensions'])}")
        else:
            print("❌ Could not retrieve schema information")

    except Exception as e:
        print(f"❌ Error exploring schema: {e}")

def main():
    """
    Main function - demonstrates comprehensive Universal Credit data collection
    """
    print("🚀 Universal Credit Data Extractor - Enhanced Version")
    print("=" * 60)

    # Load API key from environment
    API_KEY = os.getenv('STAT_XPLORE_API_KEY')

    if not API_KEY:
        print("⚠️  Please set your API key in .env file:")
        print("   STAT_XPLORE_API_KEY=your_actual_api_key")
        print("   Get your API key from https://stat-xplore.dwp.gov.uk")
        return

    # Explore available data first (optional)
    explore_schema(API_KEY)

    print("\n" + "=" * 60)

    # Collect comprehensive data for dashboard
    saved_files = collect_comprehensive_uc_data(API_KEY, output_dir="uc_dashboard_data")

    if saved_files:
        print("\n🎯 Next Steps:")
        print("1. Use the combined parquet file for your Dash app (most efficient)")
        print("2. The CSV files are available for manual inspection")
        print("3. Each geographic level and dataset type is saved separately")
        print("4. The combined file includes all data with standardized columns")
        print("\n📖 Dashboard Development Tips:")
        print("- Use 'Geography_Code' for mapping/filtering by area")
        print("- Use 'Date' column for time series analysis")
        print("- Use 'Dataset' column to distinguish data types")
        print("- 'Value' column contains the actual counts")
    else:
        print("\n❌ Data collection failed. Check your API key and connection.")

if __name__ == "__main__":
    main()

# Additional helper functions for data analysis

def analyze_regional_trends(df, region_filter=None):
    """
    Analyze trends in Universal Credit claims over time
    
    Args:
        df (pd.DataFrame): UC data from API
        region_filter (list): Optional list of LSOA codes to filter
    """
    if region_filter:
        df = df[df['LSOA_Code'].isin(region_filter)]
    
    # Convert Time_Period to datetime if needed
    df['Date'] = pd.to_datetime(df['Time_Period'], format='%Y-%m', errors='coerce')
    
    # Monthly totals
    monthly_totals = df.groupby('Time_Period')['UC_Households'].sum().reset_index()
    
    # Growth rates
    monthly_totals['Growth_Rate'] = monthly_totals['UC_Households'].pct_change() * 100
    
    return monthly_totals

def get_top_lsoas_by_claims(df, n=20):
    """
    Get the top N LSOAs by Universal Credit households (latest data)
    """
    latest_month = df['Time_Period'].max()
    latest_data = df[df['Time_Period'] == latest_month]
    
    top_lsoas = latest_data.nlargest(n, 'UC_Households')[
        ['LSOA_Code', 'LSOA_Name', 'UC_Households']
    ]
    
    return top_lsoas