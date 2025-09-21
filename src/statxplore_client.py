import os
import requests
import json
import pandas as pd
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
import time

load_dotenv()

class StatXploreClient:
    """Client for accessing DWP Stat-Xplore Open Data API"""

    def __init__(self):
        self.base_url = "https://stat-xplore.dwp.gov.uk/webapi/rest/v1"
        self.api_key = os.getenv('STATXPLORE_API_KEY')

        if not self.api_key:
            raise ValueError("STATXPLORE_API_KEY not found in environment variables")

        self.headers = {
            'APIKey': self.api_key,
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get current rate limit information"""
        try:
            response = self.session.get(f"{self.base_url}/rate_limit")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting rate limit info: {e}")
            return {}

    def get_info(self) -> Dict[str, Any]:
        """Get general information about Stat-Xplore"""
        try:
            response = self.session.get(f"{self.base_url}/info")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting info: {e}")
            return {}

    def get_schema(self, path: str = "") -> Dict[str, Any]:
        """
        Get schema information for datasets

        Args:
            path: Optional path to specific schema element (e.g., dataset ID)
        """
        url = f"{self.base_url}/schema"
        if path:
            url += f"/{path}"

        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting schema for path '{path}': {e}")
            return {}

    def get_all_datasets(self) -> List[Dict[str, Any]]:
        """Get information about all available datasets"""
        root_schema = self.get_schema()
        datasets = []

        if 'children' in root_schema:
            for child in root_schema['children']:
                if child.get('type') == 'FOLDER':
                    # Get datasets within this folder
                    folder_schema = self.get_schema(child['id'])
                    if 'children' in folder_schema:
                        for dataset in folder_schema['children']:
                            if dataset.get('type') == 'DATABASE':
                                datasets.append({
                                    'folder': child['label'],
                                    'folder_id': child['id'],
                                    'dataset_name': dataset['label'],
                                    'dataset_id': dataset['id'],
                                    'location': dataset.get('location', '')
                                })
                elif child.get('type') == 'DATABASE':
                    # Top-level dataset
                    datasets.append({
                        'folder': 'Root',
                        'folder_id': '',
                        'dataset_name': child['label'],
                        'dataset_id': child['id'],
                        'location': child.get('location', '')
                    })

        return datasets

    def get_dataset_fields(self, dataset_id: str) -> Dict[str, Any]:
        """Get detailed field information for a specific dataset"""
        return self.get_schema(dataset_id)

    def query_table(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit a table query to get data

        Args:
            query: JSON query object containing database, measures, and dimensions
        """
        try:
            response = self.session.post(
                f"{self.base_url}/table",
                json=query
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error executing table query: {e}")
            return {}

    def extract_all_raw_data(self, save_to_file: bool = True) -> Dict[str, Any]:
        """
        Extract as much raw data as possible from all available datasets

        Args:
            save_to_file: Whether to save results to JSON files

        Returns:
            Dictionary containing all extracted data
        """
        print("Starting comprehensive data extraction from Stat-Xplore...")

        # Get rate limit info
        rate_info = self.get_rate_limit_info()
        print(f"Rate limit info: {rate_info}")

        # Get all datasets
        print("Discovering all available datasets...")
        datasets = self.get_all_datasets()
        print(f"Found {len(datasets)} datasets")

        all_data = {
            'rate_limit_info': rate_info,
            'system_info': self.get_info(),
            'datasets': datasets,
            'dataset_schemas': {},
            'extracted_data': {}
        }

        # For each dataset, get its schema and attempt basic data extraction
        for i, dataset in enumerate(datasets):
            dataset_id = dataset['dataset_id']
            dataset_name = dataset['dataset_name']

            print(f"Processing dataset {i+1}/{len(datasets)}: {dataset_name}")

            # Get detailed schema for this dataset
            schema = self.get_dataset_fields(dataset_id)
            all_data['dataset_schemas'][dataset_id] = schema

            # Try to extract basic data if the dataset has measures and dimensions
            if 'children' in schema:
                measures = []
                dimensions = []

                for child in schema['children']:
                    if child.get('type') == 'MEASURES':
                        if 'children' in child:
                            measures.extend([m['id'] for m in child['children']])
                    elif child.get('type') == 'FIELD':
                        dimensions.append(child['id'])

                # If we have measures and dimensions, try a basic query
                if measures and dimensions:
                    try:
                        # Create a simple query with first measure and first dimension
                        basic_query = {
                            "database": dataset_id,
                            "measures": measures[:1],  # Just first measure to avoid large results
                            "dimensions": [dimensions[:1]]  # Just first dimension
                        }

                        print(f"  Extracting sample data for {dataset_name}...")
                        sample_data = self.query_table(basic_query)
                        all_data['extracted_data'][dataset_id] = sample_data

                        # Add small delay to respect rate limits
                        time.sleep(0.1)

                    except Exception as e:
                        print(f"  Could not extract data from {dataset_name}: {e}")
                        all_data['extracted_data'][dataset_id] = {"error": str(e)}

            # Check remaining rate limit
            remaining = self.session.get(f"{self.base_url}/rate_limit").json().get('remaining', 0)
            if remaining < 50:
                print(f"Rate limit getting low ({remaining} remaining), slowing down...")
                time.sleep(2)

        if save_to_file:
            # Save to JSON files
            os.makedirs('data/raw', exist_ok=True)

            with open('data/raw/statxplore_full_extraction.json', 'w') as f:
                json.dump(all_data, f, indent=2)

            # Save just the datasets list as CSV for easy viewing
            df = pd.DataFrame(datasets)
            df.to_csv('data/raw/available_datasets.csv', index=False)

            print(f"Data saved to:")
            print(f"  - data/raw/statxplore_full_extraction.json")
            print(f"  - data/raw/available_datasets.csv")

        return all_data

    def get_dataset_summary(self) -> pd.DataFrame:
        """Get a summary of all available datasets as a DataFrame"""
        datasets = self.get_all_datasets()
        return pd.DataFrame(datasets)


def main():
    """Example usage of the StatXploreClient"""
    try:
        client = StatXploreClient()

        print("=== Stat-Xplore Data Extraction ===")
        print(f"API Key loaded: {'Yes' if client.api_key else 'No'}")

        # Extract all available data
        all_data = client.extract_all_raw_data()

        print(f"\nExtraction complete!")
        print(f"Found {len(all_data['datasets'])} datasets")
        print(f"Extracted data from {len(all_data['extracted_data'])} datasets")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()