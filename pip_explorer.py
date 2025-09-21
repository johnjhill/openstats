#!/usr/bin/env python3
"""
PIP Data Explorer for Stat-Xplore API
Investigate Personal Independence Payment data availability and geographic breakdowns
"""

import requests
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PIPDataExplorer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://stat-xplore.dwp.gov.uk/webapi/rest/v1"
        self.headers = {
            'APIKey': api_key,
            'Content-Type': 'application/json'
        }
        self.rate_limit_delay = 1.5

    def rate_limit(self):
        """Apply rate limiting between requests"""
        time.sleep(self.rate_limit_delay)

    def get_schema(self) -> Dict[str, Any]:
        """Get the full API schema to find PIP datasets"""
        url = f"{self.base_url}/schema"

        try:
            self.rate_limit()
            response = requests.get(url, headers=self.headers)

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get schema: {response.status_code} - {response.text}")
                return {}

        except Exception as e:
            logger.error(f"Error getting schema: {e}")
            return {}

    def find_pip_datasets(self, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find all PIP-related datasets in the schema"""
        pip_datasets = []

        def search_folder(folder, path=""):
            if 'children' in folder:
                for child in folder['children']:
                    current_path = f"{path}/{child['label']}" if path else child['label']

                    # Look for PIP-related folders/datasets
                    if any(keyword in child['label'].lower() for keyword in ['pip', 'personal independence', 'disability']):
                        if child.get('type') == 'database':
                            pip_datasets.append({
                                'id': child['id'],
                                'label': child['label'],
                                'path': current_path
                            })
                        elif child.get('type') == 'folder':
                            logger.info(f"🔍 Found PIP folder: {current_path}")
                            search_folder(child, current_path)

                    # Always recurse into folders to find nested PIP data
                    if child.get('type') == 'folder':
                        search_folder(child, current_path)

        if 'children' in schema:
            search_folder(schema)

        return pip_datasets

    def explore_database(self, database_id: str) -> Dict[str, Any]:
        """Explore a specific database to understand its structure"""
        url = f"{self.base_url}/schema/{database_id}"

        try:
            self.rate_limit()
            response = requests.get(url, headers=self.headers)

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to explore database {database_id}: {response.status_code}")
                return {}

        except Exception as e:
            logger.error(f"Error exploring database {database_id}: {e}")
            return {}

    def analyze_geography_groups(self, database_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze geographic groups and levels available in a database"""
        geography_analysis = {
            'groups': [],
            'levels': [],
            'has_regional': False,
            'geographic_fields': []
        }

        def search_fields(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == 'label' and isinstance(value, str):
                        label_lower = value.lower()
                        if any(geo_term in label_lower for geo_term in
                              ['geography', 'region', 'local authority', 'area', 'ward', 'postcode', 'lsoa']):
                            geography_analysis['geographic_fields'].append({
                                'path': prefix,
                                'label': value,
                                'id': obj.get('id', 'unknown')
                            })

                            # Check if this indicates regional data
                            if any(regional_term in label_lower for regional_term in
                                  ['region', 'local authority', 'ward', 'lsoa', 'area']):
                                geography_analysis['has_regional'] = True

                    if isinstance(value, (dict, list)):
                        search_fields(value, f"{prefix}.{key}" if prefix else key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    search_fields(item, f"{prefix}[{i}]" if prefix else f"[{i}]")

        search_fields(database_info)
        return geography_analysis

    def get_time_dimensions(self, database_info: Dict[str, Any]) -> List[Dict[str, str]]:
        """Find time dimensions in the database"""
        time_fields = []

        def search_time_fields(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == 'label' and isinstance(value, str):
                        label_lower = value.lower()
                        if any(time_term in label_lower for time_term in
                              ['date', 'time', 'month', 'year', 'period', 'quarter']):
                            time_fields.append({
                                'path': prefix,
                                'label': value,
                                'id': obj.get('id', 'unknown')
                            })

                    if isinstance(value, (dict, list)):
                        search_time_fields(value, f"{prefix}.{key}" if prefix else key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    search_time_fields(item, f"{prefix}[{i}]" if prefix else f"[{i}]")

        search_time_fields(database_info)
        return time_fields

    def test_geographic_data_access(self, database_id: str, measure_id: str,
                                  geographic_field_id: str, time_field_id: str = None) -> Dict[str, Any]:
        """Test if we can actually access geographic data from a database"""
        url = f"{self.base_url}/table"

        dimensions = [geographic_field_id]
        if time_field_id:
            dimensions.append(time_field_id)

        payload = {
            "database": database_id,
            "measures": [measure_id] if measure_id else [],
            "dimensions": dimensions,
            "recodes": {}
        }

        try:
            self.rate_limit()
            logger.info(f"🧪 Testing data access for {database_id} with geography {geographic_field_id}")
            response = requests.post(url, json=payload, headers=self.headers)

            result = {
                'success': False,
                'status_code': response.status_code,
                'error': None,
                'geographic_areas': 0,
                'time_periods': 0,
                'has_data': False
            }

            if response.status_code == 200:
                data = response.json()
                result['success'] = True
                result['has_data'] = 'cubes' in data and bool(data['cubes'])

                # Count geographic areas and time periods
                if 'cubes' in data:
                    for cube_id, cube_data in data['cubes'].items():
                        if 'dimensions' in cube_data:
                            for dim in cube_data['dimensions']:
                                if dim.get('field') == geographic_field_id:
                                    result['geographic_areas'] = len(dim.get('items', []))
                                elif time_field_id and dim.get('field') == time_field_id:
                                    result['time_periods'] = len(dim.get('items', []))
            else:
                try:
                    error_data = response.json()
                    result['error'] = error_data.get('message', response.text)
                except:
                    result['error'] = response.text

            return result

        except Exception as e:
            return {
                'success': False,
                'status_code': None,
                'error': str(e),
                'geographic_areas': 0,
                'time_periods': 0,
                'has_data': False
            }

    def comprehensive_pip_analysis(self) -> Dict[str, Any]:
        """Perform comprehensive analysis of PIP data availability"""
        logger.info("🚀 Starting comprehensive PIP data analysis...")

        # 1. Get schema and find PIP datasets
        logger.info("📋 Getting API schema...")
        schema = self.get_schema()

        if not schema:
            logger.error("Failed to get schema")
            return {}

        logger.info("🔍 Searching for PIP datasets...")
        pip_datasets = self.find_pip_datasets(schema)

        if not pip_datasets:
            logger.warning("No PIP datasets found in schema")
            return {'datasets': [], 'analysis': 'No PIP datasets found'}

        logger.info(f"📊 Found {len(pip_datasets)} PIP datasets:")
        for dataset in pip_datasets:
            logger.info(f"  - {dataset['label']} (ID: {dataset['id']})")

        # 2. Analyze each dataset
        analysis_results = {
            'datasets': pip_datasets,
            'detailed_analysis': {},
            'regional_data_available': False,
            'best_candidates': [],
            'summary': {}
        }

        for dataset in pip_datasets:
            logger.info(f"\n📊 Analyzing dataset: {dataset['label']}")

            # Get detailed database info
            db_info = self.explore_database(dataset['id'])

            if not db_info:
                continue

            # Analyze geography
            geo_analysis = self.analyze_geography_groups(db_info)

            # Find time dimensions
            time_dimensions = self.get_time_dimensions(db_info)

            # Find measures
            measures = []
            if 'children' in db_info:
                for child in db_info['children']:
                    if child.get('type') == 'valueset':
                        measures.append({
                            'id': child['id'],
                            'label': child['label']
                        })

            dataset_analysis = {
                'database_info': db_info,
                'geography': geo_analysis,
                'time_dimensions': time_dimensions,
                'measures': measures,
                'access_tests': []
            }

            # Test data access if we have geographic fields
            if geo_analysis['geographic_fields'] and measures:
                measure_id = measures[0]['id']

                for geo_field in geo_analysis['geographic_fields'][:3]:  # Test up to 3 geo fields
                    time_field_id = time_dimensions[0]['id'] if time_dimensions else None

                    access_test = self.test_geographic_data_access(
                        dataset['id'],
                        measure_id,
                        geo_field['id'],
                        time_field_id
                    )

                    access_test['geographic_field'] = geo_field
                    access_test['time_field'] = time_dimensions[0] if time_dimensions else None
                    dataset_analysis['access_tests'].append(access_test)

                    if access_test['success'] and access_test['geographic_areas'] > 1:
                        analysis_results['regional_data_available'] = True
                        analysis_results['best_candidates'].append({
                            'dataset': dataset,
                            'geographic_field': geo_field,
                            'time_field': time_dimensions[0] if time_dimensions else None,
                            'access_result': access_test
                        })

            analysis_results['detailed_analysis'][dataset['id']] = dataset_analysis

        # 3. Create summary
        total_datasets = len(pip_datasets)
        datasets_with_geo = len([d for d in analysis_results['detailed_analysis'].values()
                               if d['geography']['has_regional']])
        datasets_with_access = len(analysis_results['best_candidates'])

        analysis_results['summary'] = {
            'total_pip_datasets': total_datasets,
            'datasets_with_geography': datasets_with_geo,
            'datasets_with_regional_access': datasets_with_access,
            'regional_data_available': analysis_results['regional_data_available']
        }

        return analysis_results

def main():
    """Run PIP data exploration"""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv('STAT_XPLORE_API_KEY')
    if not api_key:
        logger.error("API key not found in environment variables")
        return

    explorer = PIPDataExplorer(api_key)

    # Run comprehensive analysis
    results = explorer.comprehensive_pip_analysis()

    # Save results
    output_file = f"pip_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print("\n" + "="*80)
    print("🎯 PIP DATA ANALYSIS RESULTS")
    print("="*80)

    if 'summary' in results:
        summary = results['summary']
        print(f"📊 Total PIP datasets found: {summary['total_pip_datasets']}")
        print(f"🗺️ Datasets with geography: {summary['datasets_with_geography']}")
        print(f"✅ Datasets with regional access: {summary['datasets_with_regional_access']}")
        print(f"🎯 Regional data available: {summary['regional_data_available']}")

    if results.get('best_candidates'):
        print(f"\n🌟 BEST CANDIDATES FOR REGIONAL PIP DATA:")
        for i, candidate in enumerate(results['best_candidates'], 1):
            dataset = candidate['dataset']
            geo_field = candidate['geographic_field']
            access = candidate['access_result']

            print(f"\n{i}. {dataset['label']}")
            print(f"   Geographic Field: {geo_field['label']}")
            print(f"   Geographic Areas: {access['geographic_areas']}")
            print(f"   Time Periods: {access['time_periods']}")
            print(f"   Database ID: {dataset['id']}")

    if not results.get('regional_data_available'):
        print("\n❌ No regional PIP data access found through public API")
        print("   - All datasets may only provide national totals")
        print("   - Regional data may require enhanced API access")
        print("   - Consider contacting DWP for research access")

    print(f"\n📁 Detailed results saved to: {output_file}")

    return results

if __name__ == "__main__":
    main()