#!/usr/bin/env python3
"""
Comprehensive schema explorer for maximum UC data collection
"""

import os
import json
from dotenv import load_dotenv
from data_extractor import StatXploreAPI

load_dotenv()

def explore_all_uc_datasets():
    """Explore all UC datasets and their capabilities"""
    api_key = os.getenv('STAT_XPLORE_API_KEY')
    if not api_key:
        print("❌ API key not found")
        return

    api = StatXploreAPI(api_key)

    # Get all UC databases
    schema = api.get_schema_info()
    uc_folder = None

    for child in schema.get('children', []):
        if child.get('label') == 'Universal Credit':
            uc_folder = child
            break

    if not uc_folder:
        print("❌ UC folder not found")
        return

    uc_details = api.get_dataset_info(uc_folder['id'])
    databases = [child for child in uc_details.get('children', []) if child.get('type') == 'DATABASE']

    print(f"🔍 Found {len(databases)} UC databases:")

    all_capabilities = {}

    for db in databases:
        db_id = db['id']
        db_label = db['label']
        print(f"\n📊 Exploring: {db_label}")

        details = api.get_dataset_info(db_id)
        if not details:
            continue

        capabilities = {
            'id': db_id,
            'label': db_label,
            'measures': [],
            'geography_groups': [],
            'time_fields': [],
            'demographic_fields': [],
            'other_fields': []
        }

        for child in details.get('children', []):
            child_type = child.get('type', '')
            child_id = child.get('id', '')
            child_label = child.get('label', '')

            if child_type in ['COUNT', 'MEASURE']:
                capabilities['measures'].append({
                    'id': child_id,
                    'label': child_label,
                    'type': child_type
                })
                print(f"  📊 Measure: {child_label}")

            elif child_type == 'GROUP' and 'Geography' in child_label:
                capabilities['geography_groups'].append({
                    'id': child_id,
                    'label': child_label
                })
                print(f"  🗺️  Geography Group: {child_label}")

            elif child_type == 'FIELD':
                if 'DATE' in child_id.upper() or 'TIME' in child_id.upper():
                    capabilities['time_fields'].append({
                        'id': child_id,
                        'label': child_label
                    })
                    print(f"  📅 Time Field: {child_label}")
                elif any(demo in child_label.upper() for demo in ['AGE', 'GENDER', 'ETHNICITY', 'FAMILY', 'EMPLOYMENT']):
                    capabilities['demographic_fields'].append({
                        'id': child_id,
                        'label': child_label
                    })
                    print(f"  👥 Demographic: {child_label}")
                else:
                    capabilities['other_fields'].append({
                        'id': child_id,
                        'label': child_label
                    })
                    print(f"  📋 Other: {child_label}")

        all_capabilities[db_id] = capabilities

    # Save comprehensive schema
    with open('comprehensive_uc_schema.json', 'w') as f:
        json.dump(all_capabilities, f, indent=2)

    print(f"\n💾 Comprehensive schema saved to: comprehensive_uc_schema.json")
    return all_capabilities

def explore_geography_levels(db_id, geography_groups):
    """Explore specific geography levels for a database"""
    api_key = os.getenv('STAT_XPLORE_API_KEY')
    api = StatXploreAPI(api_key)

    geography_details = {}

    for geo_group in geography_groups[:2]:  # Limit to avoid rate limits
        group_id = geo_group['id']
        group_label = geo_group['label']

        print(f"\n🔍 Exploring geography group: {group_label}")

        group_details = api.get_dataset_info(group_id)
        if group_details and 'children' in group_details:
            geo_levels = []
            for child in group_details['children'][:10]:  # Limit items
                if child.get('type') == 'FIELD':
                    field_id = child.get('id')
                    field_label = child.get('label', '')

                    # Check if this might be a geographic level
                    if any(level in field_label.upper() for level in ['LSOA', 'WARD', 'DISTRICT', 'COUNTY', 'REGION', 'AUTHORITY']):
                        geo_levels.append({
                            'id': field_id,
                            'label': field_label,
                            'potential_level': field_label
                        })
                        print(f"    📍 Geographic level: {field_label}")

            geography_details[group_id] = {
                'group_label': group_label,
                'levels': geo_levels
            }

    return geography_details

def main():
    print("🚀 Comprehensive UC Data Schema Exploration")
    print("=" * 60)

    # Explore all datasets
    capabilities = explore_all_uc_datasets()

    if capabilities:
        print(f"\n📋 Summary:")
        for db_id, cap in capabilities.items():
            print(f"\n{cap['label']}:")
            print(f"  - Measures: {len(cap['measures'])}")
            print(f"  - Geography groups: {len(cap['geography_groups'])}")
            print(f"  - Time fields: {len(cap['time_fields'])}")
            print(f"  - Demographics: {len(cap['demographic_fields'])}")

            # Explore geography for databases with geography groups
            if cap['geography_groups']:
                geo_details = explore_geography_levels(db_id, cap['geography_groups'])
                cap['geography_details'] = geo_details

        # Save updated schema with geography details
        with open('comprehensive_uc_schema_with_geo.json', 'w') as f:
            json.dump(capabilities, f, indent=2)

        print(f"\n💾 Complete schema with geography saved to: comprehensive_uc_schema_with_geo.json")

if __name__ == '__main__':
    main()