#!/usr/bin/env python3
"""
Schema explorer for Stat-Xplore API
This script helps discover the correct field identifiers for Universal Credit data
"""

import os
import json
from dotenv import load_dotenv
from data_extractor import StatXploreAPI

load_dotenv()

def explore_schema():
    """Explore the Stat-Xplore schema to find correct field identifiers"""
    api_key = os.getenv('STAT_XPLORE_API_KEY')
    if not api_key:
        print("❌ API key not found in .env file")
        return

    api = StatXploreAPI(api_key)

    print("🔍 Exploring Stat-Xplore Schema...")
    print("=" * 50)

    # Get main schema
    schema = api.get_schema_info()
    if not schema:
        print("❌ Failed to get schema")
        return

    print(f"📊 Root schema contains {len(schema.get('children', []))} folders")

    # Look for Universal Credit folder (main one, not assessments)
    uc_folder = None
    print("\n📁 Available folders:")
    for child in schema.get('children', []):
        folder_name = child.get('label', '')
        print(f"  - {folder_name}")
        if folder_name == 'Universal Credit':  # Look for exact match
            uc_folder = child
            print(f"    🎯 Found main Universal Credit folder: {folder_name}")
            break

    if not uc_folder:
        print("\n❌ No Universal Credit folder found")
        return

    # Explore UC folder
    uc_folder_id = uc_folder['id']
    print(f"\n🔍 Exploring Universal Credit folder: {uc_folder_id}")

    uc_details = api.get_dataset_info(uc_folder_id)
    if not uc_details:
        print("❌ Failed to get UC folder details")
        return

    print(f"\nUC folder contains {len(uc_details.get('children', []))} items:")
    uc_databases = []

    for child in uc_details.get('children', []):
        item_type = child.get('type', 'UNKNOWN')
        item_label = child.get('label', 'No label')
        item_id = child.get('id', 'No ID')

        print(f"  - {item_label} ({item_type})")
        print(f"    ID: {item_id}")

        if item_type == 'DATABASE':
            uc_databases.append(child)

    print(f"\n🎯 Found {len(uc_databases)} Universal Credit databases:")
    for db in uc_databases:
        print(f"  - {db}")

    # Explore UC Households database in detail
    households_db = None
    for db in uc_databases:
        if 'Households' in db['label']:
            households_db = db
            break

    if households_db:
        db_id = households_db['id']
        print(f"\n🔍 Exploring {households_db['label']} ({db_id}) in detail...")

        details = api.get_dataset_info(db_id)
        if details:
            print("\n📋 Database structure:")

            if 'measures' in details:
                print("\n  📊 Measures:")
                for measure in details['measures'][:5]:  # Show first 5
                    print(f"    - {measure.get('label', 'No label')}")
                    print(f"      ID: {measure.get('id', 'No ID')}")

            if 'dimensions' in details:
                print("\n  📍 Dimensions:")
                for dim in details['dimensions'][:10]:  # Show first 10
                    print(f"    - {dim.get('label', 'No label')}")
                    print(f"      ID: {dim.get('id', 'No ID')}")

                    # Look for geography dimensions
                    if 'GEOGRAPHY' in dim.get('id', '').upper():
                        print("      🗺️  This looks like a geography dimension!")
                        # Get sub-levels
                        if 'dimensionitems' in dim:
                            print("      Geographic levels:")
                            for item in dim['dimensionitems'][:5]:
                                print(f"        * {item.get('label', 'No label')}")
                                print(f"          ID: {item.get('id', 'No ID')}")

                    # Look for time dimensions
                    if 'DATE' in dim.get('id', '').upper() or 'TIME' in dim.get('id', '').upper():
                        print("      📅 This looks like a time dimension!")

            # Save full schema to file for reference
            with open('schema_details.json', 'w') as f:
                json.dump(details, f, indent=2)
            print(f"\n💾 Full schema saved to: schema_details.json")

    print("\n✅ Schema exploration complete!")

if __name__ == '__main__':
    explore_schema()