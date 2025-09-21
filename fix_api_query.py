#!/usr/bin/env python3
"""
Fix the Stat-Xplore API query format based on official documentation
"""

import os
import json
from dotenv import load_dotenv
from data_extractor import StatXploreAPI

load_dotenv()

def explore_geography_field():
    """Explore geography field to get correct value IDs"""
    api_key = os.getenv('STAT_XPLORE_API_KEY')
    if not api_key:
        print("❌ API key not found")
        return

    api = StatXploreAPI(api_key)

    # Explore the geography group to find field values
    geography_group_id = "str:group:UC_Households:X_Geography+%28residence-based%29"

    print(f"🔍 Exploring geography group: {geography_group_id}")

    geography_details = api.get_dataset_info(geography_group_id)

    if geography_details:
        print("📍 Geography group structure:")
        print(json.dumps(geography_details, indent=2)[:2000] + "...")

        # Look for children that might be geographic levels
        if 'children' in geography_details:
            print(f"\n📊 Found {len(geography_details['children'])} geography options:")
            for child in geography_details['children'][:10]:  # Show first 10
                print(f"  - {child.get('label', 'No label')}")
                print(f"    ID: {child.get('id', 'No ID')}")
                print(f"    Type: {child.get('type', 'Unknown')}")

                # If this is a FIELD, explore its values
                if child.get('type') == 'FIELD':
                    field_id = child.get('id')
                    print(f"\n    🔍 Exploring field: {field_id}")

                    field_details = api.get_dataset_info(field_id)
                    if field_details and 'children' in field_details:
                        print(f"      Found {len(field_details['children'])} values:")
                        for value in field_details['children'][:5]:  # Show first 5
                            print(f"        * {value.get('label', 'No label')}")
                            print(f"          ID: {value.get('id', 'No ID')}")
                    break

        # Save full details
        with open('geography_details.json', 'w') as f:
            json.dump(geography_details, f, indent=2)
        print(f"\n💾 Full geography details saved to: geography_details.json")

def test_corrected_query():
    """Test a corrected query format"""
    api_key = os.getenv('STAT_XPLORE_API_KEY')
    if not api_key:
        print("❌ API key not found")
        return

    api = StatXploreAPI(api_key)

    # Try a very simple query with just time dimension
    simple_query = {
        "database": "str:database:UC_Households",
        "measures": ["str:count:UC_Households:V_F_UC_HOUSEHOLDS"],
        "dimensions": [
            ["str:field:UC_Households:F_UC_HH_DATE:DATE_NAME"]
        ]
    }

    print("🧪 Testing corrected simple query...")
    print("Query:", json.dumps(simple_query, indent=2))

    result = api._make_request('POST', f"{api.base_url}/table", json=simple_query)

    if result:
        print("✅ Simple query successful!")
        print("Result keys:", list(result.keys()))

        # Save result first
        with open('corrected_query_result.json', 'w') as f:
            json.dump(result, f, indent=2)
        print("💾 Result saved to: corrected_query_result.json")

        if 'cubes' in result and result['cubes']:
            cube = result['cubes'][0]
            print(f"Cube keys: {list(cube.keys())}")
            print(f"Data shape: {len(cube.get('values', []))} rows")

            if 'dimen' in cube and cube['dimen']:
                time_dimension = cube['dimen'][0]
                print(f"Time periods available: {len(time_dimension)}")
                if time_dimension:
                    print(f"First period: {time_dimension[0].get('label', 'No label')}")
                    print(f"Last period: {time_dimension[-1].get('label', 'No label')}")

        return True
    else:
        print("❌ Simple query still failed")
        return False

def main():
    print("🔧 Fixing Stat-Xplore API Query Format")
    print("=" * 50)

    # First test the corrected simple query
    if test_corrected_query():
        print("\n" + "=" * 50)
        explore_geography_field()
    else:
        print("❌ Basic query structure still has issues")

if __name__ == '__main__':
    main()