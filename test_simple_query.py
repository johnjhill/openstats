#!/usr/bin/env python3
"""
Simple test query to verify UC Households API works
"""

import os
import json
from dotenv import load_dotenv
from data_extractor import StatXploreAPI

load_dotenv()

def test_simple_query():
    """Test a basic query to UC Households database"""
    api_key = os.getenv('STAT_XPLORE_API_KEY')
    if not api_key:
        print("❌ API key not found")
        return

    api = StatXploreAPI(api_key)

    # Simple query structure
    query = {
        "database": "str:database:UC_Households",
        "measures": ["str:count:UC_Households:V_F_UC_HOUSEHOLDS"],
        "dimensions": [
            [{
                "dimension": "str:field:UC_Households:F_UC_HH_DATE:DATE_NAME",
                "values": ["*"]
            }]
        ]
    }

    print("🧪 Testing simple query...")
    print("Query:", json.dumps(query, indent=2))

    result = api._make_request('POST', f"{api.base_url}/table", json=query)

    if result:
        print("✅ Query successful!")
        print("Result structure:")
        for key in result.keys():
            print(f"  - {key}: {type(result[key])}")

        if 'cubes' in result:
            cube = result['cubes'][0]
            print(f"\nCube structure:")
            for key in cube.keys():
                print(f"  - {key}: {type(cube[key])}")

            if 'dimen' in cube:
                print(f"\nDimensions: {len(cube['dimen'])}")
                for i, dim in enumerate(cube['dimen']):
                    print(f"  Dimension {i}: {len(dim)} items")
                    if dim:
                        print(f"    Sample: {dim[0]}")

        # Save result for inspection
        with open('test_query_result.json', 'w') as f:
            json.dump(result, f, indent=2)
        print("\n💾 Full result saved to: test_query_result.json")

    else:
        print("❌ Query failed")

if __name__ == '__main__':
    test_simple_query()