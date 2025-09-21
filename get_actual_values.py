#!/usr/bin/env python3
"""
Get actual individual values from Housing Benefit valuesets
"""

import requests
import json
import os
import time
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_regional_values():
    """Get actual regional values from the Regional valueset"""

    load_dotenv()
    api_key = os.getenv('STAT_XPLORE_API_KEY')
    if not api_key:
        logger.error("API key not found")
        return

    headers = {
        'APIKey': api_key,
        'Content-Type': 'application/json'
    }

    base_url = "https://stat-xplore.dwp.gov.uk/webapi/rest/v1"

    # Get actual values from the Regional valueset
    regional_valueset = "str:valueset:hb_new:V_F_HB_NEW:COA_CODE:V_C_MASTERGEOG11_REGION_TO_COUNTRY"

    logger.info(f"📍 Getting values from Regional valueset...")

    try:
        url = f"{base_url}/schema/{regional_valueset}"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            valueset_data = response.json()
            logger.info(f"✅ Got Regional valueset data")

            # Save the valueset data
            with open('hb_regional_valueset.json', 'w') as f:
                json.dump(valueset_data, f, indent=2)
            logger.info("💾 Saved to hb_regional_valueset.json")

            # Look for actual values
            if 'children' in valueset_data:
                logger.info(f"Found {len(valueset_data['children'])} regional values")
                actual_values = []

                for i, child in enumerate(valueset_data['children'][:10]):  # First 10
                    value_id = child.get('id', '')
                    value_label = child.get('label', 'No label')
                    logger.info(f"  {i+1}. {value_id} - {value_label}")

                    if value_id:
                        actual_values.append(value_id)

                return actual_values[:3]  # Return first 3 for testing

        else:
            logger.error(f"Failed to get valueset: {response.status_code} - {response.text}")

    except Exception as e:
        logger.error(f"Error getting regional values: {e}")

    return []

def test_with_actual_values(regional_values):
    """Test Housing Benefit query with actual individual values"""

    load_dotenv()
    api_key = os.getenv('STAT_XPLORE_API_KEY')
    headers = {
        'APIKey': api_key,
        'Content-Type': 'application/json'
    }

    if not regional_values:
        logger.error("No regional values to test with")
        return False

    # Use actual individual values in the recodes
    payload = {
        "database": "str:database:hb_new",
        "measures": ["str:count:hb_new:V_F_HB_NEW"],
        "recodes": {
            "str:field:hb_new:V_F_HB_NEW:COA_CODE": {
                "map": [[value] for value in regional_values],  # Each value in its own map array
                "total": True
            },
            "str:field:hb_new:F_HB_NEW_DATE:NEW_DATE_NAME": {
                "map": [
                    ["str:value:hb_new:F_HB_NEW_DATE:NEW_DATE_NAME:C_DATE:202401"],
                    ["str:value:hb_new:F_HB_NEW_DATE:NEW_DATE_NAME:C_DATE:202402"]
                ],
                "total": False
            }
        },
        "dimensions": [
            ["str:field:hb_new:V_F_HB_NEW:COA_CODE"],
            ["str:field:hb_new:F_HB_NEW_DATE:NEW_DATE_NAME"]
        ]
    }

    url = "https://stat-xplore.dwp.gov.uk/webapi/rest/v1/table"

    try:
        logger.info("🧪 Testing Housing Benefit query with actual individual values...")
        logger.info(f"Using values: {regional_values}")

        response = requests.post(url, json=payload, headers=headers)
        logger.info(f"Response status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            logger.info("🎉 SUCCESS! Regional Housing Benefit data accessed!")

            # Save the successful response
            with open('working_hb_regional_response.json', 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("💾 Saved successful response")

            # Show basic stats
            if 'cubes' in data:
                for cube_id, cube_data in data['cubes'].items():
                    if 'values' in cube_data:
                        logger.info(f"Got {len(cube_data['values'])} data points")
                    if 'dimensions' in cube_data:
                        for i, dim in enumerate(cube_data['dimensions']):
                            logger.info(f"Dimension {i}: {len(dim.get('items', []))} items")

            return True

        else:
            try:
                error_data = response.json()
                logger.error(f"API error {response.status_code}: {error_data.get('message', response.text)}")
            except:
                logger.error(f"API error {response.status_code}: {response.text}")
            return False

    except Exception as e:
        logger.error(f"Error: {e}")
        return False

if __name__ == "__main__":
    regional_values = get_regional_values()
    if regional_values:
        success = test_with_actual_values(regional_values)
        if success:
            print("🎉 BREAKTHROUGH: Regional data access confirmed!")
            print(f"Working pattern uses actual value IDs: {regional_values}")
        else:
            print("❌ Still not working")
    else:
        print("❌ Could not get regional values")