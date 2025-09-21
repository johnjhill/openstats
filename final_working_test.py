#!/usr/bin/env python3
"""
Final test with correct regional and time values
"""

import requests
import json
import os
import time
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_time_values():
    """Get actual time values from the time field"""

    load_dotenv()
    api_key = os.getenv('STAT_XPLORE_API_KEY')
    headers = {
        'APIKey': api_key,
        'Content-Type': 'application/json'
    }

    base_url = "https://stat-xplore.dwp.gov.uk/webapi/rest/v1"
    time_field = "str:field:hb_new:F_HB_NEW_DATE:NEW_DATE_NAME"

    logger.info(f"📅 Getting time values...")

    try:
        url = f"{base_url}/schema/{time_field}"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            time_data = response.json()
            logger.info(f"✅ Got time field data")

            # Save the time data
            with open('hb_time_field.json', 'w') as f:
                json.dump(time_data, f, indent=2)

            # Look for recent time values
            if 'children' in time_data:
                logger.info(f"Found {len(time_data['children'])} time values")
                recent_values = []

                # Get the last few time values (most recent)
                for child in time_data['children'][-6:]:  # Last 6 months
                    time_id = child.get('id', '')
                    time_label = child.get('label', 'No label')
                    logger.info(f"  {time_id} - {time_label}")
                    if time_id:
                        recent_values.append(time_id)

                return recent_values[-3:]  # Return last 3 months

        else:
            logger.error(f"Failed to get time field: {response.status_code}")

    except Exception as e:
        logger.error(f"Error getting time values: {e}")

    return []

def final_working_test():
    """Test with both correct regional and time values"""

    load_dotenv()
    api_key = os.getenv('STAT_XPLORE_API_KEY')
    headers = {
        'APIKey': api_key,
        'Content-Type': 'application/json'
    }

    # Known working regional values from previous test
    regional_values = [
        'str:value:hb_new:V_F_HB_NEW:COA_CODE:V_C_MASTERGEOG11_REGION_TO_COUNTRY:E12000001',  # North East
        'str:value:hb_new:V_F_HB_NEW:COA_CODE:V_C_MASTERGEOG11_REGION_TO_COUNTRY:E12000002',  # North West
        'str:value:hb_new:V_F_HB_NEW:COA_CODE:V_C_MASTERGEOG11_REGION_TO_COUNTRY:E12000007'   # London
    ]

    # Get correct time values
    time_values = get_time_values()
    if not time_values:
        logger.error("Could not get time values")
        return False

    logger.info(f"Using time values: {time_values}")

    # Build the final correct payload
    payload = {
        "database": "str:database:hb_new",
        "measures": ["str:count:hb_new:V_F_HB_NEW"],
        "recodes": {
            "str:field:hb_new:V_F_HB_NEW:COA_CODE": {
                "map": [[value] for value in regional_values],
                "total": True
            },
            "str:field:hb_new:F_HB_NEW_DATE:NEW_DATE_NAME": {
                "map": [[value] for value in time_values],
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
        logger.info("🧪 FINAL TEST: Housing Benefit with correct regional and time values...")

        response = requests.post(url, json=payload, headers=headers)
        logger.info(f"Response status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            logger.info("🎉🎉🎉 BREAKTHROUGH CONFIRMED! Regional Housing Benefit data collected!")

            # Save the successful response
            with open('final_working_hb_regional.json', 'w') as f:
                json.dump(data, f, indent=2)

            # Save the working payload pattern
            with open('working_payload_pattern.json', 'w') as f:
                json.dump(payload, f, indent=2)

            logger.info("💾 Saved successful response and working pattern")

            # Show comprehensive stats
            if 'cubes' in data:
                for cube_id, cube_data in data['cubes'].items():
                    logger.info(f"📊 Cube: {cube_id}")
                    if 'values' in cube_data:
                        values = cube_data['values']
                        logger.info(f"  📈 Data points: {len(values)}")
                        logger.info(f"  📊 Total claims: {sum(v for v in values if v is not None):,.0f}")

                    if 'dimensions' in cube_data:
                        for i, dim in enumerate(cube_data['dimensions']):
                            items = dim.get('items', [])
                            logger.info(f"  🗂️ Dimension {i}: {len(items)} items")
                            # Show first few dimension items
                            for j, item in enumerate(items[:3]):
                                labels = item.get('labels', ['Unknown'])
                                logger.info(f"    {j+1}. {labels[0]}")

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
    success = final_working_test()
    if success:
        print("\n" + "="*80)
        print("🎉 REGIONAL DATA ACCESS BREAKTHROUGH CONFIRMED!")
        print("="*80)
        print("✅ Successfully accessed Housing Benefit data by region")
        print("✅ Confirmed working recodes pattern")
        print("✅ Pattern can now be applied to UC and PIP data")
        print("📁 Working pattern saved to working_payload_pattern.json")
        print("📁 Response data saved to final_working_hb_regional.json")
        print("="*80)
    else:
        print("❌ Still working on the breakthrough...")