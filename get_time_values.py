#!/usr/bin/env python3
"""
Get individual time values from Housing Benefit time valueset
"""

import requests
import json
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_individual_time_values():
    """Get individual time values from the time valueset"""

    load_dotenv()
    api_key = os.getenv('STAT_XPLORE_API_KEY')
    headers = {
        'APIKey': api_key,
        'Content-Type': 'application/json'
    }

    base_url = "https://stat-xplore.dwp.gov.uk/webapi/rest/v1"

    # The time valueset found in previous exploration
    time_valueset = "str:valueset:hb_new:F_HB_NEW_DATE:NEW_DATE_NAME:C_HB_NEW_DATE"

    logger.info(f"📅 Getting individual time values from valueset...")

    try:
        url = f"{base_url}/schema/{time_valueset}"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            time_data = response.json()
            logger.info(f"✅ Got time valueset data")

            # Save the time valueset data
            with open('hb_time_valueset.json', 'w') as f:
                json.dump(time_data, f, indent=2)
            logger.info("💾 Saved to hb_time_valueset.json")

            # Look for individual time values
            if 'children' in time_data:
                logger.info(f"Found {len(time_data['children'])} individual time values")
                time_values = []

                # Show all time values and get recent ones
                for i, child in enumerate(time_data['children']):
                    time_id = child.get('id', '')
                    time_label = child.get('label', 'No label')

                    if i < 10:  # Show first 10
                        logger.info(f"  {i+1}. {time_id} - {time_label}")

                    if time_id and ('2024' in time_id or '2023' in time_id):
                        time_values.append(time_id)

                # Return the most recent few values
                logger.info(f"Found {len(time_values)} recent time values")
                return time_values[-3:] if time_values else []

        else:
            logger.error(f"Failed to get time valueset: {response.status_code} - {response.text}")

    except Exception as e:
        logger.error(f"Error getting time values: {e}")

    return []

def test_with_complete_values():
    """Test with both correct regional and individual time values"""

    # Known working regional values
    regional_values = [
        'str:value:hb_new:V_F_HB_NEW:COA_CODE:V_C_MASTERGEOG11_REGION_TO_COUNTRY:E12000001',  # North East
        'str:value:hb_new:V_F_HB_NEW:COA_CODE:V_C_MASTERGEOG11_REGION_TO_COUNTRY:E12000007'   # London
    ]

    # Get individual time values
    time_values = get_individual_time_values()
    if not time_values:
        logger.error("Could not get individual time values")
        return False

    load_dotenv()
    api_key = os.getenv('STAT_XPLORE_API_KEY')
    headers = {
        'APIKey': api_key,
        'Content-Type': 'application/json'
    }

    # Build the complete correct payload
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
        logger.info("🎯 TESTING WITH COMPLETE INDIVIDUAL VALUES...")
        logger.info(f"Regional values: {regional_values}")
        logger.info(f"Time values: {time_values}")

        response = requests.post(url, json=payload, headers=headers)
        logger.info(f"Response status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            logger.info("🎉🎉🎉 BREAKTHROUGH ACHIEVED! Regional data access working!")

            # Save everything
            with open('breakthrough_hb_regional.json', 'w') as f:
                json.dump(data, f, indent=2)

            with open('breakthrough_payload.json', 'w') as f:
                json.dump(payload, f, indent=2)

            logger.info("💾 Saved breakthrough response and payload")

            # Show the data structure
            if 'cubes' in data:
                for cube_id, cube_data in data['cubes'].items():
                    logger.info(f"📊 Successfully collected regional Housing Benefit data!")
                    if 'values' in cube_data:
                        values = [v for v in cube_data['values'] if v is not None]
                        logger.info(f"  📈 {len(values)} data points collected")
                        logger.info(f"  📊 Total HB claimants: {sum(values):,.0f}")

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
    success = test_with_complete_values()
    if success:
        print("\n" + "="*80)
        print("🎉🎉🎉 REGIONAL DATA BREAKTHROUGH ACHIEVED!")
        print("="*80)
        print("✅ Housing Benefit regional data successfully collected")
        print("✅ Working recodes pattern confirmed")
        print("✅ Ready to apply to UC and PIP datasets")
        print("📁 Files saved: breakthrough_hb_regional.json, breakthrough_payload.json")
        print("="*80)
    else:
        print("❌ Still working on the breakthrough...")