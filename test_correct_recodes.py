#!/usr/bin/env python3
"""
Test corrected recodes pattern using valuesets instead of individual values
"""

import requests
import json
import os
import time
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_corrected_housing_benefit_query():
    """Test Housing Benefit query with correct valueset structure"""

    load_dotenv()
    api_key = os.getenv('STAT_XPLORE_API_KEY')
    if not api_key:
        logger.error("API key not found")
        return

    headers = {
        'APIKey': api_key,
        'Content-Type': 'application/json'
    }

    # Use the corrected payload with valueset IDs instead of individual value IDs
    payload = {
        "database": "str:database:hb_new",
        "measures": ["str:count:hb_new:V_F_HB_NEW"],
        "recodes": {
            "str:field:hb_new:V_F_HB_NEW:COA_CODE": {
                # Use the Regional level valueset from our schema exploration
                "map": [["str:valueset:hb_new:V_F_HB_NEW:COA_CODE:V_C_MASTERGEOG11_REGION_TO_COUNTRY"]],
                "total": True
            },
            "str:field:hb_new:F_HB_NEW_DATE:NEW_DATE_NAME": {
                "map": [
                    ["str:value:hb_new:F_HB_NEW_DATE:NEW_DATE_NAME:C_DATE:202401"],
                    ["str:value:hb_new:F_HB_NEW_DATE:NEW_DATE_NAME:C_DATE:202402"],
                    ["str:value:hb_new:F_HB_NEW_DATE:NEW_DATE_NAME:C_DATE:202403"]
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
        logger.info("🧪 Testing corrected Housing Benefit query with valuesets...")
        response = requests.post(url, json=payload, headers=headers)

        logger.info(f"Response status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            logger.info("✅ SUCCESS! Housing Benefit regional query worked with valuesets!")
            logger.info(f"Response keys: {list(data.keys())}")

            if 'cubes' in data:
                logger.info(f"Cubes found: {list(data['cubes'].keys())}")
                for cube_id, cube_data in data['cubes'].items():
                    if 'dimensions' in cube_data:
                        logger.info(f"Cube {cube_id} dimensions: {len(cube_data['dimensions'])}")
                        for i, dim in enumerate(cube_data['dimensions']):
                            logger.info(f"  Dimension {i}: {len(dim.get('items', []))} items")
                            # Show first few items
                            for j, item in enumerate(dim.get('items', [])[:3]):
                                item_label = item.get('labels', ['Unknown'])[0]
                                logger.info(f"    {j+1}. {item_label}")
                    if 'values' in cube_data:
                        logger.info(f"Cube {cube_id} values: {len(cube_data['values'])} total")
                        logger.info(f"First few values: {cube_data['values'][:10]}")

            # Save the successful response
            with open('successful_hb_regional_response.json', 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("💾 Saved successful response to successful_hb_regional_response.json")

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
    success = test_corrected_housing_benefit_query()
    if success:
        print("🎉 Regional data access pattern confirmed!")
    else:
        print("❌ Still having issues with regional data access")