#!/usr/bin/env python3
"""
Test the exact recodes pattern from the Housing Benefit JSON examples
"""

import requests
import json
import os
import time
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_exact_housing_benefit_query():
    """Test the exact Housing Benefit query from the JSON examples"""

    load_dotenv()
    api_key = os.getenv('STAT_XPLORE_API_KEY')
    if not api_key:
        logger.error("API key not found")
        return

    headers = {
        'APIKey': api_key,
        'Content-Type': 'application/json'
    }

    # Use the exact payload from the Housing Benefit JSON example
    payload = {
        "database": "str:database:hb_new",
        "measures": ["str:count:hb_new:V_F_HB_NEW"],
        "recodes": {
            "str:field:hb_new:V_F_HB_NEW:COA_CODE": {
                "map": [
                    ["str:value:hb_new:V_F_HB_NEW:COA_CODE:V_C_MASTERGEOG11_GB:K03000001"],
                    ["str:value:hb_new:V_F_HB_NEW:COA_CODE:V_C_MASTERGEOG11_GB:K04000001"],
                    ["str:value:hb_new:V_F_HB_NEW:COA_CODE:V_C_MASTERGEOG11_GB:K02000001"]
                ],
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
        logger.info("🧪 Testing exact Housing Benefit query from JSON examples...")
        response = requests.post(url, json=payload, headers=headers)

        logger.info(f"Response status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            logger.info("✅ SUCCESS! Housing Benefit regional query worked!")
            logger.info(f"Response keys: {list(data.keys())}")

            if 'cubes' in data:
                logger.info(f"Cubes found: {list(data['cubes'].keys())}")
                for cube_id, cube_data in data['cubes'].items():
                    if 'dimensions' in cube_data:
                        logger.info(f"Cube {cube_id} dimensions: {len(cube_data['dimensions'])}")
                        for i, dim in enumerate(cube_data['dimensions']):
                            logger.info(f"  Dimension {i}: {len(dim.get('items', []))} items")
                    if 'values' in cube_data:
                        logger.info(f"Cube {cube_id} values: {len(cube_data['values'])} total")

            # Save the successful response
            with open('successful_hb_response.json', 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("💾 Saved successful response to successful_hb_response.json")

        else:
            try:
                error_data = response.json()
                logger.error(f"API error {response.status_code}: {error_data.get('message', response.text)}")
            except:
                logger.error(f"API error {response.status_code}: {response.text}")

    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    test_exact_housing_benefit_query()