#!/usr/bin/env python3
"""
Explore Housing Benefit geography groups to find specific field IDs
"""

import requests
import json
import os
import time
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def explore_geography_groups():
    """Explore Housing Benefit geography groups"""

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

    # Geography groups from the schema
    geography_groups = [
        "str:group:hb_new:X_Geography+%28admin-based%29",
        "str:group:hb_new:X_Geography+%28residence-based%29"
    ]

    for group_id in geography_groups:
        logger.info(f"📍 Exploring geography group: {group_id}")

        try:
            time.sleep(1)  # Rate limit
            url = f"{base_url}/schema/{group_id}"
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                group_data = response.json()
                logger.info(f"✅ Got data for {group_data.get('label', 'Unknown')}")

                # Save the group data
                filename = f"hb_geo_group_{group_id.split(':')[-1].replace('%28', '').replace('%29', '')}.json"
                with open(filename, 'w') as f:
                    json.dump(group_data, f, indent=2)
                logger.info(f"💾 Saved to {filename}")

                # Look for COA_CODE fields in the children
                if 'children' in group_data:
                    logger.info(f"Found {len(group_data['children'])} child items")
                    for child in group_data['children']:
                        child_id = child.get('id', '')
                        child_label = child.get('label', 'No label')
                        logger.info(f"  - {child_id} : {child_label}")

                        # If this looks like a COA_CODE field, explore it further
                        if 'COA_CODE' in child_id:
                            logger.info(f"🎯 Found COA_CODE field: {child_id}")
                            try:
                                time.sleep(1)
                                field_url = f"{base_url}/schema/{child_id}"
                                field_response = requests.get(field_url, headers=headers)

                                if field_response.status_code == 200:
                                    field_data = field_response.json()

                                    # Save field data
                                    field_filename = f"hb_coa_field.json"
                                    with open(field_filename, 'w') as f:
                                        json.dump(field_data, f, indent=2)
                                    logger.info(f"💾 Saved COA field to {field_filename}")

                                    # Look for first few geographic values
                                    if 'children' in field_data:
                                        logger.info(f"Found {len(field_data['children'])} geographic areas")
                                        logger.info("First 5 areas:")
                                        for i, area in enumerate(field_data['children'][:5]):
                                            area_id = area.get('id', 'No ID')
                                            area_label = area.get('label', 'No label')
                                            logger.info(f"    {i+1}. {area_id} - {area_label}")

                                        # Test with first area
                                        if field_data['children']:
                                            first_area = field_data['children'][0]
                                            test_area_id = first_area.get('id')
                                            if test_area_id:
                                                logger.info(f"🧪 Will test with area: {test_area_id}")
                                                return child_id, test_area_id

                            except Exception as e:
                                logger.error(f"Error exploring COA field {child_id}: {e}")

            else:
                logger.error(f"Failed to get group data: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Error exploring group {group_id}: {e}")

    return None, None

if __name__ == "__main__":
    field_id, test_area_id = explore_geography_groups()
    if field_id and test_area_id:
        print(f"✅ Found geographic field: {field_id}")
        print(f"✅ Test area: {test_area_id}")
    else:
        print("❌ No valid geographic field found")