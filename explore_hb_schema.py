#!/usr/bin/env python3
"""
Explore Housing Benefit schema to find valid geographic values
"""

import requests
import json
import os
import time
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def explore_hb_schema():
    """Explore Housing Benefit database schema"""

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

    # 1. Explore the Housing Benefit database
    logger.info("📋 Exploring Housing Benefit database...")

    try:
        url = f"{base_url}/schema/str:database:hb_new"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            hb_schema = response.json()
            logger.info("✅ Got Housing Benefit schema")

            # Save the schema for analysis
            with open('hb_schema.json', 'w') as f:
                json.dump(hb_schema, f, indent=2)
            logger.info("💾 Saved HB schema to hb_schema.json")

            # Look for geographic fields
            logger.info("🔍 Searching for geographic fields...")

            def find_fields(obj, path="", depth=0):
                if depth > 10:  # Prevent infinite recursion
                    return

                if isinstance(obj, dict):
                    # Check if this is a field with COA_CODE
                    if 'id' in obj and 'COA_CODE' in obj.get('id', ''):
                        logger.info(f"📍 Found geographic field: {obj['id']}")
                        logger.info(f"   Label: {obj.get('label', 'No label')}")

                        # Try to get values for this field
                        time.sleep(1)  # Rate limit
                        try:
                            field_url = f"{base_url}/schema/{obj['id']}"
                            field_response = requests.get(field_url, headers=headers)

                            if field_response.status_code == 200:
                                field_data = field_response.json()
                                if 'children' in field_data:
                                    logger.info(f"   Found {len(field_data['children'])} geographic values")
                                    # Show first few values
                                    for i, child in enumerate(field_data['children'][:5]):
                                        logger.info(f"     {i+1}. {child.get('id', 'No ID')} - {child.get('label', 'No label')}")

                                    # Save field data
                                    field_filename = f"hb_field_{obj['id'].split(':')[-1]}.json"
                                    with open(field_filename, 'w') as f:
                                        json.dump(field_data, f, indent=2)
                                    logger.info(f"     💾 Saved to {field_filename}")

                        except Exception as e:
                            logger.error(f"Error exploring field {obj['id']}: {e}")

                    # Recurse into children
                    for key, value in obj.items():
                        if key in ['children', 'items'] and isinstance(value, list):
                            for item in value:
                                find_fields(item, f"{path}.{key}", depth + 1)
                        elif isinstance(value, (dict, list)):
                            find_fields(value, f"{path}.{key}", depth + 1)

                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        find_fields(item, f"{path}[{i}]", depth + 1)

            find_fields(hb_schema)

        else:
            logger.error(f"Failed to get HB schema: {response.status_code} - {response.text}")

    except Exception as e:
        logger.error(f"Error exploring HB schema: {e}")

if __name__ == "__main__":
    explore_hb_schema()