#!/usr/bin/env python3
"""
Fixed regional data converter that handles the correct nested array structure
"""

import pandas as pd
import json
from typing import Dict, List

def convert_regional_response_to_dataframe(response: Dict, collection_name: str) -> pd.DataFrame:
    """Convert regional API response to DataFrame with correct nested structure handling"""
    records = []

    if 'cubes' not in response:
        return pd.DataFrame()

    # Get field information from the response
    fields_info = {}
    if 'fields' in response:
        for field in response['fields']:
            field_uri = field.get('uri', '')
            field_info = {
                'label': field.get('label', ''),
                'items': field.get('items', [])
            }
            fields_info[field_uri] = field_info

    for cube_id, cube_data in response['cubes'].items():
        if 'values' not in cube_data:
            continue

        values = cube_data['values']

        # Extract geographic and time information from fields
        geographic_field = None
        time_field = None

        for field_uri, field_info in fields_info.items():
            if 'COA_CODE' in field_uri or 'geography' in field_info['label'].lower():
                geographic_field = field_info
            elif 'DATE' in field_uri or 'time' in field_info['label'].lower() or 'month' in field_info['label'].lower():
                time_field = field_info

        if not geographic_field or not time_field:
            continue

        # Extract geographic areas (excluding Total)
        geo_items = [item for item in geographic_field['items'] if item.get('type') != 'Total']
        time_items = time_field['items']

        # Process the nested values structure
        for i, geo_item in enumerate(geo_items):
            geo_labels = geo_item.get('labels', [])
            geo_uris = geo_item.get('uris', [])
            geo_name = geo_labels[0] if geo_labels else f'Area_{i}'
            geo_code = geo_uris[0].split(':')[-1] if geo_uris else f'GEO_{i}'

            # Get the values for this geographic area
            if i < len(values):
                area_values = values[i]

                # Handle both nested arrays and flat arrays
                if isinstance(area_values, list):
                    value_list = area_values
                else:
                    value_list = [area_values]

                for j, time_item in enumerate(time_items):
                    time_labels = time_item.get('labels', [])
                    time_uris = time_item.get('uris', [])
                    time_period = time_labels[0] if time_labels else f'Period_{j}'

                    # Get the value for this time period
                    if j < len(value_list):
                        value = value_list[j]

                        if value is not None and value > 0:
                            records.append({
                                'Geography_Code': geo_code,
                                'Geography_Name': geo_name,
                                'Time_Period': time_period,
                                'Value': value,
                                'Dataset': collection_name,
                                'Measure': cube_id,
                                'Benefit_Type': get_benefit_type(collection_name)
                            })

    df = pd.DataFrame(records)

    if not df.empty:
        # Process dates
        try:
            # Handle different date formats
            if df['Time_Period'].str.contains(r'\d{6}').any():
                # Format like "202410 (Oct-24)"
                df['Date'] = pd.to_datetime(df['Time_Period'].str.extract(r'(\d{6})')[0], format='%Y%m', errors='coerce')
            else:
                df['Date'] = pd.to_datetime(df['Time_Period'], format='%Y %b', errors='coerce')
        except:
            df['Date'] = pd.to_datetime(df['Time_Period'], errors='coerce')

        # Add helpful columns
        if 'Date' in df.columns and df['Date'].notna().any():
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Quarter'] = df['Date'].dt.quarter

        print(f"✅ {collection_name}: {len(df)} records processed")
        if len(df) > 0:
            print(f"   📍 Geographic areas: {df['Geography_Code'].nunique()}")
            print(f"   📅 Time periods: {df['Time_Period'].nunique()}")
            print(f"   📊 Total cases: {df['Value'].sum():,.0f}")

    return df

def get_benefit_type(collection_name: str) -> str:
    """Determine benefit type from collection name"""
    if 'uc' in collection_name.lower() or 'universal' in collection_name.lower():
        return 'Universal Credit'
    elif 'pip' in collection_name.lower():
        return 'Personal Independence Payment'
    elif 'hb' in collection_name.lower() or 'housing' in collection_name.lower():
        return 'Housing Benefit'
    else:
        return 'Unknown'

# Test with the breakthrough response
def test_converter():
    """Test the converter with the breakthrough response"""

    with open('breakthrough_hb_regional.json', 'r') as f:
        response = json.load(f)

    print("Testing fixed converter with breakthrough response...")
    df = convert_regional_response_to_dataframe(response, 'housing_benefit_test')

    if not df.empty:
        print(f"\n🎉 SUCCESS! Converted {len(df)} records")
        print(f"Geographic areas: {df['Geography_Name'].unique()}")
        print(f"Time periods: {df['Time_Period'].unique()}")
        print(f"Total HB claimants: {df['Value'].sum():,.0f}")

        # Save test result
        df.to_csv('breakthrough_hb_converted.csv', index=False)
        print("💾 Saved to breakthrough_hb_converted.csv")
        return True
    else:
        print("❌ Conversion failed")
        return False

if __name__ == "__main__":
    test_converter()