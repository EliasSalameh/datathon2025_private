import os
import json
from collections import defaultdict
from pathlib import Path


def profile_is_consistent(profile):
    # Checks if graduation years are logically consistent.
    birth_year = int(profile["birth_date"].split("-")[0])
    secondary_year = secondary_year = profile["secondary_school"]["graduation_year"]
            
    if secondary_year - birth_year < 17:
        return False
    if secondary_year - birth_year > 20:
        return False
    
    # Get earliest higher education year
    min_higher_ed_year = None
    for education in profile["higher_education"]:
        year = education["graduation_year"]
        if min_higher_ed_year is None or year < min_higher_ed_year:
            min_higher_ed_year = year

    # Check higher education consistency
    if min_higher_ed_year:
        if min_higher_ed_year < secondary_year:
            return False
        if min_higher_ed_year - birth_year < 18:
            return False

    # Checks if real_estate_value is consistent with the reported values below
    real_estate_value = profile["aum"]["real_estate_value"]
    property_value_sum = 0
    for property in profile["real_estate_details"]:
        property_value_sum += property["property value"]

    if real_estate_value != property_value_sum:
        return False

    return True

def test_profile_consistency(CLIENT_DATA_PATH = "data/clients"):
    clients_dir = Path(CLIENT_DATA_PATH) 

    cnt = 0
    for client_dir in clients_dir.iterdir():
        if not client_dir.is_dir():
            continue
            
        profile_path = client_dir / "client_profile.json"
        label_path = client_dir / "label.json"
        
        try:
            profile = json.load(profile_path.open("r", encoding="utf-8"))
            label = json.load(label_path.open("r", encoding="utf-8")).get("label")
        except Exception as e:
            print(f"Error reading files for {client_dir}: {e}")
            continue

        is_consistent = profile_is_consistent(profile)
        if not is_consistent:
            cnt += 1
            # Uncomment to verify if inconsistent profiles are rejected
            assert label == "Reject", client_dir
    
    print(f"{cnt} clients with inconsistency detected")

if __name__ == "__main__":
    test_profile_consistency() 