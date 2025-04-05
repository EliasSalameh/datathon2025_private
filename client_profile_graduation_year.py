import os
import json
from collections import defaultdict
from pathlib import Path


def profile_is_consistent(profile):
<<<<<<< HEAD
    """
    Checks if graduation years and employment dates are logically consistent.
    Returns True if the profile is consistent, False otherwise.
    """
    
    # Extract birth year
    birth_year = None
    if "birth_date" in profile and profile["birth_date"]:
        try:
            birth_year = int(profile["birth_date"].split("-")[0])
        except:
            pass
    
    # Get secondary school graduation year
    secondary_year = None
    if "secondary_school" in profile and isinstance(profile["secondary_school"], dict):
        if "graduation_year" in profile["secondary_school"]:
            secondary_year = profile["secondary_school"]["graduation_year"]
=======
    # Checks if graduation years are logically consistent.
    birth_year = int(profile["birth_date"].split("-")[0])
    secondary_year = secondary_year = profile["secondary_school"]["graduation_year"]
>>>>>>> 9ec5a3df53a04dc1a655fb3b457bcb9eb2005bee
            
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
<<<<<<< HEAD
    
    # Check employment history consistency
    if "employment_history" in profile and isinstance(profile["employment_history"], list):
        for employment in profile["employment_history"]:
            if isinstance(employment, dict):
                # Extract employment years
                start_year = employment.get("start_year")
                end_year = employment.get("end_year")
                
                # Check start year is valid (not before birth + 16 years)
                if start_year and birth_year:
                    # Person should be at least 16 years old to start working
                    if start_year - birth_year < 16:
                        return False
                
                # Check end year is greater than start year
                if start_year and end_year:
                    # End year should be after start year
                    if end_year < start_year:
                        return False
                
                # Check that employment doesn't start before graduation
                if start_year and secondary_year:
                    # Typically employment starts after secondary education
                    # Only flag if start year is more than 2 years before graduation
                    if secondary_year - start_year > 2:
                        return False
    
    # If no issues were found, the profile is consistent
=======

    # Checks if real_estate_value is consistent with the reported values below
    real_estate_value = profile["aum"]["real_estate_value"]
    property_value_sum = 0
    for property in profile["real_estate_details"]:
        property_value_sum += property["property value"]

    if real_estate_value != property_value_sum:
        return False

>>>>>>> 9ec5a3df53a04dc1a655fb3b457bcb9eb2005bee
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
    
<<<<<<< HEAD
    print(f"{cnt} clients with inconsistent dates (education or employment) detected")
=======
    print(f"{cnt} clients with inconsistency detected")
>>>>>>> 9ec5a3df53a04dc1a655fb3b457bcb9eb2005bee

if __name__ == "__main__":
    test_profile_consistency() 