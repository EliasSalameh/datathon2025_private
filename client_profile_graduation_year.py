import os
import json
from collections import defaultdict
from pathlib import Path

# Path to client data - final version with all validations
CLIENT_DATA_PATH = "data/clients"

def profile_is_consistent(profile):
    """
    Checks if graduation years, employment dates, and property values are logically consistent.
    Returns True if the profile is consistent, False otherwise.
    """
    
    # Extract birth year
    birth_year = int(profile["birth_date"].split("-")[0])
    
    # Get secondary school graduation year
    secondary_year = profile["secondary_school"]["graduation_year"]
    if birth_year:
        if secondary_year - birth_year < 17:
            return False
        if secondary_year - birth_year > 20:
            return False
    
    # Get earliest higher education year
    min_higher_ed_year = None
    max_higher_ed_year = None
    for education in profile["higher_education"]:
        year = education["graduation_year"]
        if min_higher_ed_year is None or year < min_higher_ed_year:
            min_higher_ed_year = year
        if max_higher_ed_year is None or year > max_higher_ed_year:
            max_higher_ed_year = year
    
    # Check higher education consistency
    if min_higher_ed_year:
        if secondary_year and min_higher_ed_year < secondary_year:
            return False
        if birth_year and min_higher_ed_year - birth_year < 18:
            return False
    
    # Check employment history consistency
    min_employment_start = None
    employment_intervals = []
    for employment in profile["employment_history"]:
        # Extract employment years
        start_year = employment.get("start_year")
        end_year = employment.get("end_year")
        
        if end_year is None:
            employment_intervals.append([start_year, 2030])
        else:
            employment_intervals.append([start_year, end_year])

        if min_employment_start is None or start_year < min_employment_start:
            min_employment_start = start_year

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

    # Check whether jobs overlap
    employment_intervals.sort()
    for i in range(len(employment_intervals)-1):
        if employment_intervals[i][1] > employment_intervals[i+1][0]:
            return False

    # Checks if real_estate_value is consistent with the reported values
    if "aum" in profile and "real_estate_value" in profile["aum"] and "real_estate_details" in profile:
        real_estate_value = profile["aum"]["real_estate_value"]
        property_value_sum = 0
        for property in profile["real_estate_details"]:
            if "property value" in property:
                property_value_sum += property["property value"]

        if real_estate_value != property_value_sum:
            return False
        
    # Remove non-standard values
    if profile["investment_risk_profile"] in ["Aggressive", "Balanced", "Conservative"]:
        return False
    if profile["investment_horizon"] not in ["Long-Term", "Medium", "Short"]:
        return False
    if profile["type_of_mandate"] not in ["Advisory", "Discretionary"]:
        return False

    # If no issues were found, the profile is consistent
    return True

def test_profile_consistency():
    clients_dir = Path(CLIENT_DATA_PATH) 

    cnt = 0
    for client_dir in clients_dir.iterdir():
        if not client_dir.is_dir():
            continue
            
        client_id = os.path.basename(client_dir)
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
    
    print(f"{cnt} clients with inconsistencies (education, employment, or property values) detected")

if __name__ == "__main__":
    test_profile_consistency()