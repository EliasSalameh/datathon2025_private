import json
from datetime import datetime
from pathlib import Path

def client_profile_and_passport_are_consistent(client_profile, passport):
    # Get passport data
    passport_first_name = passport["first_name"]
    passport_middle_name = passport["middle_name"]
    passport_last_name = passport["last_name"]
    passport_full_name = passport_first_name + " " + passport_middle_name + " " + passport_last_name
    passport_full_name_no_space = passport_first_name.strip() + " " + passport_middle_name.strip() + " " + passport_last_name.strip()
    if passport_middle_name == "":
        passport_full_name = passport_first_name + " " + passport_last_name
        passport_full_name_no_space = passport_first_name.strip() + " " + passport_last_name.strip()

    passport_birth_date_str = passport["birth_date"]
    passport_gender = passport["gender"]
    passport_country = passport["country"]
    passport_nationality = passport["nationality"]
    passport_number = passport["passport_number"]
    passport_issue_date_str = passport["passport_issue_date"]
    passport_expiry_date_str = passport["passport_expiry_date"]

    # Get client profile data
    profile_full_name = client_profile["name"]
    profile_country_of_domicile = client_profile["country_of_domicile"]
    profile_birth_date_str = client_profile["birth_date"]
    profile_nationality = client_profile["nationality"]
    profile_gender = client_profile["gender"]
    profile_passport_number = client_profile["passport_number"]
    profile_issue_date_str = client_profile["passport_issue_date"]
    profile_expiry_date_str = client_profile["passport_expiry_date"]

    # Check for consistency
    if passport_full_name != profile_full_name and passport_full_name_no_space != profile_full_name:
        return False
    if passport_birth_date_str != profile_birth_date_str or passport_gender != profile_gender or passport_nationality != profile_nationality:
        return False
    if passport_number != profile_passport_number or passport_issue_date_str != profile_issue_date_str or passport_expiry_date_str != profile_expiry_date_str:
        return False
    return True

def test_passport_consistency():
    clients_dir = Path("data/clients") 

    cnt = 0
    for client_dir in clients_dir.iterdir():           
        passport_path = client_dir / "passport.json"
        client_profile_path = client_dir / "client_profile.json"  
        label_path = client_dir / "label.json"

        passport = json.load(passport_path.open("r", encoding="utf-8"))
        client_profile = json.load(client_profile_path.open("r", encoding="utf-8"))
        label = json.load(label_path.open("r", encoding="utf-8")).get("label")

        is_consistent = client_profile_and_passport_are_consistent(client_profile, passport)
        if not is_consistent:
            cnt += 1
            assert label == "Reject", client_dir
    print(f"{cnt} rejects detected")

if __name__ == "__main__":
    test_passport_consistency()