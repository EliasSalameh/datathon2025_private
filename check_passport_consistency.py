import json
from datetime import datetime
from pathlib import Path

def check_passport_consistency(account_form, client_description, client_profile, passport):
    """
    Checks consistency on the passport file only
    """

    # Reading data
    first_name = passport["first_name"]
    middle_name = passport["middle_name"]
    last_name = passport["last_name"]
    country = passport["country"]
    country_code = passport["country_code"]
    nationality = passport["nationality"]
    number = passport["passport_number"]
    birth_date_str = passport["birth_date"]
    gender = passport["gender"]
    mrz_line_0 = passport["passport_mrz"][0]
    mrz_line_1 = passport["passport_mrz"][1]
    issue_date_str = passport["passport_issue_date"]
    expiry_date_str = passport["passport_expiry_date"]

    ## GENDER CHECK
    if gender == "":
        return False

    ## MRZ CHECKS
    # Compute expected mrz line 0
    expected_mrz_line0 = "P<" + country_code + last_name.upper() + "<<" + first_name.upper()
    if middle_name != "":
        expected_mrz_line0 += "<" + middle_name.upper()
    while len(expected_mrz_line0) < len(mrz_line_0):
        expected_mrz_line0 += "<"
        
    # Compute expected mrz line 1
    expected_mrz_line1 = number + country_code + birth_date_str[2:4] + birth_date_str[5:7] + birth_date_str[8:10]
    while len(expected_mrz_line1) < len(mrz_line_1):
        expected_mrz_line1 += "<"

    # Check consistency of mrz 
    if mrz_line_0 != expected_mrz_line0 or mrz_line_1 != expected_mrz_line1:
        return False
    
    ## DATES CHECKS
    birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d").date()
    issue_date = datetime.strptime(issue_date_str, "%Y-%m-%d").date()
    expiry_date = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
    current_date = datetime.now().date()

    if birth_date > issue_date or issue_date > expiry_date or birth_date > current_date or issue_date > current_date:
        return False

    ## NATIONALITY CHECKS
    # Get valid country, country_code, nationality data (obtained from the accepted labels in the training set)
    with open('country_mappings.json', 'r') as f:
        country_data = json.load(f)     
    assert country in country_data

    if country_data[country][0] != country_code or country_data[country][1] != nationality:
        return False
    return True
   

def test_passport_consistency():
    base_dir = Path("data/clients") 

    cnt = 0
    for client_dir in base_dir.iterdir():           
        passport_path = client_dir / "passport.json"
        label_path = client_dir / "label.json"

        passport = json.load(passport_path.open("r", encoding="utf-8"))
        label = json.load(label_path.open("r", encoding="utf-8")).get("label")

        is_consistent = check_passport_consistency(None, None, None, passport)
        if not is_consistent:
            cnt += 1
            assert label == "Reject", client_dir
    print(f"{cnt} rejects detected")

test_passport_consistency()