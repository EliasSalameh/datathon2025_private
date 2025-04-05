import json
from pathlib import Path

def account_form_and_client_profile_are_consistent(account_form, client_profile):
    # Get passport data
    account_form_name = account_form["name"]
    account_form_passport_number = account_form["passport_number"]
    account_form_currency = account_form["currency"]
    account_form_adress = account_form["address"]  # this is a dictionary
    account_form_country_of_domicile = account_form["country_of_domicile"]
    account_form_phone_number = account_form["phone_number"].strip()
    account_form_email_address = account_form["email_address"]

    # Get client profile data
    profile_full_name = client_profile["name"]
    profile_address = client_profile["address"]  # this is a dictionary
    profile_country_of_domicile = client_profile["country_of_domicile"] 
    profile_passport_number = client_profile["passport_number"]
    profile_phone_number = client_profile["phone_number"].strip()
    profile_email_address = client_profile["email_address"]
    profile_currency = client_profile["currency"]

    # Check for consistency
    if account_form_name != profile_full_name or account_form_passport_number != profile_passport_number:
        return False
    if account_form_country_of_domicile != profile_country_of_domicile or account_form_currency != profile_currency:
        return False
    if profile_email_address != account_form_email_address or profile_address != account_form_adress:
        return False
    if profile_phone_number not in account_form_phone_number and account_form_phone_number not in profile_phone_number:
        return False
    return True

def test_account_form_and_client_profile_consistency():
    clients_dir = Path("data/clients") 

    cnt = 0
    for client_dir in clients_dir.iterdir():           
        account_form_path = client_dir / "account_form.json"
        client_profile_path = client_dir / "client_profile.json"  
        label_path = client_dir / "label.json"

        account_form = json.load(account_form_path.open("r", encoding="utf-8"))
        client_profile = json.load(client_profile_path.open("r", encoding="utf-8"))
        label = json.load(label_path.open("r", encoding="utf-8")).get("label")

        is_consistent = account_form_and_client_profile_are_consistent(account_form, client_profile)
        if not is_consistent:
            cnt += 1
            assert label == "Reject", client_dir
    print(f"{cnt} rejects detected")

if __name__ == "__main__":
    test_account_form_and_client_profile_consistency()