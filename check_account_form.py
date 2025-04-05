import json
from pathlib import Path

def check_name_consistency(data):
    full_name = " ".join(data.get("name", "").split()).strip()
    first = data.get("first_name", "").strip()
    middle = data.get("middle_name", "").strip()
    last = data.get("last_name", "").strip()

    if middle:
        expected_full_name = f"{first} {middle} {last}"
    else:
        expected_full_name = f"{first} {last}"
    return True if full_name == expected_full_name else False

def check_email(data):
    return "@" in data.get("email_address")

def is_single_domicile(data):
    return len(data.get("country_of_domicile").split(",")) == 1

def check_address_zip(data):
    return data.get("address").get("postal code") != ""

def account_form_is_consistent(account_form):
    consistent_name = check_name_consistency(account_form)
    correct_email = check_email(account_form)
    single_domicile = is_single_domicile(account_form)
    address_zip = check_address_zip(account_form)

    return consistent_name and correct_email and single_domicile and address_zip 


def test_account_form_consistency():
    clients_dir = Path("data/clients") 

    cnt = 0
    for client_dir in clients_dir.iterdir():           
        account_form_path = client_dir / "account_form.json"
        label_path = client_dir / "label.json"

        account_form = json.load(account_form_path.open("r", encoding="utf-8"))
        label = json.load(label_path.open("r", encoding="utf-8")).get("label")

        is_consistent = account_form_is_consistent(account_form)
        if not is_consistent:
            cnt += 1
            assert label == "Reject", client_dir
    print(f"{cnt} rejects detected")

if __name__ == "__main__":
    test_account_form_consistency()
