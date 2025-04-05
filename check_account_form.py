import json
import sys
import glob
from tqdm import tqdm

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

def full_check(data):
    consistent_name = check_name_consistency(data)
    correct_email = check_email(data)
    single_domicile = is_single_domicile(data)

    return consistent_name and correct_email and single_domicile


def main():
    num_clients = len(glob.glob("clients/*"))

    for i in tqdm(range(num_clients)):
        # with open(f"datathon2025_2/client_{i}/label.json") as f:
        #     label = json.load(f)['label']

        with open(f"clients/client_{i}/account_form.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        status = full_check(data)

if __name__ == "__main__":
    main()
