import json
from pathlib import Path
from tqdm import tqdm


def account_form_and_passport_are_consistent(account_form, passport):
    if account_form.get("first_name", "").lower().strip() != passport.get("first_name", "").lower().strip():
        return False

    if account_form.get("middle_name", "").lower().strip() != passport.get("middle_name", "").lower().strip():
        return False

    if account_form.get("last_name", "").lower().strip() != passport.get("last_name", "").lower().strip():
        return False

    if isinstance(account_form.get("passport_number"), list):
        account_form["passport_number"] = account_form.get("passport_number")[0]

    if account_form.get("passport_number", "").strip() != passport.get("passport_number", "").strip():
        return False

    return True


def test_passport_form_consistency():
    def passport_form_is_consistent(client_dir):
        account_form_path = client_dir / "account_form.json"
        passport_form_path = client_dir / "passport.json"
        label_path = client_dir / "label.json"

        with account_form_path.open("r", encoding="utf-8") as f:
            data_acc = json.load(f)

        with passport_form_path.open("r", encoding="utf-8") as f:
            data_passport = json.load(f)

        with label_path.open("r", encoding="utf-8") as f:
            label = json.load(f).get("label")

        is_consistent = account_form_and_passport_are_consistent(data_acc, data_passport)
        return is_consistent, label

    clients_dir = Path("data/clients")
    rejections = 0
    for client_dir in tqdm(list(clients_dir.iterdir())):
        is_consistent, label = passport_form_is_consistent(client_dir)

        if not is_consistent:
            rejections += 1
            if label == "Accept":
                print(f"Wrong cross check in {client_dir}")

    print(f"{rejections} rejects detected")


if __name__ == "__main__":
    test_passport_form_consistency()