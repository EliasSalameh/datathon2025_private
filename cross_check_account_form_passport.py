import json
from pathlib import Path
from tqdm import tqdm


def check_passport_consistency(data_acc, data_passport):
    if data_acc.get("first_name", "").lower().strip() != data_passport.get("first_name", "").lower().strip():
        return False

    if data_acc.get("middle_name", "").lower().strip() != data_passport.get("middle_name", "").lower().strip():
        return False

    if data_acc.get("last_name", "").lower().strip() != data_passport.get("last_name", "").lower().strip():
        return False

    if isinstance(data_acc.get("passport_number"), list):
        data_acc["passport_number"] = data_acc.get("passport_number")[0]

    if data_acc.get("passport_number", "").strip() != data_passport.get("passport_number", "").strip():
        return False

    return True


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

    is_consistent = check_passport_consistency(data_acc, data_passport)
    return is_consistent, label


def test_passport_form_consistency():
    clients_dir = Path("clients")
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