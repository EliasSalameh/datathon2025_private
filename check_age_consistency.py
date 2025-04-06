import json
import re
from datetime import datetime
from pathlib import Path


def age_is_consistent(description, profile):
    """
    Checks if declared age in description matches calculated age from profile
    using fixed current date 2025-04-01
    """
    # Extract declared age from description
    declared_age = None
    for field in ["Summary Note", "Occupation History"]:
        text = description.get(field, "")
        match = re.search(r"(\d+)\s+year old", text)
        if match:
            declared_age = int(match.group(1))
            break

    # No age declared - nothing to check
    if declared_age is None:
        return True

    # Calculate actual age from profile
    birth_date = datetime.strptime(profile["birth_date"], "%Y-%m-%d").date()
    current_date = datetime.strptime("2025-04-01", "%Y-%m-%d").date()

    age = current_date.year - birth_date.year
    if (current_date.month, current_date.day) < (birth_date.month, birth_date.day):
        age -= 1

    return declared_age == age


def test_age_consistency():
    clients_dir = Path("data/clients")
    cnt = 0

    for client_dir in clients_dir.iterdir():
        desc_path = client_dir / "client_description.json"
        profile_path = client_dir / "client_profile.json"
        label_path = client_dir / "label.json"

        if not all([desc_path.exists(), profile_path.exists(), label_path.exists()]):
            continue

        with desc_path.open("r", encoding="utf-8") as f:
            description = json.load(f)
        with profile_path.open("r", encoding="utf-8") as f:
            profile = json.load(f)
        label = json.load(label_path.open("r")).get("label")

        is_consistent = age_is_consistent(description, profile)
        if not is_consistent:
            cnt += 1
            assert label == "Reject", f"Failed on {client_dir.name}"

    print(f"{cnt} age mismatches detected")


if __name__ == "__main__":
    test_age_consistency()