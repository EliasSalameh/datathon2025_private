import json
from pathlib import Path

def education_is_consistent(description, profile):
    """
    Checks if education details in description match profile data
    """
    # Extract education data from profile
    try:
        secondary = {
            "name": profile["secondary_school"]["name"],
            "year": str(profile["secondary_school"]["graduation_year"])
        }
        higher_edus = [{
            "name": edu["university"],
            "year": str(edu["graduation_year"])
        } for edu in profile["higher_education"]]
    except KeyError:
        return False

    # Process education text from description
    edu_text = description.get("Education Background", "")
    parts = [p.strip() for p in edu_text.split("\n") if p.strip()]

    # Check secondary school
    secondary_found = any(
        secondary["name"] in part and secondary["year"] in part
        for part in parts
    )

    # Check higher education
    higher_matches = []
    for edu in higher_edus:
        found = any(
            edu["name"] in part and edu["year"] in part
            for part in parts
        )
        higher_matches.append(found)

    return secondary_found and all(higher_matches)

def test_education_consistency():
    clients_dir = Path("data/clients")
    cnt = 0

    for client_dir in clients_dir.iterdir():
        profile_path = client_dir / "client_profile.json"
        desc_path = client_dir / "client_description.json"
        label_path = client_dir / "label.json"

        if not all([profile_path.exists(), desc_path.exists(), label_path.exists()]):
            continue

        with profile_path.open("r", encoding="utf-8") as f:
            profile = json.load(f)
        with desc_path.open("r", encoding="utf-8") as f:
            description = json.load(f)
        label = json.load(label_path.open("r")).get("label")

        is_consistent = education_is_consistent(description, profile)
        if not is_consistent:
            cnt += 1
            assert label == "Reject", f"Inconsistent education in {client_dir.name}"

    print(f"{cnt} education mismatches detected")

if __name__ == "__main__":
    test_education_consistency()