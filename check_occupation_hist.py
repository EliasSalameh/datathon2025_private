import json
import re
from pathlib import Path


def employment_is_consistent(description, profile):
    """
    Checks if employment history in description matches profile data
    """
    try:
        employment_history = profile["employment_history"]
        occupation_text = description.get("Occupation History", "")
        sentences = [s.strip() for s in occupation_text.split("\n") if s.strip()]
    except KeyError:
        return False

    # Skip summary sentences containing "years of experience"
    filtered_sentences = sentences[1:]

    for i, job in enumerate(employment_history):
        company = job["company"]
        position = job["position"]
        start_year = str(job["start_year"])
        end_year = str(job["end_year"]) if job["end_year"] else None

        # Check company and position mention
        if company.strip() not in filtered_sentences[i] or position.strip() not in filtered_sentences[i]:
            return False

        # Extract all 4-digit years
        years = re.findall(r"\b\d{4}\b", filtered_sentences[i])

        if len(years) == 0:
            continue

        for year in years:
            if year not in [start_year, end_year]:
                return False

    return True


def test_employment_consistency():
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

        is_consistent = employment_is_consistent(description, profile)
        if not is_consistent:
            cnt += 1
            assert label == "Reject", f"Inconsistent employment in {client_dir.name}"

    print(f"{cnt} employment mismatches detected")


if __name__ == "__main__":
    test_employment_consistency()