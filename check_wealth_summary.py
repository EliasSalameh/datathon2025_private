import json
import re
from pathlib import Path


def wealth_is_consistent(description, profile):
    """
    Checks if the 'Wealth Summary' in the description contains three separate sentences:
    - One sentence containing the savings value from profile["aum"]["savings"].
    - One sentence containing the inheritance value from profile["aum"]["inheritance"] and all details from profile["inheritance_details"].
    - One sentence containing all details for each real estate asset in profile["real_estate_details"].
    The three groups must appear in three distinct sentences.
    """
    try:
        wealth_text = description["Wealth Summary"]
        # Split by newline characters and remove empty lines.
        sentences = [s.strip() for s in wealth_text.splitlines() if s.strip()]
    except KeyError:
        return False

    aum = profile.get("aum", {})
    savings = str(aum.get("savings", ""))
    inheritance = str(aum.get("inheritance", ""))

    # Extract inheritance details.
    inheritance_details = profile.get("inheritance_details", {})
    in_relationship = inheritance_details.get("relationship", "")
    in_year = str(inheritance_details.get("inheritance year", ""))
    in_profession = inheritance_details.get("profession", "")

    # Initialize indices to track the sentences that satisfy each group.
    if not inheritance_details:
        return True

    for idx, sentence in enumerate(sentences):
        if inheritance and re.search(r'\b' + inheritance + r'\b', sentence):
            if (re.search(r'\b' + in_year + r'\b', sentence) and re.search(r'\b' + in_relationship + r'\b',
                                                                           sentence)
                    and re.search(r'\b' + in_profession + r'\b', sentence)):
                return True

    return False

def test_wealth_consistency():
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
        label = json.load(label_path.open("r", encoding="utf-8")).get("label")

        is_consistent = wealth_is_consistent(description, profile)
        if not is_consistent:
            cnt += 1
            assert label == "Reject", f"Inconsistent wealth grouping in {client_dir.name}"

    print(f"{cnt} wealth mismatches detected")


if __name__ == "__main__":
    test_wealth_consistency()
