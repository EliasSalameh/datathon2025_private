import os
import json
import re
from pathlib import Path

def extract_family_background(background: str):
    background = re.sub(r'\s{2,}', ' ', background)
    background = background.split(". ")

    marital_status_sentence = background[0].strip().lower()
    if len(background) == 1:
        children_number_sentence = ""
    else:
        children_number_sentence = background[1].strip()

    if "married" in marital_status_sentence or "tied the knot" in marital_status_sentence:
        profile_marital_status = "married"
    elif "widowed" in marital_status_sentence:
        profile_marital_status = "widowed"
    elif "single" in marital_status_sentence:
        profile_marital_status = "single"
    elif "divorced" in marital_status_sentence:
        profile_marital_status = "divorced"
    else:
        profile_marital_status = "unknown"
        children_number_sentence = marital_status_sentence

    match = re.search(r'\d+', children_number_sentence)
    if match:
        number = int(match.group())
    else:
        if "is named" in children_number_sentence: 
            number = 1
        elif "are named" in children_number_sentence:
            one_more_split = children_number_sentence.split("are named")[-1]
            number = one_more_split.count(',') + 2
        else:
            number = 0

    return {"marital_status": profile_marital_status, "number_of_children": number}


def family_background_is_consistent(description, profile, cache_dir_path, client_id): 
    """
    Uses an LLM to parse the family background description, storing the marital status and number of children on file
    Then, it checks if marital status is consistent
    """
    cur_dir_path = cache_dir_path / client_id
    os.makedirs(cur_dir_path, exist_ok=True)

    family_bckg_file_path = cur_dir_path / "family_background.json"
    if os.path.exists(family_bckg_file_path):
        family_background = json.load(family_bckg_file_path.open("r", encoding="utf-8"))
    else:
        family_background = extract_family_background(description["Family Background"])
        with open(family_bckg_file_path, "w", encoding="utf-8") as f:
            json.dump(family_background, f, indent=4)

    profile_marital_status = profile["marital_status"]
    if profile_marital_status != family_background["marital_status"]:
        return False
    return True


def test_account_form_and_client_profile_consistency():
    clients_dir = Path("data/clients") 
    cache_dir = Path("llm_outputs_train")

    cnt = 0
    for client_dir in clients_dir.iterdir():           
        client_id = os.path.basename(client_dir)
        account_form_path = client_dir / "account_form.json"
        client_profile_path = client_dir / "client_profile.json"  
        label_path = client_dir / "label.json"

        client_description = json.load((client_dir / "client_description.json").open("r", encoding="utf-8"))
        client_profile = json.load(client_profile_path.open("r", encoding="utf-8"))
        label = json.load(label_path.open("r", encoding="utf-8")).get("label")

        is_consistent = family_background_is_consistent(client_description, client_profile, cache_dir, client_id)
        if not is_consistent:
            cnt += 1
            assert label == "Reject", client_dir
    print(f"{cnt} rejects detected")

if __name__ == "__main__":
    test_account_form_and_client_profile_consistency()