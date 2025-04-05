import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
client = OpenAI(api_key= os.environ.get("OPEN_AI_KEY"))

class FamilyBackgroundExtraction(BaseModel):
    marital_status: str
    number_of_children: int

def extract_family_background(text: str):
    prompt = f""" You are given a few sentences describing the familial background of an individual. Your job is to extract this individual's marital status (which must be one of married, single, divorced, widowed) and their number of children. 
    Examples:
    Input 1: Scholz Schmidt Fischer and Gwendolyn tied the knot in 1999. Their children are named Anna, Herrmann and Schulz.
    Output 1: 
    {{ 
        "marital_status": "married",
        "number_of_children": 3,       
    }}
    
    Now extract the marital status and number of children from the following sentences:
    {text}
    """

    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at structured data extraction. You will be given unstructured text and should convert it into the given structure."},
                {"role": "user", "content": prompt}
            ],
            response_format=FamilyBackgroundExtraction,
        )
        return response.choices[0].message.parsed.model_dump()
    except Exception as e:
        print("LLM Exception")
        print(e)
        return {"marital_status": "single", "number_of_children": 0}


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
        with open("output.json", "w", encoding="utf-8") as f:
            json.dump(family_background, f, indent=4)

    profile_marital_status = profile["marital_status"]
    if profile_marital_status != family_background["marital_status"]:
        return False
    return True


# THIS IS JUST TO BE SURE IT WORKS, TO REMOVE CZ IT NEEDS LABEL
def family_background_is_consistent_with_label(description, profile, cache_dir_path, client_id, label): 
    """
    Uses an LLM to parse the family background description, storing the marital status and number of children on file
    Then, it checks if marital status is consistent
    """
    cur_dir_path = cache_dir_path + "/" + client_id
    os.makedirs(cur_dir_path, exist_ok=True)

    family_bckg_file_path = cur_dir_path + "/family_background.json"
    if os.path.exists(family_bckg_file_path):
        with open(family_bckg_file_path, "r", encoding="utf-8") as file:
            family_background = json.load(file)
    else:
        family_background = extract_family_background(description["Family Background"])
        with open(family_bckg_file_path, "w", encoding="utf-8") as f:
            json.dump(family_background, f, indent=4)

    profile_marital_status = profile["marital_status"]
    if profile_marital_status != family_background["marital_status"]:
        assert label == "Reject", client_id
        return False
    return True

