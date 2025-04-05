# remove her, his; remove first name, middle name and last name, keep the rest

import os
import re
import json
from pathlib import Path
from check_passport_consistency import passport_is_consistent

def extract_summary_type(summary: str, passport: str):
    def remove_words(input_string, words_to_remove):
        words = input_string.split()
        filtered_words = [word for word in words if word.lower() not in words_to_remove]
        return ' '.join(filtered_words)

    summary = re.sub(r'\s{2,}', ' ', summary)
    first_name = passport["first_name"].strip().lower()
    middle_name = passport["middle_name"].strip().lower()
    last_name = passport["last_name"].strip().lower()

    to_remove = ["he", "she", "his", "her", "him", first_name, last_name] 
    if len(first_name.split()) > 1:
        to_remove.remove(first_name)
        to_remove.extend(first_name.split())
    if middle_name != "":
        to_remove.append(middle_name)
    summary = remove_words(summary, to_remove)
    return summary


def get_client_summary_type(description, passport, all_summary_types): 
    summary = extract_summary_type(description["Client Summary"], passport)
    return summary, all_summary_types[summary]

# This is to obtain the summary types from the training set
def get_all_client_summary_types():
    clients_dir = Path("data/clients") 

    all_summaries = set()
    for client_dir in clients_dir.iterdir():           
        client_description = json.load((client_dir / "client_description.json").open("r", encoding="utf-8"))
        client_passport = json.load((client_dir / "passport.json").open("r", encoding="utf-8"))

        if not passport_is_consistent(client_passport):
            continue

        cur_summary = extract_summary_type(client_description["Client Summary"], client_passport)
        if cur_summary not in all_summaries:
            all_summaries.add(cur_summary)
        
    with open("summary_types.txt", "w", encoding="utf-8") as f:
        for summary in sorted(all_summaries):
            f.write(summary + "\n")
