import os
import csv 
import json

from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from extract_files import extract_files
from check_passport_consistency import passport_is_consistent
from check_account_form import account_form_is_consistent
from client_profile_graduation_year import profile_is_consistent
from check_family_background_consistency import family_background_is_consistent
from cross_check_passport_client_profile_form import client_profile_and_passport_are_consistent
from cross_check_account_form_client_profile import account_form_and_client_profile_are_consistent
from cross_check_account_form_passport import account_form_and_passport_are_consistent

def process_single_client(client_dir, llm_output_path: Path):
    cur_client_id = os.path.basename(client_dir)

    account_form = json.load((client_dir / "account_form.json").open("r", encoding="utf-8"))
    client_description = json.load((client_dir / "client_description.json").open("r", encoding="utf-8"))
    client_profile = json.load((client_dir / "client_profile.json").open("r", encoding="utf-8"))
    passport = json.load((client_dir / "passport.json").open("r", encoding="utf-8"))

    if not passport_is_consistent(passport) or not account_form_is_consistent(account_form):
        return cur_client_id, "Reject"
    if not client_profile_and_passport_are_consistent(client_profile, passport):
        return cur_client_id, "Reject"
    if not account_form_and_client_profile_are_consistent(account_form, client_profile) or \
        not account_form_and_passport_are_consistent(account_form, passport):
        return cur_client_id, "Reject"
    if not profile_is_consistent(client_profile):
        return cur_client_id, "Reject"
    if not family_background_is_consistent(client_description, client_profile, llm_output_path, cur_client_id):
        return cur_client_id, "Reject"

    return cur_client_id, "Accept"

def get_predictions(data_path: str, llm_output_path: Path):
    clients_dir = os.path.join(data_path, 'clients')
    if not os.path.exists(clients_dir):
        extract_files(data_path)

    predicted_labels = {}
    clients_dir = Path(clients_dir)
    sorted_clients = sorted(clients_dir.iterdir(), key=lambda x: int(x.name.split('_')[1]))

    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(process_single_client, client_dir, llm_output_path): client_dir for client_dir in sorted_clients}
        for future in tqdm(as_completed(futures), total=len(futures)):
            client_id, status = future.result()
            predicted_labels[client_id] = status

    output_file = "predictions.csv"
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        for client_dir in sorted_clients:
            client_id = os.path.basename(client_dir)
            writer.writerow([client_id, predicted_labels[client_id]])


def get_real_train_set_solutions():
    # Do that at least once to get a correct solution.csv file
    client_ids = []
    real_labels = []
    clients_dir = Path("data/clients") 
    sorted_clients = sorted(clients_dir.iterdir(), key=lambda x: int(x.name.split('_')[1]))
    for client_dir in sorted_clients:   
        client_ids.append(os.path.basename(client_dir))
        label_path = client_dir / "label.json"
        label = json.load(label_path.open("r", encoding="utf-8")).get("label")
        real_labels.append(label)

    output_file = "solution.csv"
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        for client_id, status in zip(client_ids, real_labels):
            writer.writerow([client_id, status])

def compute_accuracy():
    def read_csv_to_dict(file_path):
        client_status = {}
        with open(file_path, mode='r') as file:
            reader = csv.reader(file, delimiter=';')
            for row in reader:
                client_status[row[0]] = row[1]
        return client_status

    status_file_real = read_csv_to_dict("solution.csv")
    status_file_predicted = read_csv_to_dict("predictions.csv")

    correct_matches = 0
    total_clients = len(status_file_real)
    for client_id in status_file_real:
        assert client_id in status_file_predicted
        if status_file_real[client_id] == status_file_predicted[client_id]:
            correct_matches += 1

    accuracy = correct_matches / total_clients * 100 

    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    cache_dir = Path("llm_outputs_train")
    get_predictions("data", cache_dir)
    compute_accuracy()