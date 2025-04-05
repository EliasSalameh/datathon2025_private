import os
import csv 
import json

from tqdm import tqdm
from pathlib import Path
from extract_files import extract_files
from check_passport_consistency import passport_is_consistent
from check_account_form import account_form_is_consistent
from cross_check_passport_client_profile_form import client_profile_and_passport_are_consistent
from cross_check_account_form_client_profile import account_form_and_client_profile_are_consistent
from cross_check_account_form_passport import account_form_and_passport_are_consistent

def get_predictions(data_path: str):
    clients_dir = os.path.join(data_path, 'clients')
    if not os.path.exists(clients_dir):
        extract_files(data_path)

    client_ids = []
    predicted_labels = []
    clients_dir = Path(data_path + "/clients") 
    sorted_clients = sorted(clients_dir.iterdir(), key=lambda x: int(x.name.split('_')[1]))
    for client_dir in tqdm(sorted_clients):   
        client_ids.append(os.path.basename(client_dir))

        account_form_path = client_dir / "account_form.json"
        client_description_path = client_dir / "client_description.json"
        client_profile_path = client_dir / "client_profile.json"  
        passport_path = client_dir / "passport.json"

        account_form = json.load(account_form_path.open("r", encoding="utf-8"))
        client_description = json.load(client_description_path.open("r", encoding="utf-8"))
        client_profile = json.load(client_profile_path.open("r", encoding="utf-8"))
        passport = json.load(passport_path.open("r", encoding="utf-8"))

        if not passport_is_consistent(passport) or not account_form_is_consistent(account_form):
            predicted_labels.append("Reject")
            continue
        if not client_profile_and_passport_are_consistent(client_profile, passport):
            predicted_labels.append("Reject")
            continue
        if not account_form_and_client_profile_are_consistent(account_form, client_profile) or not account_form_and_passport_are_consistent(account_form, passport):
            predicted_labels.append("Reject")
            continue

        #TODO Add more logic here
    
        predicted_labels.append("Accept")
    
    # Write predictions to csv file
    output_file = "predictions.csv"
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        for client_id, status in zip(client_ids, predicted_labels):
            writer.writerow([client_id, status])
    

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

get_predictions("data")
compute_accuracy()