import os
import csv 
import json

from utils import *
from pathlib import Path

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
