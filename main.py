import os
import csv
import json

from utils import *
from tqdm import tqdm
from pathlib import Path
from helpers import compute_accuracy

from extract_files import extract_files
from get_client_summary_feature import get_client_summary_type
from check_passport_consistency import passport_is_consistent
from check_account_form import account_form_is_consistent
from client_profile_graduation_year import profile_is_consistent
from check_family_background_consistency import family_background_is_consistent
from cross_check_passport_client_profile_form import client_profile_and_passport_are_consistent
from cross_check_account_form_client_profile import account_form_and_client_profile_are_consistent
from cross_check_account_form_passport import account_form_and_passport_are_consistent
from check_age_consistency import age_is_consistent
from check_education_background import education_is_consistent
from check_occupation_hist import employment_is_consistent
from check_wealth_summary import wealth_is_consistent

def get_predictions(data_path: str, llm_output_path: Path):
    clients_dir = os.path.join(data_path, 'clients')
    if not os.path.exists(clients_dir):
        extract_files(data_path)

    # Obtain the client summary types for more feature engineering
    with open("summary_types.txt", "r", encoding="utf-8") as f:
        all_summary_types_list = f.readlines()
        all_summary_types = {}
        for idx, summary_type in enumerate(all_summary_types_list):
            summary_type = summary_type[:-1]
            all_summary_types[summary_type] = idx

    predicted_labels = {}
    clients_dir = Path(data_path + "/clients")
    sorted_clients = sorted(clients_dir.iterdir(), key=lambda x: int(x.name.split('_')[1]))
    cur_summary_type_list = {}
    for client_dir in tqdm(sorted_clients):
        client_id = os.path.basename(client_dir)

        account_form = json.load((client_dir / "account_form.json").open("r", encoding="utf-8"))
        client_description = json.load((client_dir / "client_description.json").open("r", encoding="utf-8"))
        client_profile = json.load((client_dir / "client_profile.json").open("r", encoding="utf-8"))
        passport = json.load((client_dir / "passport.json").open("r", encoding="utf-8"))

        if not passport_is_consistent(passport) or not account_form_is_consistent(account_form):
            predicted_labels[client_id] = "Reject"
        elif not client_profile_and_passport_are_consistent(client_profile, passport):
            predicted_labels[client_id] = "Reject"
        elif not account_form_and_client_profile_are_consistent(account_form,
                                                                client_profile) or not account_form_and_passport_are_consistent(
                account_form, passport):
            predicted_labels[client_id] = "Reject"
        elif not profile_is_consistent(client_profile):
            predicted_labels[client_id] = "Reject"
        elif not family_background_is_consistent(client_description, client_profile, llm_output_path, client_id):
            predicted_labels[client_id] = "Reject"
        elif not age_is_consistent(client_description, client_profile):
            predicted_labels[client_id] = "Reject"
        elif not education_is_consistent(client_description, client_profile):
            predicted_labels[client_id] = "Reject"
        elif not employment_is_consistent(client_description, client_profile):
            predicted_labels[client_id] = "Reject"
        elif not wealth_is_consistent(client_description, client_profile):
            predicted_labels[client_id] = "Reject"
        else:
            cur_summary, cur_summary_type = get_client_summary_type(client_description, passport, all_summary_types)
            cur_summary_type_list[int(client_id.split('_')[-1])] = cur_summary_type
            predicted_labels[client_id] = "Accept"

    # We pass consistent cases through an ensemble model of boosted and random trees to determine whether we should reject them or not
    df = load_clients_data(clients_dir, llm_output_path, cur_summary_type_list)
    mask_prelim = [
        1 if predicted_labels[f"client_{client_id}"] == 'Accept' else 0
        for client_id in sorted([int(k.split("_")[1]) for k in predicted_labels.keys()])
    ]
    df = df[pd.Series(mask_prelim).astype(bool)]
    df.loc[:, 'summary_type'] = df.index.map(cur_summary_type_list)
    # get dummies from summary_type
    df = craft_features(df)
    X, y = get_X_y(df, save_x_y=True)
    X_train, X_val, y_train, y_val = get_X_y_split(X, y, test_ratio=0.3)

    trained_model_CB = CatBoost_CV(X_train, y_train, X_val, y_val, save_model=True, k=10,
                                   plot_importance=False)  # Takes v long to CV
    y_pred_CB = CatBoost_Predicitons(trained_model_CB, X_val, y_val)

    train_model_RF = RandomForestClassifier_train(X_train, y_train, save_model=False, k=10)
    y_pred_RF = RandomForestClassifier_predict(train_model_RF, X_val, y_val)

    y_pred = [1 if y_pred_CB[i] or y_pred_RF[i] else 0 for i in range(len(y_pred_CB))]

    for i, client_id in enumerate(X_val.index):
        mask_prelim[client_id] = y_pred[i]

    for i, pred in enumerate(mask_prelim):
        predicted_labels[f"client_{i}"] = 'Accept' if pred else 'Reject'
    # Write predictions to csv file

    output_file = "predictions.csv"
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        for client_dir in sorted_clients:
            client_id = os.path.basename(client_dir)
            writer.writerow([client_id, predicted_labels[client_id]])


if __name__ == "__main__":
    cache_dir = Path("data")
    get_predictions("data", cache_dir)
    compute_accuracy()