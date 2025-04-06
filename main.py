import os
import csv 
import json

from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.impute import SimpleImputer

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
from utils import *

def get_predictions(data_path: str, llm_output_path: Path):
    clients_dir = os.path.join(data_path, 'clients')
    if not os.path.exists(clients_dir):
        extract_files(data_path)
    
    # Obtain the client summary types for more feature engineering
    with open("/home/elias/Anatomic-Diffusion-Models/configs/experiment/autoencoder/summary_types.txt", "r", encoding="utf-8") as f:
        all_summary_types_list = f.readlines()
        all_summary_types = {}
        for idx, summary_type in enumerate(all_summary_types_list):
            summary_type = summary_type[:-1]
            all_summary_types[summary_type] = idx

    a = 0
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
        elif not account_form_and_client_profile_are_consistent(account_form, client_profile) or not account_form_and_passport_are_consistent(account_form, passport):
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
            a += 1
            cur_summary, cur_summary_type = get_client_summary_type(client_description, passport, all_summary_types)
            cur_summary_type_list[int(client_id.split('_')[-1])] = cur_summary_type
            predicted_labels[client_id] = "Accept"
    
    print(f"Number of clients with no revealed inconsistency yet: {a}")
    
    df = load_clients_data(clients_dir, llm_output_path, cur_summary_type_list)
    mask_prelim = [
    1 if predicted_labels[f"client_{client_id}"] == 'Accept' else 0
    for client_id in sorted([int(k.split("_")[1]) for k in predicted_labels.keys()])
    ]
    df = df[pd.Series(mask_prelim).astype(bool)]
    df.loc[:, 'summary_type'] = df.index.map(cur_summary_type_list)
    df = craft_features(df)
    X = df
    #NOTE: This is when training
    # X, y = get_X_y(df,save_x_y=True)
    # X_train, X_val, y_train, y_val = get_X_y_split(X, y, test_ratio=0.0)
    # trained_model_CB = CatBoost_CV(X_train, y_train, X_val, y_val, save_model=True, k=20, plot_importance=False)   #Takes v long to CV
    # trained_model_CB = CatBoost_Train(X, y, save_model=True, k=30)   
    # train_model_RF = RandomForestClassifier_train(X, y, save_model=True, k=30)
    
    #NOTE: Just loading checkpointed models
    trained_model_CB = CatBoostClassifier()
    trained_model_CB.load_model("catboost_model_topK_full_filtered.cbm")
    
    train_model_RF = joblib.load('rf_model_topK_full_filtered.pkl')
    
    X_val = X
    
    y_pred_CB = CatBoost_Predicitons(trained_model_CB, X_val, y_val=None)
    y_pred_RF = RandomForestClassifier_predict(train_model_RF, X_val, y_val=None)
    
    
    
    
    y_pred = [1 if (0.5*y_pred_CB[i] + 0.5*y_pred_RF[i]) > 0.55  else 0 for i in range(len(y_pred_CB))]
    
    #NOTE: This is when training
    # print(f"Accuracy of CatBoost: {accuracy_score(y_val,y_pred_CB > 0.5):.4f}")
    # print(f"Accuracy of Random Forest: {accuracy_score(y_val,y_pred_RF > 0.5):.4f}")
    # print(f"Accuracy of Ensemble: {accuracy_score(y_val,y_pred):.4f}")
    # print(f"Confusion Matrix of Ensemble:")
    # print(confusion_matrix(y_val,y_pred))
    
    for i, client_id in enumerate(X_val.index):
        mask_prelim[client_id] = y_pred[i]

    for i,pred in enumerate(mask_prelim):
        predicted_labels[f"client_{i}"] = 'Accept' if pred else 'Reject'
    # Write predictions to csv file

    output_file = "kazanon.csv"
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        for client_dir in sorted_clients:
            client_id = os.path.basename(client_dir)
            writer.writerow([client_id, predicted_labels[client_id]])

def get_real_train_set_solutions():
    # Do that at least once to get a correct solution.csv file
    client_ids = []
    real_labels = []
    clients_dir = Path("/home/elias/Anatomic-Diffusion-Models/configs/experiment/autoencoder/data/clients") 
    sorted_clients = sorted(clients_dir.iterdir(), key=lambda x: int(x.name.split('_')[1]))
    for client_dir in sorted_clients:   
        client_ids.append(os.path.basename(client_dir))
        label_path = client_dir / "label.json"
        label = json.load(label_path.open("r", encoding="utf-8")).get("label")
        real_labels.append(label)

    output_file = "/home/elias/Anatomic-Diffusion-Models/configs/experiment/autoencoder/solution.csv"
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

    status_file_real = read_csv_to_dict("/home/elias/Anatomic-Diffusion-Models/configs/experiment/autoencoder/solution.csv")
    status_file_predicted = read_csv_to_dict("predictions.csv")

    correct_matches = 0
    total_clients = len(status_file_real)
    for client_id in status_file_real:
        assert client_id in status_file_predicted
        if status_file_real[client_id] == status_file_predicted[client_id]:
            correct_matches += 1

    accuracy = correct_matches / total_clients * 100 

    print(f"Accuracy: {accuracy:.2f}%")

if _name_ == "_main_":
    cache_dir = Path("/home/elias/Anatomic-Diffusion-Models/data_test/llm_outputs_train")
    get_predictions("/home/elias/Anatomic-Diffusion-Models/data_test", cache_dir)
    # compute_accuracy()