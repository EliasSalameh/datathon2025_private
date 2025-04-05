import os
import json
# !pip install pandas
# !pip install scikit-learn
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from sklearn.model_selection import train_test_split
import catboost
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def flatten_json(y, prefix=''):
    """Recursively flattens nested JSON"""
    out = {}
    for k, v in y.items():
        key = f"{prefix}{k}".replace(" ", "_").lower()
        if isinstance(v, dict):
            out.update(flatten_json(v, prefix=key + "_"))
        elif isinstance(v, list):
            out[key] = v #str(v)  # You can choose to serialize or extract specific features
        else:
            out[key] = v
    return out

def load_clients_data(base_path):
    client_rows = []
    clients_dir = Path(base_path) 
    sorted_clients = sorted(clients_dir.iterdir(), key=lambda x: int(x.name.split('_')[1]))

    for client_folder in tqdm(sorted_clients): #os.listdir(base_path):
        client_path = client_folder
        if not os.path.isdir(client_path):
            print(f"Warning: {client_path} is not a directory")
            continue

        client_data = {}
        for file_name in ['passport.json', 'client_profile.json', 'account_form.json', 'label.json']:
            file_path = os.path.join(client_path, file_name)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        flat_data = flatten_json(data)
                        client_data.update(flat_data)
                    except json.JSONDecodeError:
                        print(f"Warning: could not decode {file_path}")

        # Normalize label
        label = client_data.get('label', '').lower()
        client_data['label'] = 1 if label == 'accept' else 0

        client_rows.append(client_data)

    return pd.DataFrame(client_rows)

def craft_features(filtered_df):
    # filtered_df 
    # country_counts = filtered_df["country"].value_counts()
    # print(country_counts)   #NOTE: Removing country as feature
    # nationality_count = filtered_df["nationality"].value_counts()
    marital_status_count = filtered_df["marital_status"].value_counts()   #NOTE: 3 values, will turn to categorical
    print(marital_status_count)
    # print(nationality_count) 
    # inheritance_details_rs_count = filtered_df["inheritance_details_relationship"].value_counts()   #NOTE: 5 values, will turn to categorical
    # print(inheritance_details_rs_count)
    inheritance_details_profession_count = filtered_df["inheritance_details_profession"].value_counts()   #NOTE: 11 values, will turn to categorical
    print(inheritance_details_profession_count)
    filtered_df_ = filtered_df.drop(columns=["country","nationality","first_name","last_name","middle_name","passport_number","passport_issue_date","passport_expiry_date", "gender","country_code","passport_mrz","name","address_city","address_street_name","address_street_number","address_postal_code","email_address","secondary_school_name","country_of_domicile","phone_number"])
    filtered_df__ = filtered_df_

    current_date = pd.to_datetime("2025-04-01")
    filtered_df__["birth_date"] = pd.to_datetime(filtered_df__["birth_date"], errors="coerce")
    filtered_df__["inheritance_details_inheritance_year"] = pd.to_numeric(filtered_df__["inheritance_details_inheritance_year"], errors="coerce")
    filtered_df__["secondary_school_graduation_year"] = pd.to_numeric(filtered_df__["secondary_school_graduation_year"], errors="coerce")

    filtered_df__["age"] = filtered_df__["birth_date"].apply(lambda x: current_date.year - x.year - ((current_date.month, current_date.day) < (x.month, x.day)) if pd.notnull(x) else None)
    filtered_df__["inheritance_age"] = filtered_df__["inheritance_details_inheritance_year"] - filtered_df__["birth_date"].dt.year# filtered_df__["secondary_school_graduation_age"] = filtered_df__["secondary_school_graduation_year"].apply(lambda x: current_date.year - x.year - ((current_date.month, current_date.day) < (x.month, x.day)) if pd.notnull(x) else None)
    filtered_df__["secondary_school_graduation_age"] = filtered_df__["secondary_school_graduation_year"] - filtered_df__["birth_date"].dt.year
    filtered_df__["number_of_universities"] = filtered_df__["higher_education"].apply(lambda x: len(x) if isinstance(x, list) else 0)

    filtered_df__["earliest_university_graduation_age"] = filtered_df__.apply(
        lambda row: min([int(university["graduation_year"]) for university in row["higher_education"]]) - row["birth_date"].year
        if isinstance(row["higher_education"], list) and len(row["higher_education"]) > 0 else None,
        axis=1
    )
    filtered_df__["latest_university_graduation_age"] = filtered_df__.apply(
        lambda row: max([int(university["graduation_year"]) for university in row["higher_education"]]) - row["birth_date"].year
        if isinstance(row["higher_education"], list) and len(row["higher_education"]) > 0 else None,
        axis=1
    )

    current_year = 2025  # Assuming current year is 2025
    filtered_df__["total_years_of_employment"] = filtered_df__["employment_history"].apply(
        lambda x: sum([(current_year - job["start_year"]) if job["end_year"] is None else (job["end_year"] - job["start_year"]) for job in x])
        if isinstance(x, list) else 0
    )
    filtered_df__["num_jobs"] = filtered_df__["employment_history"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    filtered_df__["longest_job_duration"] = filtered_df__["employment_history"].apply(
        lambda x: max([(current_year - job["start_year"]) if job["end_year"] is None else (job["end_year"] - job["start_year"]) for job in x], default=0)
        if isinstance(x, list) else 0
    )
    filtered_df__["average_salary"] = filtered_df__["employment_history"].apply(
        lambda x: sum([job["salary"] for job in x if job["salary"] is not None]) / len(x) if isinstance(x, list) and len(x) > 0 else None
    )
    filtered_df__["most_recent_job_end_age"] = filtered_df__.apply(
    lambda row: max(
        [
            (job["end_year"] - row["birth_date"].year) if job["end_year"] is not None 
            else (current_year - row["birth_date"].year)
            for job in row["employment_history"]
        ],
        default=None
    ) if isinstance(row["employment_history"], list) else None,
    axis=1
    )
    
    filtered_df__["most_recent_job_start_age"] = filtered_df__.apply(
    lambda row: max(
        [job["start_year"] - row["birth_date"].year for job in row["employment_history"]],
        default=None
    ) if isinstance(row["employment_history"], list) else None,
    axis=1
    )

    property_types = filtered_df__["real_estate_details"].apply(
        lambda x: [property['property type'] for property in x] if isinstance(x, list) else []
    )

    # Flatten the list of lists and get unique values
    # unique_property_types = set([item for sublist in property_types for item in sublist])       #NOTE: This was to get the available property types
    # Print all unique property types
    # print(unique_property_types)

    property_types = ['flat', 'villa', 'townhouse', 'condo', 'house']

    filtered_df__["total_property_count"] = filtered_df__["real_estate_details"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )

    # Calculate total investment value across all properties
    filtered_df__["total_property_value"] = filtered_df__["real_estate_details"].apply(
        lambda x: sum(property.get('property value', 0) for property in x) if isinstance(x, list) else 0
    )

    # Count distinct property types (e.g., flat, villa, condo)
    filtered_df__["num_property_types"] = filtered_df__["real_estate_details"].apply(
        lambda x: len(set(property.get('property type') for property in x)) if isinstance(x, list) else 0
    )

    # Calculate max property value
    filtered_df__["max_property_value"] = filtered_df__["real_estate_details"].apply(
        lambda x: max((property.get('property value', 0) for property in x), default=0) if isinstance(x, list) else 0
    )

    # Calculate min property value
    filtered_df__["min_property_value"] = filtered_df__["real_estate_details"].apply(
        lambda x: min((property.get('property value', 0) for property in x), default=0) if isinstance(x, list) else 0
    )

    filtered_df__ = pd.get_dummies(filtered_df__, columns=["marital_status"], drop_first=False)
    filtered_df__ = pd.get_dummies(filtered_df__, columns=["inheritance_details_profession"], drop_first=False)
    filtered_df__ = pd.get_dummies(filtered_df__, columns=["inheritance_details_relationship"], drop_first=False)
    filtered_df__ = pd.get_dummies(filtered_df__, columns=["investment_risk_profile"], drop_first=False)
    filtered_df__ = pd.get_dummies(filtered_df__, columns=["investment_horizon"], drop_first=False)
    filtered_df__ = pd.get_dummies(filtered_df__, columns=["investment_experience"], drop_first=False)
    filtered_df__ = pd.get_dummies(filtered_df__, columns=["type_of_mandate"], drop_first=False)
    filtered_df__ = pd.get_dummies(filtered_df__, columns=["currency"], drop_first=False)

    available_markets = ['Spain', 'Denmark', 'Germany', 'Italy', 'Netherlands', 'France', 'Finland', 'Switzerland', 'Belgium', 'Austria']

    # Convert 'preferred_markets' into categorical features (one per available market)
    for market in available_markets:
        filtered_df__[market] = filtered_df__["preferred_markets"].apply(lambda x: 1 if market in x else 0)

    filtered_df__.drop(columns=["birth_date","inheritance_details_inheritance_year","secondary_school_graduation_year",
    "higher_education","employment_history","real_estate_details", "preferred_markets"], inplace=True)


    filtered_df__ = filtered_df__.apply(lambda x: x.astype(int) if x.dtype == 'bool' else x)


    return filtered_df__

def get_X_y_split(filtered_df__, test_ratio=0.2, save_x_y=False):
    """
    Splits the DataFrame into features (X) and target variable (y).
    """
    # Ensure the 'label' column is present
    if "label" not in filtered_df__.columns:
        raise ValueError("The DataFrame must contain a 'label' column.")

        # Split into features and target variable
    X = filtered_df__.drop(columns=["label"])  # Drop the 'label' column for features
    y = filtered_df__["label"]  # The 'label' column will be the target variable
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_ratio, random_state=42)
    if save_x_y:
        X.to_csv("X.csv", index=False)
        y.to_csv("y.csv", index=False)
    return X_train, X_val, y_train, y_val



def CatBoost_Predicitons(X_train, X_val, y_train, y_val):
    # Initialize CatBoost model for binary classification
    model = CatBoostClassifier(
        iterations=1000,       # Number of trees
        depth=9,               # Depth of the tree
        learning_rate=0.05,    # Learning rate
        loss_function='Logloss',  # Loss function for binary classification
        cat_features=[],       # Specify categorical feature indices if needed
        verbose=100            # Print progress every 100 iterations
    )

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the validation set
    y_pred_val = model.predict(X_val)

    # Evaluate the model
    accuracy = accuracy_score(y_val, y_pred_val)
    cm = confusion_matrix(y_val, y_pred_val)

    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Optional: Save the model
    model.save_model('catboost_model.cbm')

    # Optional: Predict probabilities (if needed)
    y_pred_prob = model.predict_proba(X_val)[:, 1]  # Probability for the positive class
    
    return y_pred_val

    #NOTE: Step 1
    # base_path = "datathon2025_2"  # Change this to your actual path
    # df = load_clients_data(base_path)


