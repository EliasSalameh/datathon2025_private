import os
import json
# !pip install pandas
# !pip install scikit-learn
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, cv
from catboost import Pool
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.ensemble import RandomForestClassifier




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

def load_clients_data(base_path, llm_output_path, cur_summary_type):
    client_rows = []
    clients_dir = Path(base_path) 
    clients_dir_llm = Path(llm_output_path)
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
        
        llm_client_folder = clients_dir_llm / client_folder.name
        family_background_path = os.path.join(llm_client_folder, 'family_background.json')
        if os.path.exists(family_background_path):
            with open(family_background_path, 'r', encoding='utf-8') as f:
                try:
                    family_data = json.load(f)
                    flat_family_data = flatten_json(family_data)
                    client_data.update(flat_family_data)  # Add the family background data to client_data
                except json.JSONDecodeError:
                    print(f"Warning: could not decode {family_background_path}")
        

        # Normalize label
        label = client_data.get('label', '').lower()
        client_data['label'] = 1 if label == 'accept' else 0
        
        # Extract the number of children from the family data and add it to the features
        client_data['number_of_children'] = family_data.get('number_of_children', 0)  # Default to 0 if not present
        
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
    
    def compute_gap_years(employment_history):
        if isinstance(employment_history, list) and len(employment_history) > 1:
            gap_years = 0
            # Iterate through consecutive pairs of jobs
            for prev_job, job in zip([None] + employment_history[:-1], employment_history):
                if prev_job is not None and prev_job.get('end_year') is not None and job.get('start_year') is not None:
                    gap_years += (job["start_year"] - prev_job["end_year"])
            return gap_years
        return 0  # Return 0 if there 

    filtered_df__["total_gap_years"] = filtered_df__["employment_history"].apply(compute_gap_years)

    
    filtered_df__["num_jobs"] = filtered_df__["employment_history"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    filtered_df__["num_jobs_squared"] = filtered_df__["num_jobs"] ** 2
    
    filtered_df__["longest_job_duration"] = filtered_df__["employment_history"].apply(
        lambda x: max([(current_year - job["start_year"]) if job["end_year"] is None else (job["end_year"] - job["start_year"]) for job in x], default=0)
        if isinstance(x, list) else 0
    )
    filtered_df_['longest_job_duration_sin'] = np.sin(2 * np.pi * filtered_df_['longest_job_duration'] / 36)
    
    filtered_df_['longest_job_duration_cos'] = np.cos(2 * np.pi * filtered_df_['longest_job_duration'] / 36)
    
    filtered_df__["shortest_job_duration"] = filtered_df__["employment_history"].apply(
        lambda x: min([(current_year - job["start_year"]) if job["end_year"] is None else (job["end_year"] - job["start_year"]) for job in x], default=0)
        if isinstance(x, list) else 0
    )
    
    filtered_df__["maximum_salary"] = filtered_df__["employment_history"].apply(
        lambda x: max([job["salary"] for job in x if job["salary"] is not None], default=0) if isinstance(x, list) else 0
    )
    filtered_df__["minimum_salary"] = filtered_df__["employment_history"].apply(
        lambda x: min([job["salary"] for job in x if job["salary"] is not None], default=0) if isinstance(x, list) else 0
    )
    
    filtered_df__["average_salary"] = np.log(filtered_df__["employment_history"].apply(
        lambda x: sum([job["salary"] for job in x if job["salary"] is not None]) / len(x) if isinstance(x, list) and len(x) > 0 else None
    ) + 1)
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
    filtered_df__["total_property_value"] = np.log(filtered_df__["real_estate_details"].apply(
        lambda x: sum(property.get('property value', 0) for property in x) if isinstance(x, list) else 0
    ) + 1)

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

def get_X_y(filtered_df__, save_x_y=False):
    # Ensure the 'label' column is present
    if "label" not in filtered_df__.columns:
        raise ValueError("The DataFrame must contain a 'label' column.")

        # Split into features and target variable
    X = filtered_df__.drop(columns=["label"])  # Drop the 'label' column for features
    y = filtered_df__["label"]  # The 'label' column will be the target variable
    
    if save_x_y:
        X.to_csv("X_no_dummies.csv", index=False)
        # y.to_csv("y_no_dummies.csv", index=False)
        
    return X, y
    
def get_X_y_split(X, y, test_ratio=0.2):
    """
    Splits the DataFrame into features (X) and target variable (y).
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_ratio, random_state=42)
    return X_train, X_val, y_train, y_val

def GPClassifier_train(X_train, y_train):
    # kernel = RBF(length_scale=1.0)  # You can adjust the kernel parameters as needed
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))

    polykernel = Polynomial(degree=2, coef0=1)
    gp_classifier = GaussianProcessRegressor(kernel=polykernel)

    # Train the model
    gp_classifier.fit(X_train, y_train)
    
    return gp_classifier

def GPClassifier_predict(gp_classifier, X_val, y_val):

    # Make predictions on test data
    mu, std = gp_classifier.predict(X_val, return_std=True)

    y_pred = np.where(mu - (0.1*std) < 0.5, 0, 1)

    # Print results
    print("Test set predictions:", y_pred)
    print("True labels:", y_val)

    # Evaluate the accuracy
    accuracy = np.mean(y_pred == y_val)
    print(f"Accuracy: {accuracy:.2f}")
    
    return y_pred


def CatBoost_CV_extended(X_train, y_train, X_val, y_val, save_model=True):
    # Hyperparameter grid
    param_grid = {
        'learning_rate': [0.01],
        'depth': [3, 4, 5],
        'l2_leaf_reg': [5, 7]
    }
    k_values = [5, 10, 20, 50]
    base_params = {
        'iterations': 3000,
        'loss_function': 'Logloss',
        'verbose': 100,
        'task_type': 'CPU'
        # 'task_type': 'GPU',
        # 'devices': '1',
    }

    # Train full model for initial feature importance
    full_model = CatBoostClassifier(**base_params, learning_rate=0.01, depth=3, l2_leaf_reg=5)
    full_model.fit(X_train, y_train)
    importances = full_model.get_feature_importance(prettified=True)
    sorted_features = importances.sort_values(by='Importances', ascending=False)['Feature Id'].tolist()

    best_score = float('inf')
    best_model = None
    best_config = {}

    for k, (lr, depth, l2) in product(k_values, 
                                      product(param_grid['learning_rate'], 
                                              param_grid['depth'], 
                                              param_grid['l2_leaf_reg'])):
        learning_rate, depth, l2_leaf_reg = lr, depth, l2
        top_k_features = sorted_features[:k]
        X_k = X_train[top_k_features]
        data_pool = Pool(X_k, y_train)

        params = {
            **base_params,
            'learning_rate': learning_rate,
            'depth': depth,
            'l2_leaf_reg': l2_leaf_reg
        }

        scores = cv(
            params=params,
            pool=data_pool,
            fold_count=5,
            shuffle=True,
            partition_random_seed=42,
            stratified=True,
            verbose=100
        )

        mean_logloss = scores['test-Logloss-mean'].min()
        print(f"k={k}, lr={learning_rate}, depth={depth}, l2={l2_leaf_reg} → Logloss={mean_logloss:.5f}")

        if mean_logloss < best_score:
            best_score = mean_logloss
            best_config = {
                'k': k,
                'learning_rate': learning_rate,
                'depth': depth,
                'l2_leaf_reg': l2_leaf_reg
            }

    print(f"\n✅ Best Config: {best_config} with Logloss: {best_score:.5f}")

    if save_model:
        top_k_features = sorted_features[:best_config['k']]
        X_train_k = X_train[top_k_features]
        X_val_k = X_val[top_k_features]

        final_params = {
            **base_params,
            'learning_rate': best_config['learning_rate'],
            'depth': best_config['depth'],
            'l2_leaf_reg': best_config['l2_leaf_reg'],
            'verbose': 100
        }

        final_model = CatBoostClassifier(**final_params)
        final_model.fit(X_train_k, y_train)

        final_model.save_model('catboost_model_best.cbm')
        val_score = final_model.score(X_val_k, y_val)
        print(f"Validation Accuracy with top {best_config['k']} features: {val_score:.5f}")
        return final_model

    return None

def CatBoost_CV_2(X_train, y_train, X_val, y_val, save_model=True):
    # Define the base model parameters
    params = {
        'iterations': 5000,
        'learning_rate': 0.025,
        'depth': 2,
        'l2_leaf_reg': 10,
        'loss_function': 'Logloss',
        'verbose': 100,
        'class_weights' : [1.3, 1],  # Adjust class weights if needed
    }

    # Train initial model on full feature set to get feature importance
    full_model = CatBoostClassifier(**params)
    full_model.fit(X_train, y_train)

    # Get sorted top features by importance
    importances = full_model.get_feature_importance(prettified=True)
    sorted_features = importances.sort_values(by='Importances', ascending=False)['Feature Id'].tolist()

    # Define different values of k to test
    k_values = [5, 10, 20, 50]
    cv_results = {}

    # Run cross-validation for each value of k
    for k in k_values:
        top_k_features = sorted_features[:k]
        X_k = X_train[top_k_features]
        data_pool = Pool(X_k, y_train)
        
        scores = cv(
            params=params,
            pool=data_pool,
            fold_count=5,
            shuffle=True,
            partition_random_seed=42,
            stratified=True,
            verbose=100
        )
        
        # Store the best score (lowest logloss)
        cv_results[k] = scores['test-Logloss-mean'].min()

    # Print all scores
    for k, score in cv_results.items():
        print(f"k = {k}, Logloss = {score:.5f}")

    # Select best k
    best_k = min(cv_results, key=cv_results.get)
    print(f"\n✅ Best k: {best_k} with Logloss: {cv_results[best_k]:.5f}")
    if save_model:
        # Train final model with best k features
        top_k_features = sorted_features[:best_k]
        X_train_k = X_train[top_k_features]
        X_val_k = X_val[top_k_features]

        final_model = CatBoostClassifier(**params)
        final_model.fit(X_train_k, y_train)

        # Save the model
        final_model.save_model('catboost_model_topk.cbm')

        # Evaluate on validation set
        val_score = final_model.score(X_val_k, y_val)
        print(f"Validation Score with top {best_k} features: {val_score:.5f}")
    return final_model

def RandomForestClassifier_train(X_train, y_train, save_model=True, k=20):
    
    # Initialize RandomForestClassifier
    rf_params = {
        'n_estimators': 750,
        'max_depth': 3,
        'min_samples_leaf': 5,
        'min_samples_split': 5,
        'max_features': 0.8,
        'random_state': 42,
        'class_weight': "balanced",
        'n_jobs': -1,
        'verbose': 100,
        }
    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(X_train, y_train)
    if k > 0:
        # Select top k features based on feature importance
        rf_model.fit(X_train, y_train)
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1][:k]
        top_k_features = X_train.columns[indices]
        X_train_topk = X_train[top_k_features]
        rf_model.fit(X_train_topk, y_train)
        # if save_model:
        #     rf_model.save_model('rf_model_topk.pkl')
        return rf_model

    # Train the model
    # rf_model.fit(X_train, y_train)

    return rf_model

def RandomForestClassifier_predict(rf_model, X_val, y_val):
    # Make predictions on test data
    # y_pred = rf_model.predict(X_val)
    X_val = X_val[rf_model.feature_names_in_]
    # y_prob = rf_model.predict_proba(X_val)
    y_pred = rf_model.predict(X_val)

    # class_0_prob = y_prob[:, 0]
    # return class_0_prob

    
    # Print results
    print("Test set predictions:", y_pred)
    print("True labels:", y_val)

    # Evaluate the accuracy
    accuracy = np.mean(y_pred == y_val)
    print(f"Accuracy: {accuracy:.2f}")
    
    return y_pred
def CatBoost_CV(X_train, y_train, X_val, y_val, save_model=True, plot_importance=True, k=20):
    train_data = Pool(X_train, label=y_train, cat_features=[])

    # Define the parameter grid for cross-validation
    params = {
        'iterations': 3000,
        'learning_rate': 1e-2,
        'depth': 3,
        'l2_leaf_reg': 5,
        'cat_features': [],
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'random_seed': 42,
        'verbose': 100,
        'class_weights': [1, 1],  # Adjust class weights if needed
    }

    # Perform cross-validation
    cv_results = cv(
        train_data,
        params=params,
        fold_count=5,
        shuffle=True,
        partition_random_seed=42,
        # verbose=True
    )

    # Get the best iteration (you can choose other metrics like AUC, F1, etc.)
    best_iteration = cv_results['iterations'][cv_results['test-Logloss-mean'].idxmin()]

    # Use the best parameters to train the final model
    final_model = CatBoostClassifier(**params)
    

# Set the best iteration explicitly (since the best_iteration was obtained from cross-validation)
    final_model.set_params(iterations=best_iteration)


    # Use the best iteration to train the final model
    # final_model = CatBoostClassifier(iterations=best_iteration, **params)
    final_model.fit(X_train, y_train)
    if k > 0:
        top_k_cols = X_train.columns[final_model.get_feature_importance().argsort()[::-1][:k]]
        X_train_topk = X_train[top_k_cols]
        X_val_topk = X_val[top_k_cols]
        final_model.fit(X_train_topk, y_train)
        print("Validation Score:", final_model.score(X_val_topk, y_val))
        if save_model:
            final_model.save_model('catboost_model_topk.cbm')
            return final_model
        
    if plot_importance:            
        
        feature_importances = final_model.get_feature_importance()

        # Print the feature importances
        for feature, importance in zip(X_train.columns, feature_importances):
            print(f'{feature}: {importance}')
            
        # Evaluate on the validation set
        print("Validation Score:", final_model.score(X_val, y_val))
        
        # Create a DataFrame for easier visualization
        importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': feature_importances
        })

        # Sort the features by importance
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Plot
        plt.figure(figsize=(10, 20))
        plt.barh(importance_df['Feature'], importance_df['Importance'], height= 1.0)
        plt.xlabel('Importance')
        plt.title('CatBoost Feature Importance')

        # Save the figure
        plt.savefig('/home/elias/Anatomic-Diffusion-Models/configs/experiment/autoencoder/feature_importance.png', bbox_inches='tight')  # bbox_inches='tight' ensures the figure is not cropped
        plt.close()  # Close the figure to free up memory
    
    if save_model:
        final_model.save_model('catboost_model.cbm')
    
    return final_model
    
def CatBoost_Train(X_train, y_train, save_model=True):
    # Initialize CatBoost model for binary classification
    model = CatBoostClassifier(
        iterations=5000,       # Number of trees
        depth=9,               # Depth of the tree
        learning_rate=0.05,    # Learning rate
        loss_function='Logloss',  # Loss function for binary classification
        cat_features=[], #['marital_status', 'inheritance_details_profession', 'inheritance_details_relationship', 'investment_risk_profile', 'investment_horizon', 'investment_experience', 'type_of_mandate', 'currency'],       # Specify categorical feature indices if needed
        verbose=100,            # Print progress every 100 iterations
        
    )

    # Train the model
    model.fit(X_train, y_train)

    # Optional: Save the model
    if save_model:
        model.save_model('catboost_model.cbm')

    return model

def CatBoost_Predicitons(model, X_val, y_val):
    # Predict on the validation set
    y_pred_val = model.predict(X_val)
    
    
    # y_prob = model.predict_proba(X_val)

    # # y_prob contains the probability for each class (for binary classification, two columns)
    # # For binary classification, column 1 (y_prob[:, 1]) is the probability of class 1 (positive class)
    # class_1_prob = y_prob[:, 1]
    # class_0_prob = y_prob[:, 0]
    
    # return class_0_prob
    # # Custom decision rule: If (mu - std) < 0.5, predict 0
    # You can compute mu (mean probability) and std (standard deviation) based on some conditions
    # For example, you might want to take the probability from class 1 and apply your custom rule

    # y_pred_val = [0 if prob > 0.21 else 1 for prob in class_0_prob]

    # Evaluate the model
    accuracy = accuracy_score(y_val, y_pred_val)
    cm = confusion_matrix(y_val, y_pred_val)

    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Optional: Save the model
    # model.save_model('catboost_model.cbm')

    # Optional: Predict probabilities (if needed)
    # y_pred_prob = model.predict_proba(X_val)[:, 1]  # Probability for the positive class
    
    return y_pred_val

    #NOTE: Step 1
    # base_path = "datathon2025_2"  # Change this to your actual path
    # df = load_clients_data(base_path)


