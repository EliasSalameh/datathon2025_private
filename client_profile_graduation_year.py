import os
import json
from collections import defaultdict

# Path to client data
CLIENT_DATA_PATH = "NewRepo/datathon2025_2"

def check_graduation_consistency(profile):
    """Check if graduation years are logically consistent."""
    issues = []
    
    # Extract birth year
    birth_year = None
    if "birth_date" in profile and profile["birth_date"]:
        try:
            birth_year = int(profile["birth_date"].split("-")[0])
        except:
            pass
    
    # Get secondary school graduation year
    secondary_year = None
    if "secondary_school" in profile and isinstance(profile["secondary_school"], dict):
        if "graduation_year" in profile["secondary_school"]:
            secondary_year = profile["secondary_school"]["graduation_year"]
            
            # Check against birth year
            if birth_year:
                if secondary_year - birth_year < 17:
                    issues.append("secondary_school_too_young")
                if secondary_year - birth_year > 20:
                    issues.append("secondary_school_too_old")
    
    # Get earliest higher education year
    min_higher_ed_year = None
    if "higher_education" in profile and isinstance(profile["higher_education"], list):
        for education in profile["higher_education"]:
            if isinstance(education, dict) and "graduation_year" in education:
                year = education["graduation_year"]
                if min_higher_ed_year is None or year < min_higher_ed_year:
                    min_higher_ed_year = year
    
    # Check higher education consistency
    if min_higher_ed_year:
        if secondary_year and min_higher_ed_year < secondary_year:
            issues.append("higher_education_before_secondary")
        if birth_year and min_higher_ed_year - birth_year < 18:
            issues.append("higher_education_too_young")
    
    return issues

def main():
    # Statistics counters
    issue_stats = defaultdict(lambda: {'total': 0, 'accept': 0, 'reject': 0})
    total_clients = 0
    accepted_clients = 0
    rejected_clients = 0
    error_clients = 0
    clients_with_issues = 0
    
    # Get client folders
    client_folders = [f for f in os.listdir(CLIENT_DATA_PATH) 
                     if f.startswith("client_") and os.path.isdir(os.path.join(CLIENT_DATA_PATH, f))]
    
    # Process each client
    for client_folder in client_folders:
        total_clients += 1
        client_path = os.path.join(CLIENT_DATA_PATH, client_folder)
        
        # Load client data directly
        try:
            with open(os.path.join(client_path, 'client_profile.json'), 'r') as f:
                profile = json.load(f)
            with open(os.path.join(client_path, 'label.json'), 'r') as f:
                label_data = json.load(f)
        except Exception:
            error_clients += 1
            continue
        
        # Get decision
        decision = label_data.get('label', 'Unknown')
        if decision == "Accept":
            accepted_clients += 1
        elif decision == "Reject":
            rejected_clients += 1
        
        # Check for graduation year issues
        issues = check_graduation_consistency(profile)
        
        # Update statistics
        if issues:
            clients_with_issues += 1
            for issue in issues:
                issue_stats[issue]['total'] += 1
                if decision == "Accept":
                    issue_stats[issue]['accept'] += 1
                elif decision == "Reject":
                    issue_stats[issue]['reject'] += 1
    
    # Display results
    print("Total clients: {} (Accepted: {}, Rejected: {}, Errors: {})".format(
        total_clients, accepted_clients, rejected_clients, error_clients))
    print("Clients with graduation year issues: {} ({:.2f}%)".format(
        clients_with_issues, (float(clients_with_issues) / total_clients) * 100 if total_clients > 0 else 0))
    
    print("\nGraduation Year Issue Statistics:")
    print("-" * 80)
    print("{:<35} | {:<10} | {:<10} | {:<10}".format(
        "Issue Type", "Total", "% Accept", "% Reject"))
    print("-" * 80)
    
    # Sort and display issues
    sorted_issues = sorted(issue_stats.items(), key=lambda x: x[1]['total'], reverse=True)
    for issue, counts in sorted_issues:
        total = counts['total']
        percent_accept = (float(counts['accept']) / total) * 100 if total > 0 else 0
        percent_reject = (float(counts['reject']) / total) * 100 if total > 0 else 0
        
        print("{:<35} | {:<10} | {:<10.2f} | {:<10.2f}".format(
            issue, total, percent_accept, percent_reject))

if __name__ == "__main__":
    main() 