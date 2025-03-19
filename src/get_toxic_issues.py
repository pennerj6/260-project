from dotenv import load_dotenv
import os
import pandas as pd
import requests
import time
import pickle

if not 'GITHUB_ACCESS_TOKEN' in os.environ:
    load_dotenv()

GITHUB_ACCESS_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")
HEADERS = {
    "Authorization": f"token {GITHUB_ACCESS_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

def get_toxic_issues():
    data_file = '../list_of_repos_to_analyze/incivility_dataset.csv'
    pickle_file = '../data/toxic_issues.pkl'
    issues_list = pd.read_csv(data_file)
    issues = []

    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)
    
    for idx, url in enumerate(issues_list['url']):
        max_attempts = 5
        attempt = 0
        
        while attempt < max_attempts:
            response = requests.get(url, headers=HEADERS)
            
            if response.status_code == 200:
                issues.append(response.json())
                print(f"Found {idx+1} of {len(issues_list)} issues")
                break
            elif response.status_code == 403:  # Rate limit exceeded
                attempt += 1
                remaining_attempts = max_attempts - attempt
                
                if attempt < max_attempts:
                    print(f"GitHub rate limit exceeded, retrying in 1 second ({remaining_attempts} attempts remaining)")
                    time.sleep(1)
                else:
                    print(f"Failed to retrieve issue after {max_attempts} attempts: {url}")
            else:
                print(f"Error retrieving issue, status code: {response.status_code}, URL: {url}")
                break
    
    with open(pickle_file, 'wb') as f:
        pickle.dump(issues, f)
    return issues

if __name__ == "__main__":
    issues = get_toxic_issues()
    print(issues)