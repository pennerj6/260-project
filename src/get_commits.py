from dotenv import load_dotenv
from datetime import datetime, timedelta
import os
import urllib.parse
import requests
import time
import pickle
import numpy as np
from scipy.stats import spearmanr

if not 'GITHUB_ACCESS_TOKEN' in os.environ:
    load_dotenv()

GITHUB_ACCESS_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")
HEADERS = {
    "Authorization": f"token {GITHUB_ACCESS_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

def get_commits_for_issues(issues, num_days):
    pickle_file = '../data/commits.pkl'
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)
    commits = {}
    count = 0
    for issue in issues:
        count += 1
        repo_path = urllib.parse.urlparse(issue['repository_url']).path
        owner = os.path.basename(os.path.dirname(repo_path))
        repo = os.path.basename(repo_path)
        last_updated_date = datetime.fromisoformat(issue['updated_at'])
        start_date = last_updated_date - timedelta(days=num_days)
        end_date = last_updated_date + timedelta(days=num_days)
        print(f'Fetching commits from {owner}/{repo} between {start_date.isoformat()} and {end_date.isoformat()} ({count} of {len(issues)} done)')
        commits[last_updated_date] = get_commits(owner, repo, start_date, end_date)
    with open(pickle_file, 'wb') as f:
        pickle.dump(commits, f)
    return commits

def get_commits(owner, repo, start_timestamp, end_timestamp):
    max_attempts = 5
    attempt = 0
    page = 0
    page_size = 100

    url = f'https://api.github.com/repos/{owner}/{repo}/commits'
    params={
        'since': start_timestamp.isoformat(),
        'until': end_timestamp.isoformat(),
        'per_page': page_size
    }

    commits = []
    
    while True:
        params['page'] = page
        response = requests.get(url, params=params, headers=HEADERS)
        
        if response.status_code == 200:
            data = response.json()
            commits.extend(data)
            if len(data) < page_size:
                break
            attempt = 0
            page += 1
        elif response.status_code == 403:  # Rate limit exceeded
            attempt += 1
            remaining_attempts = max_attempts - attempt
            
            if attempt < max_attempts:
                print(f"GitHub rate limit exceeded, retrying in 1 second ({remaining_attempts} attempts remaining)")
                time.sleep(1)
            else:
                print(f"Failed to retrieve commits after {max_attempts} attempts: {url}")
        else:
            print(f"Error retrieving commits, status code: {response.status_code}, URL: {url}")
            break
    return commits

def to_days(date):
    return date.timestamp() / (60 * 60 * 24)

if __name__ == "__main__":
    from get_toxic_issues import get_toxic_issues
    import matplotlib.pyplot as plt
    from datetime import datetime

    num_days=180

    issues = get_toxic_issues()
    issue_commits = get_commits_for_issues(issues, num_days=num_days)
    
    issue_timestamps = {}
    
    for central_date, commits in issue_commits.items():
        if not central_date in issue_timestamps.keys():
            issue_timestamps[central_date] = []
        for commit in commits:
            commit_date = datetime.fromisoformat(commit['commit']['author']['date'])
            relative_date = to_days(commit_date) - to_days(central_date)
            if relative_date >= -num_days and relative_date <= num_days:
                issue_timestamps[central_date].append(relative_date)
    
    weighted_hist = np.array([])
    weighted_bins = []
    
    # Normalize contribution from each repo
    for central_date, timestamps in issue_timestamps.items():
        if not timestamps:
            continue
        
        hist, bins = np.histogram(timestamps, bins=100, range=(-num_days, num_days))
        hist = hist.astype(float)
        hist /= np.sum(hist)
        if len(weighted_hist) == 0:
            weighted_hist = hist
            weighted_bins = bins
        else:
            weighted_hist += hist
    
    plt.figure(figsize=(12, 6))
    
    corr, p_value = spearmanr(weighted_bins[:-1], weighted_hist)
    print(f'\nCorrelation: {corr:.3f} (p={p_value:.3e})')
    weighted_hist /= np.sum(weighted_hist)
    plt.hist(weighted_bins[:-1], weighted_bins, weights=weighted_hist)
    plt.title('Commit density around issue lock date')
    plt.xlabel('Days relative to issue lock')
    plt.ylabel('Normalized commit density')
    
    plt.tight_layout()
    plt.savefig('commits_vs_locked_date.png')
    plt.show()
