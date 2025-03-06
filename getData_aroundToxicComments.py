import pandas as pd
import requests
from datetime import timedelta
from urllib.parse import urlparse
import os
from dotenv import load_dotenv


GITHUB_ACCESS_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")
HEADERS = {
    "Authorization": f"token {GITHUB_ACCESS_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}



def get_github_data(url):
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch {url}: {response.status_code}")
        return None

def extract_repo_info(issue_url):
    parsed_url = urlparse(issue_url)
    path_parts = parsed_url.path.strip("/").split("/")
    if len(path_parts) >= 4:
        owner, repo, _, issue_number = path_parts
        return owner, repo, issue_number
    return None, None, None

def get_commits_info(owner, repo, start_date, end_date):
    commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits?since={start_date.isoformat()}Z&until={end_date.isoformat()}Z"
    commits_data = get_github_data(commits_url)
    if not commits_data:
        return 0, [], 0, 0

    # data type 
    commits = len(commits_data)
    timestamps = [commit['commit']['author']['date'] for commit in commits_data]
    contributors = len(set(commit['author']['login'] for commit in commits_data if commit.get('author')))
    

    lines_of_code = 0
    for commit in commits_data:
        commit_url = commit['url']
        commit_data = get_github_data(commit_url)
        if commit_data and 'stats' in commit_data:
            lines_of_code += commit_data['stats'].get('total', 0)
    
    return commits, timestamps, contributors, lines_of_code

def get_release_dates(owner, repo):
    releases_url = f"https://api.github.com/repos/{owner}/{repo}/releases"
    releases_data = get_github_data(releases_url)
    if not releases_data:
        return []
    release_dates = [release['published_at'] for release in releases_data if 'published_at' in release]
    return release_dates

def process_issue_row(row, time_delta):

    toxic_date = row['created_at_toxic']
    issue_id = row['issue_id']
    issue_url = row['issue_url']
    
    owner, repo, issue_number = extract_repo_info(issue_url)
    if not owner or not repo:
        return None

    print(f"Processing: {owner}/{repo} - Issue {issue_number}")

    start_date = toxic_date - time_delta
    end_date = toxic_date + time_delta

    commits, timestamps, contributors, lines_of_code = get_commits_info(owner, repo, start_date, end_date)
    release_dates = get_release_dates(owner, repo)

    return {
        'issue_id': issue_id,
        'repo': f"{owner}/{repo}",
        'toxic_date': toxic_date,
        'start_date': start_date,
        'end_date': end_date,
        'commits': commits,
        'contributors': contributors,
        'timestamps': timestamps,
        'release_dates': release_dates,
        'lines_of_code': lines_of_code
    }


def main():
    # https://github.com/vcu-swim-lab/incivility-dataset/blob/main/dataset/issue_threads.csv
    issue_threads_df = pd.read_csv('data/issue_threads.csv')
    # https://github.com/vcu-swim-lab/incivility-dataset/blob/main/dataset/annotated_issue_level.csv
    annotated_issue_level_df = pd.read_csv('data/annotated_issue_level.csv')

    issue_threads_df['created_at'] = pd.to_datetime(issue_threads_df['created_at'])
    annotated_issue_level_df['created_at'] = pd.to_datetime(annotated_issue_level_df['created_at'])

    merged_df = pd.merge(issue_threads_df, annotated_issue_level_df, on='issue_id', suffixes=('_thread', '_toxic'))
    
    # toxic comment +- 3days
    time_delta = timedelta(days=3)
    
    analysis_results = []
    for _, row in merged_df.iterrows():
        result = process_issue_row(row, time_delta)
        if result:
            analysis_results.append(result)
    
    # save
    analysis_df = pd.DataFrame(analysis_results)
    csv_filename = "data/productivityData_aroundToxicComment.csv"
    analysis_df.to_csv(csv_filename, index=False)
    print(f"Data extracted: {csv_filename}")

if __name__ == "__main__":
    main()
