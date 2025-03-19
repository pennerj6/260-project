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
        print(f"Success to fetch: {url}")
        return response.json()
    else:
        print(f"Failed to fetch {url}: {response.status_code}")
        return None

def fetch_issue_details(issue_url):
    issue_data = get_github_data(issue_url)
    if not issue_data:
        return None, None
    
    created_at = issue_data.get('created_at', None)
    repository_url = issue_data.get('repository_url', None)
    return created_at, repository_url

def extract_repo_info(url):
    parsed_url = urlparse(url)
    path_parts = parsed_url.path.strip("/").split("/")
    if len(path_parts) >= 4:
        return path_parts[0], path_parts[1], path_parts[3]
    return None, None, None

def get_daily_commits_info(owner, repo, toxic_date, delta_days=3):

    window_start = toxic_date - timedelta(days=delta_days)
    window_end = toxic_date + timedelta(days=delta_days)
    commits_url = (
        f"https://api.github.com/repos/{owner}/{repo}/commits?"
        f"since={window_start.isoformat()}Z&until={window_end.isoformat()}Z"
    )
    commits_data = get_github_data(commits_url)
    
    daily_stats = {}
    for offset in range(-delta_days, delta_days + 1):
        key = f"day{offset}"
        daily_stats[key] = {"commits": 0, "contributors": set(), "lines_of_code": 0}
    
    if not commits_data:
        for key in daily_stats:
            daily_stats[key]["contributors"] = 0
        return daily_stats

    for commit in commits_data:
        commit_time_str = commit["commit"]["author"]["date"]
        commit_time = pd.to_datetime(commit_time_str)
        offset_days = (commit_time.date() - toxic_date.date()).days
        if offset_days < -delta_days or offset_days > delta_days:
            continue
        key = f"day{offset_days}"
        daily_stats[key]["commits"] += 1
        if commit.get("author") and commit["author"].get("login"):
            daily_stats[key]["contributors"].add(commit["author"]["login"])
        
        commit_url = commit["url"]
        commit_data = get_github_data(commit_url)
        if commit_data and "stats" in commit_data:
            daily_stats[key]["lines_of_code"] += commit_data["stats"].get("total", 0)
    
    # contributors account -> number
    for key in daily_stats:
        daily_stats[key]["contributors"] = len(daily_stats[key]["contributors"])
    
    return daily_stats
def get_release_dates(owner, repo):
    releases_url = f"https://api.github.com/repos/{owner}/{repo}/releases"
    releases_data = get_github_data(releases_url)
    if not releases_data:
        return []
    release_dates = [release['published_at'] for release in releases_data if 'published_at' in release]
    return release_dates

def collect_data(row, delta_days):
    issue_id = row["issue_id"]
    issue_url = row["url"]

    created_at, repository_url = fetch_issue_details(issue_url)
    if not created_at or not repository_url:
        return None
    toxic_date = pd.to_datetime(created_at)
    owner, repo, issue_number = extract_repo_info(issue_url)
    if not owner or not repo:
        return None

    print(f"Processing: {owner}/{repo} - Issue {issue_number}")

    daily_stats = get_daily_commits_info(owner, repo, toxic_date, delta_days)

    release_dates = get_release_dates(owner, repo)


    output = {
        "repo_url": repository_url,
        "issue_url": issue_url,
        "toxic_date": toxic_date,
    }
    for offset in range(-delta_days, delta_days + 1):
        key = f"day{offset}"
        output[f"{key}_commits"] = daily_stats[key]["commits"]
        output[f"{key}_contributors"] = daily_stats[key]["contributors"]
        output[f"{key}_lines_of_code"] = daily_stats[key]["lines_of_code"]
    output["release_dates"] = release_dates
    return output



def main():
    # https://github.com/vcu-swim-lab/incivility-dataset/blob/main/dataset/issue_threads.csv
    issue_threads_df = pd.read_csv('appendix/data/issue_threads.csv')
    # https://github.com/vcu-swim-lab/incivility-dataset/blob/main/dataset/annotated_issue_level.csv
    annotated_issue_level_df = pd.read_csv('appendix/data/annotated_issue_level.csv')

    merged_df = pd.merge(issue_threads_df, annotated_issue_level_df, on='issue_id', suffixes=('_thread', '_toxic'))
    
    # toxic comment +- 3days
    time_delta = timedelta(days=3)
    
    analysis_results = []
    for _, row in merged_df.iterrows():
        result = collect_data(row, time_delta)
        if result:
            analysis_results.append(result)
    
    # save
    analysis_df = pd.DataFrame(analysis_results)
    csv_filename = "appendix/data/productivityData_aroundToxicComment.csv"
    analysis_df.to_csv(csv_filename, index=False)
    print(f"Data extracted: {csv_filename}")

if __name__ == "__main__":
    main()
