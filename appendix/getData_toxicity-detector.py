import os
import time
import pandas as pd
import requests
from datetime import timedelta
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()
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
    
# toxic date
def fetch_issue_details(issue_url):
    issue_data = get_github_data(issue_url)
    if not issue_data:
        return None, None
    created_at = issue_data.get('created_at', None)
    repository_url = issue_data.get('repository_url', None)
    return created_at, repository_url

def construct_issue_url(row):
    _id = row["_id"]
    parts = _id.split("/")
    if len(parts) < 3:
        return None
    issue_number = parts[-1]
    owner = row["owner"]
    repo = row["repo"]
    return f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"

# contributors, commits, lines of code
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
    

    for key in daily_stats:
        daily_stats[key]["contributors"] = len(daily_stats[key]["contributors"])
    
    return daily_stats

def get_release_dates(owner, repo):

    releases_url = f"https://api.github.com/repos/{owner}/{repo}/releases"
    releases_data = get_github_data(releases_url)
    if not releases_data:
        return []
    return [release["published_at"] for release in releases_data if "published_at" in release]


def collect_data(row, delta_days):

    issue_url = construct_issue_url(row)
    if not issue_url:
        return None

    issue_id = row["_id"]
    created_at, repository_url = fetch_issue_details(issue_url)
    if not created_at or not repository_url:
        return None

    toxic_date = pd.to_datetime(created_at)
    owner = row["owner"]
    repo = row["repo"]

    print(f"Processing: {owner}/{repo} - Issue ID {issue_id.split('/')[-1]}")

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
    # https://github.com/CMUSTRUDEL/toxicity-detector/blob/master/data/training/labeled_test_issues.csv
    issues_df = pd.read_csv("appendix/data/labeled_test_issues.csv")
    issues_df = issues_df[issues_df["toxicity"] == "y"]

    # toxic issue +- 3days
    delta_days = 3 
    analysis_results = []
    for _, row in issues_df.iterrows():
        result = collect_data(row, delta_days)
        if result:
            analysis_results.append(result)
    
    analysis_df = pd.DataFrame(analysis_results)
    output_csv = "data/productivityData.csv"
    analysis_df.to_csv(output_csv, index=False)
    print(f"Data extracted: {output_csv}")

if __name__ == "__main__":
    main()
