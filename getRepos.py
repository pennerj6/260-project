import requests
import csv
from datetime import datetime
import os
from dotenv import load_dotenv

""" get repos which meet the conditions -> data/ github_repos.csv

# TODO: change conditions
1. search repos with condition1 : e.g. star>10, created:>=2020-01-01, folks, open_issues
2. filter repos with condition2 : e.g. issues, pr, contributors

"""
load_dotenv()

# TODO: Create your token https://github.com/settings/tokens and set in .env (copy ".env copy")
GITHUB_ACCESS_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")
HEADERS = {
    "Authorization": f"token {GITHUB_ACCESS_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# ref:  https://docs.github.com/en/rest/search/search?apiVersion=2022-11-28#search-repositories
GITHUB_API_URL = "https://api.github.com/search/repositories"


def main():
    repositories = search_repositories()
    filtered_repos = filter_repositories(repositories)
    save_to_csv(filtered_repos)


def search_repositories():
    # TODO: Condition 1
    params = {
        "q": "stars:>10 created:>=2020-01-01",
        "sort": "stars",
        "order": "desc",
        "per_page": 20 # number of repositories
    }

    response = requests.get(GITHUB_API_URL, params=params)

    if response.status_code == 200:
        data = response.json()
        return data.get("items", [])
    else:
        print("API Error: {response.status_code}, {response.text}")
        return []
    

def filter_repositories(repositories):
    filtered_repos = []
    for repo in repositories:
        owner, name = repo["full_name"].split("/")
        contributors_count = get_contributors_count(owner, name)
        pr_count = get_pull_requests_count(owner, name)
        issues_count = get_issues_count(owner, name)
        details = get_repo_details(owner, name)

        if details is None:
            continue

        activity_duration = details["last_update"] - details["created_at"]

        # TODO: Condition 2
        # if pr_count >= 10 and contributors_count >= 5 and issues_count>=10 and 90 <= activity_duration.days < 180:
        if pr_count >= 5:
            filtered_repos.append({
                "repo_name": details["repo_name"],
                "repo_url": details["repo_url"],
                "created_at": details["created_at"],
                "last_update": details["last_update"],
                "pr_count": pr_count,
                "issues_count": issues_count,
                "contributors_count": contributors_count
            })
            print(f"Matched: {details['repo_name']} (Issues: {issues_count}, Contributors: {contributors_count})")

    return filtered_repos


# For filtering repos ====================================

def get_repo_details(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        repo_data = response.json()

        if "full_name" in repo_data and "html_url" in repo_data:
            created_at = datetime.strptime(repo_data["created_at"], "%Y-%m-%dT%H:%M:%SZ")
            last_update = datetime.strptime(repo_data["updated_at"], "%Y-%m-%dT%H:%M:%SZ")
            return {
                "repo_name": repo_data["full_name"],
                "repo_url": repo_data["html_url"],
                "created_at": created_at,
                "last_update": last_update,
            }
    return None

def get_issues_count(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return len([issue for issue in response.json() if "pull_request" not in issue]) 
    return 0


def get_contributors_count(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/contributors"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return len(response.json()) 
    return 0


def get_pull_requests_count(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return len(response.json())  
    return 0



# For saving data ====================================

def save_to_csv(filtered_repos):
    csv_filename = "./data/github_data.csv"
    cnt = len(filtered_repos)

    with open(csv_filename, mode="w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Repository name", "Repository URL", "Created At", "Last Update Date", "Issue Count", "Contributor Count"])

        for repo in filtered_repos:
            writer.writerow([
                repo["repo_name"], repo["repo_url"], repo["created_at"], repo["last_update"],
                repo["issues_count"], repo["contributors_count"]
            ])

    print(f"found {cnt} repositories -> saved to {csv_filename}")



if __name__ == "__main__":
    main()
