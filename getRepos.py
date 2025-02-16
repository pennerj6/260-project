import os
import csv
import requests
from datetime import datetime
from dotenv import load_dotenv
from toxicityrater import ToxicityRater
import random

""" get repos which meet the conditions -> data/ github_repos.csv

# TODO: change conditions
1. search repos with condition1 : e.g. star>10, created:>=2020-01-01, folks, open_issues
2. filter repos with condition2 : e.g. issues, pr, contributors

"""
load_dotenv()

# TODO: Create your token https://github.com/settings/tokens and set in .env (copy ".env copy")
# Remember to create a .gitignore file with ".env" before you commit
GITHUB_ACCESS_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")
HEADERS = {
    "Authorization": f"token {GITHUB_ACCESS_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# ref:  https://docs.github.com/en/rest/search/search?apiVersion=2022-11-28#search-repositories
GITHUB_API_URL = "https://api.github.com/search/repositories"

tr = ToxicityRater()



def search_repositories():
    # TODO: Condition 1
    params = {
        "q": "stars:>6 created:>=2020-01-01",
        "sort": "stars",
        "order": "desc",
        "per_page": 20 # number of repositories
    }

    response = requests.get(GITHUB_API_URL, params=params)

    if response.status_code == 200:
        data = response.json()
        print("Repositories found:", len(data.get("items", [])))  # Debugging print

        return data.get("items", [])
    else:
        print("API Error: {response.status_code}, {response.text}")
        return []
    
# Fetch all issues and pull requests from a repository, open or closed
def get_issues(owner, repo_name):
    issues = []
    prs = []
    page = 1
    while True:
        url = f'https://api.github.com/repos/{owner}/{repo_name}/issues?state=all&page={page}&per_page=100'
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            print(f"Error fetching issues: {response.status_code}")
            break
        data = response.json()
        if not data:
            break
        issues.extend(data)
        page += 1

    return issues

# Fetch all comments for a specific issue or PR
def get_comments(base_url):
    comments = []
    page = 1
    while True:
        url = f'{base_url}?page={page}&per_page=100'
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            print(f"Error fetching comments for issue/pull request with URL {base_url}: {response.status_code}")
            break
        data = response.json()
        if not data:
            break
        comments.extend(data)
        page += 1
    return comments

# Fetch all comments for all issues or PRs
def get_all_comments(issues_list):
    comments = []
    for issue in issues_list:
        comments.extend(get_comments(issue['comments_url']))
    return comments
        
# Given a list of comments, return their mean toxicity ranking using the Perspective API
def analyze_toxicity(comments):
    scores = []
    max_score = 0.
    for comment in comments:
        score = tr.get_toxicity_rating(comment['body'], language='en' )
        scores.append(score)
        max_score = max(max_score, score)

    if scores:
        return sum(scores) / len(scores), max_score
    else:
        # suspiciously low toxicity
        print("!!!Possible Issue: 0% toxicity")
        return 0
#
def filter_repositories(repositories):
    for repo in repositories:
        owner, name = repo["full_name"].split("/")
        contributors_count = get_count("contributors",owner, name)
        details = get_repo_details(owner, name)
        
        issues = get_issues(owner, name)
        issues_count = len(issues)

        sample_size = 50
        
        # TODO: Condition 2
        # if contributors_count >= 5 and issues_count>=10 and 90 <= activity_duration.days < 180:
        if len(issues) > sample_size:
            issues = random.sample(issues, 50)
            issue_comments = get_all_comments(issues)
        
            if details is None:
                continue

            activity_duration = details["last_update"] - details["created_at"]

            # TODO: something like if (activity_duration.days >= 90) and (activity_duration.days < 180):




            mean_issue_toxicity, max_issue_toxicity = analyze_toxicity(issue_comments)
            print(f"Matched: {details['repo_name']} (Issues: {issues_count}, Contributors: {contributors_count})")

            yield {
                "repo_name": details["repo_name"],
                "repo_url": details["repo_url"],
                "created_at": details["created_at"],
                "last_update": details["last_update"],
                "issues_count": issues_count,
                "contributors_count": contributors_count,

                "mean_issue_toxicity": mean_issue_toxicity,
                "max_issue_toxicity": max_issue_toxicity,
            }


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

def get_count(field, owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/{field}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        return 0
    
    # Get count of Issues
    if field == "issues":
        return len([issue for issue in response.json() if "pull_request" not in issue]) 
    
    # Get count of Contributors or Pulls
    elif field in ["contributors", "pulls"]:
        return len(response.json()) 
        
    else:
        print("!!!TYPO in get_count(field, ....)")
# For saving data ====================================

def save_to_csv(filtered_repos, overwrite=False):
    csv_filename = "./data/github_data.csv"
    cnt = 0

    # Clear file if overwriting
    if overwrite:
        with open(csv_filename, mode="w", encoding="utf-8", newline="") as file:
            pass
    # Write one repo at a time so some data is saved on failure
    for repo in filtered_repos:
        with open(csv_filename, mode="a", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            if cnt == 0:
                writer.writerow(repo.keys())
            cnt += 1
            writer.writerow(repo.values())

    # with open(csv_filename, mode="w", encoding="utf-8", newline="") as file:
    #     writer = csv.writer(file)
    #     if filtered_repos:
    
    #         # Directly write each key/value into CSV
    #         for repo in filtered_repos:
    #             if cnt == 0:
    #                 writer.writerow(repo.keys())
    #             cnt += 1
    #             writer.writerow(repo.values())
    #     else:
    #         print("!!!No Filtered Repos")
        
    #         # This was the old write process: (To be DELETED after unless we need to manually rename columns)
    #         if 1 == 0:
    #             writer.writerow(["Repository name", "Repository URL", "Created At", "Last Update Date", "Issue Count", "Contributor Count"])

    #             for repo in filtered_repos:
    #                 writer.writerow([
    #                     repo["repo_name"], repo["repo_url"], repo["created_at"], repo["last_update"],
    #                     repo["issues_count"], repo["contributors_count"]
    #                 ])

    print(f"found {cnt} repositories -> saved to {csv_filename}")

def main():
    repositories = search_repositories()
    filtered_repos = filter_repositories(repositories)
    save_to_csv(filtered_repos, overwrite=True)


if __name__ == "__main__":
    main()
    
