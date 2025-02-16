import os
import csv
import time
import requests
from datetime import datetime
from dotenv import load_dotenv
from toxicityrater import ToxicityRater

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


# Filter issues to be locked and have label "too-heated", "off-topic", or "spam"  (ref: https://github.com/vcu-swim-lab/incivility-dataset)
def get_locked_issues_by_labels(owner, repo, target_labels):
    # Returns a list of issues which are locked and have at least one label in target_labels.
    url = f"https://api.github.com/repos/{owner}/{repo}/issues?state=all&per_page=100" # state=all get both open and closed issues.
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        return []
    
    issues = response.json()
    filtered_issues = []
    for issue in issues:
        # Check if the issue is locked.
        if issue.get("locked", False):
            # Check if any of its labels match our target labels.
            for label in issue.get("labels", []):
                if label.get("name", "").lower() in target_labels:
                    filtered_issues.append(issue)
                    break
    return filtered_issues




# Given a repo, return the comments where comment_type can be "Issues" or "Pulls"
def get_comments(owner, repo, comment_type):
    url = f"https://api.github.com/repos/{owner}/{repo}/{comment_type}/comments"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json() 
    else: 
        return []
        


def search_repositories(page, per_page):
    # TODO: Condition 1
    params = {
        #"q": "stars:>10 created:>=2020-01-01",
        "q": "stars:>10 created:>=2013-04-07",
        "sort": "stars",
        "order": "desc",
        "per_page": per_page, #20 # number of repositories
        "page": page
    }

    response = requests.get(GITHUB_API_URL, params=params)

    if response.status_code == 200:
        data = response.json()
        print("Repositories found:", len(data.get("items", [])))  # Debugging print

        return data.get("items", [])
    else:
        print("API Error: {response.status_code}, {response.text}")
        return []
    

# Given a list of comments, return their mean toxicity ranking using the Perspective API
def analyze_toxicity(metric, comments):
    
    if metric == "mean":
        scores = []
        for comment in comments:
            scores.append(tr.get_toxicity_rating(comment['body'], language='en' ))

            # Sleep 1s to prevent Perspective API request limit
            time.sleep(1)

        if scores:
            return sum(scores) / len(scores) 
        else:
            # suspiciously low toxicity
            print("!!!Possible Issue: 0% toxicity")
            return 0
        
    elif metric == "max":
        scores = []
        for comment in comments:
            scores.append(tr.get_toxicity_rating(comment['body'], language='en' ))

            # Sleep 1s to prevent Perspective API request limit
            time.sleep(1)

        if scores:
            return max(scores)
        else:
            # suspiciously low toxicity
            print("!!!Possible Issue: 0% toxicity")
            return 0
        

    
    
    print("!!!Please include a LISTED metric (ex: mean)")
    return 0

#
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

# For filtering repos ====================================
def filter_repositories(repositories):
    filtered_repos = []

    # Set timeframe timeframe TODO: changge timeframe to present?
    # "from April 7, 2013, to October 24, 2023"  (ref: https://github.com/vcu-swim-lab/incivility-dataset)
    start_time , end_time = (datetime(2013, 4, 7), datetime(2023, 10, 24))
    # Desired issues have at least 1 of the following labels "too-heated", "off-topic", or "spam" (ref: https://github.com/vcu-swim-lab/incivility-dataset)
    target_labels = {"too-heated", "off-topic", "spam"}

    for repo in repositories:
        owner, name = repo["full_name"].split("/")
        contributors_count = get_count("contributors",owner, name)
        pr_count = get_count("pulls", owner, name)
        issues_count = get_count("issues",owner, name)
        details = get_repo_details(owner, name)
        if details is None:
            continue

        issue_comments = get_comments(owner, name, "issues")
        pr_comments = get_comments(owner, name, "pulls")
        


    
        if details is None:
            continue

        # Criteria #1 Make sure repo is within timeframe
        # handled in "params"
        #if (details["created_at"] < start_time) or (details["created_at"] > end_time):
            # Skip, Out of date range 
            #continue
        
        # Criteria #2 Make sure at least 50 contributors 
        if contributors_count < 10: #50:
            # Skip, not enought contributors
            continue

        # Criteria #3 Make sure issue is locked & has target label
        locked_issues = get_locked_issues_by_labels(owner, name, target_labels)
        if len(locked_issues) == 0:
            # Skip, doesnt match critera 3
            continue
        # Critera #4 MORE?
        

        # activity_duration = details["last_update"] - details["created_at"]
        # EX: if 90 <= activity_duration.days < 180:



        # TODO: something like if (activity_duration.days >= 90) and (activity_duration.days < 180):


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
                "contributors_count": contributors_count,

                "locked_issues_count": len(locked_issues),

                "mean_issue_toxicity": analyze_toxicity("mean",issue_comments),
                "mean_pr_toxicity": analyze_toxicity("mean", pr_comments),

                "max_issue_toxicity": analyze_toxicity("max",issue_comments),
                "max_pr_toxicity": analyze_toxicity("max", pr_comments)
            })
            #print(f"Matched: {details['repo_name']} (Issues: {issues_count}, Contributors: {contributors_count})")
            print(f"Matched: {details['repo_name']} (Contributors: {contributors_count}, Locked Issues: {len(locked_issues)})")

    return filtered_repos



# For saving data ====================================

def save_to_csv(filtered_repos):
    csv_filename = "./data/github_data.csv"
    cnt = len(filtered_repos)

    with open(csv_filename, mode="w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        if filtered_repos:
    
            # Directly write each key/value into CSV
            writer.writerow(filtered_repos[0].keys())
            for repo in filtered_repos:
                writer.writerow(repo.values())
        else: 
            print("!!!No Filtered Repos")
        
            # This was the old write process: (To be DELETED after unless we need to manually rename columns)
            if 1 == 0:
                writer.writerow(["Repository name", "Repository URL", "Created At", "Last Update Date", "Issue Count", "Contributor Count"])

                for repo in filtered_repos:
                    writer.writerow([
                        repo["repo_name"], repo["repo_url"], repo["created_at"], repo["last_update"],
                        repo["issues_count"], repo["contributors_count"]
                    ])

    print(f"found {cnt} repositories -> saved to {csv_filename}")

import re

def parse_repo_identifier(repo_str):
    """
    Given a repository string that could be in the form "owner/repo" or a full GitHub URL,
    this function returns the "owner/repo" format.
    """
    # If it looks like a URL, extract the owner and repo using a regex.
    if repo_str.startswith("http"):
        pattern = r"github\.com/([^/]+/[^/]+)"
        match = re.search(pattern, repo_str)
        if match:
            return match.group(1)
        else:
            print(f"Warning: Could not parse repository from URL: {repo_str}")
            return None
    else:
        # Assume it's already in the "owner/repo" format. (probably not)
        return repo_str

def main():
    # Manually chosen repositories.
    my_repos = [
        "https://github.com/avelino/awesome-go",
        "https://github.com/microsoft/terminal",
        "https://github.com/django/django"
    ]

    if my_repos:
        print("Using manually defined repositories...")
        # Convert each repository string to the "owner/repo" format
        repositories = []
        for repo in my_repos:
            repo_id = parse_repo_identifier(repo)
            if repo_id:
                repositories.append({"full_name": repo_id})
        filtered_repos = filter_repositories(repositories)
    else:
        print("Searching for repositories...")
        target_repo_count = 20  # Change as needed
        all_filtered_repos = []
        page = 1
        per_page = 100  # Number of repositories per page

        while len(all_filtered_repos) < target_repo_count:
            repositories = search_repositories(page, per_page)
            if not repositories:
                print("No more repositories found from the search API.")
                break

            filtered = filter_repositories(repositories)
            all_filtered_repos.extend(filtered)
            print(f"Total filtered repos so far: {len(all_filtered_repos)}")

            page += 1
            #time.sleep(2)  # To avoid hitting rate limits

        filtered_repos = all_filtered_repos[:target_repo_count]

    save_to_csv(filtered_repos)

if __name__ == "__main__":
    main()




