import requests
import json
import gzip
from io import BytesIO
import pandas as pd
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor
import logging
import dask.bag as db

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def fetch_gharchive_data(date, hour):
    """Fetch and process GHArchive data for a specific date and hour."""
    url = f"https://data.gharchive.org/{date}-{hour}.json.gz"
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data. HTTP Status Code: {response.status_code}")

    # Decompress the gzip file and process JSON lines
    with gzip.GzipFile(fileobj=BytesIO(response.content)) as gz_file:
        events = []
        for line in gz_file:
            try:
                event = json.loads(line)
                if event["type"] in ["IssueCommentEvent", "PushEvent"]:
                    events.append(event)
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON line: {str(e)}")
        return events
# def fetch_gharchive_data(date, hour):
#     """Fetch and process GHArchive data for a specific date and hour."""
#     url = f"https://data.gharchive.org/{date}-{hour}.json.gz"
#     response = requests.get(url, stream=True)
#     if response.status_code != 200:
#         raise Exception(f"Failed to fetch data. HTTP Status Code: {response.status_code}")

#     with gzip.GzipFile(fileobj=BytesIO(response.content)) as gz_file:
#         with ThreadPoolExecutor() as executor:
#             events = list(executor.map(json.loads, gz_file))
#             return [event for event in events if event["type"] in ["IssueCommentEvent", "PushEvent"]]

def parse_gharchive_data(date, hour):
    """Parse GHArchive data for a specific date and hour into DataFrames."""
    issue_comments = []
    commits = []

    # Repos likely to have more heated discussions
    high_interest_repos = [
        "kubernetes/kubernetes",
        "tensorflow/tensorflow",
        "rust-lang/rust",
        "golang/go",
        "facebook/react",
        "nodejs/node",
        "microsoft/vscode",
        "torvalds/linux",
        "bitcoin/bitcoin",
        "ethereum/go-ethereum",
        "ansible/ansible",
        "dotnet/runtime",
        "angular/angular",
        "python/cpython",
        # "pytorch/pytorch", 
        # "apache/spark",
        # "elastic/elasticsearch",
    ]
    

    try:
        for event in fetch_gharchive_data(date, hour):
            # Extract issue comments
            if event["type"] == "IssueCommentEvent":
                # Look for signals of potential conflict
                comment_body = event["payload"]["comment"]["body"]

                # Check if this is a closed/rejected issue (where conflict might occur)
                issue_state = event["payload"]["issue"]["state"]
                issue_title = event["payload"]["issue"]["title"].lower()

                # Check for keywords that might indicate disagreement
                conflict_keywords = ["wrong", "incorrect", "disagree", "not working", "bug",
                                     "error", "won't fix", "wontfix", "rejected", "closed",
                                     "can't reproduce", "!important", "garbage", "useless",
                                     "terrible", "awful", "never", "stupid", "ridiculous",
                                     "please don't", "stop", "hate", "impossible", "absurd"]

                has_keywords = any(keyword in comment_body.lower() for keyword in conflict_keywords)

                repo_name = event["repo"]["name"]
                is_high_interest = repo_name in high_interest_repos

                # Check for closed PRs which often have higher confrontation
                is_closed_pr = issue_state == "closed" and ("pr" in issue_title or "pull request" in issue_title)

                # Save all comments for analysis, but flag potential high-toxicity ones
                issue_comments.append({
                    "repo": repo_name,
                    "issue_number": event["payload"]["issue"]["number"],
                    "comment_body": comment_body,
                    "comment_author": event["actor"]["login"],
                    "created_at": event["payload"]["comment"]["created_at"],
                    "issue_state": issue_state,
                    "has_conflict_keywords": has_keywords,
                    "high_interest_repo": is_high_interest,
                    "is_closed_pr": is_closed_pr,
                    "comment_length": len(comment_body)
                })

            # Extract commits
            if event["type"] == "PushEvent":
                for commit in event["payload"]["commits"]:
                    commits.append({
                        "repo": event["repo"]["name"],
                        "sha": commit["sha"],
                        "author": commit["author"]["name"],
                        "date": event["created_at"],
                    })
    except Exception as e:
        logging.error(f"Error processing {date}-{hour}: {str(e)}")

    issue_df = pd.DataFrame(issue_comments) if issue_comments else pd.DataFrame()
    commits_df = pd.DataFrame(commits) if commits else pd.DataFrame()
    return issue_df, commits_df

def save_data(issues_data, commits_data):
    """Save data to CSV files."""
    if issues_data is not None and not issues_data.empty:
        issues_data.to_csv("issue_comments.csv", index=False)
    if commits_data is not None and not commits_data.empty:
        commits_data.to_csv("commits.csv", index=False)