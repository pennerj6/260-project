import requests
import json
import gzip
from io import BytesIO
import pandas as pd
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from .env file
load_dotenv()

# # GitHub API settings (for additional API calls if needed)
# GITHUB_API_URL = "https://api.github.com"
# GITHUB_TOKENS = os.getenv("GITHUB_ACCESS_TOKENS").split(",")  # Load tokens from .env
# token_pool = cycle(GITHUB_TOKENS)  # Create a token pool for rotation
# headers = {
#     "Accept": "application/vnd.github.v3+json"
# }

# def get_next_token():
#     """Get the next token from the pool."""
#     return next(token_pool)
'''
def fetch_gharchive_data(date, hour):
    """Fetch and process GHArchive data for a specific date and hour."""
    # Construct the URL for the GH Archive file
    url = f"https://data.gharchive.org/{date}-{hour}.json.gz"
    
    # Send a GET request to download the file
    response = requests.get(url, stream=True)
    
    # Check if the request was successful
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data. HTTP Status Code: {response.status_code}")
    
    # Decompress the gzipped content and process each event
    with gzip.GzipFile(fileobj=BytesIO(response.content)) as gz_file:
        for line in gz_file:
            event = json.loads(line)
            if event["type"] in ["IssueCommentEvent", "PushEvent"]:  # Filter relevant events
                yield event
'''

def fetch_gharchive_data(date, hour):
    url = f"https://data.gharchive.org/{date}-{hour}.json.gz"
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data. HTTP Status Code: {response.status_code}")
    
    with gzip.GzipFile(fileobj=BytesIO(response.content)) as gz_file:
        with ThreadPoolExecutor() as executor:
            events = list(executor.map(json.loads, gz_file))
            return [event for event in events if event["type"] in ["IssueCommentEvent", "PushEvent"]]
        
def parse_gharchive_data(date, hour):
    """Parse GHArchive data for a specific date and hour into DataFrames."""
    issue_comments = []
    commits = []
    
    for event in fetch_gharchive_data(date, hour):
        # Extract issue comments
        if event["type"] == "IssueCommentEvent":
            issue_comments.append({
                "repo": event["repo"]["name"],
                "issue_number": event["payload"]["issue"]["number"],
                "comment_body": event["payload"]["comment"]["body"],
                "comment_author": event["actor"]["login"],
                "created_at": event["payload"]["comment"]["created_at"],
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
    
    return pd.DataFrame(issue_comments), pd.DataFrame(commits)

def save_data(issues_data, commits_data):
    """Save data to CSV files."""
    if issues_data is not None:
        issues_data.to_csv("issue_comments.csv", index=False)
    if commits_data is not None:
        commits_data.to_csv("commits.csv", index=False)

if __name__ == "__main__":
    # Example usage
    date = "2023-10-01"  # Replace with your desired date
    hour = "15"          # Replace with your desired hour (0-23)

    print(f"Fetching and parsing GHArchive data for {date}-{hour}...")
    issue_comments_df, commits_df = parse_gharchive_data(date, hour)
    save_data(issue_comments_df, commits_df)
    print("Data saved to issue_comments.csv and commits.csv.")