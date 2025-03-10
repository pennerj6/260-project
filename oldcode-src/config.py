import os
import csv
import pandas as pd
from dotenv import load_dotenv
import requests
import time
from tqdm import tqdm  # For a progress bar

load_dotenv()

# GitHub API config
GITHUB_ACCESS_TOKEN = os.getenv('GITHUB_ACCESS_TOKEN')
# Headers for GitHub API requests
headers = {
    'Authorization': f'token {GITHUB_ACCESS_TOKEN}',
    'Accept': 'application/vnd.github.v3+json'
}
BASE_URL = "https://api.github.com"

# Variables that we can mess around with
toxicity_threshold = 0.30 #0.05 #0.01  # Threshold for classifying a comment as toxic (0.01 for now for testing)
analysis_window_days = 7  # Days before/after toxic comment to analyze
release_window_days = 14  # Days before release to analyze for toxicity increase



'''
    TA reccommendation from today's Check In (3/6)
    Anayse around 30 to 50 repos to 
    Given our dataset of issue links from various repos, sort it from Issues w highest comment count to lowest
    Idea is higer comments, more toxicity
    The sorted list is sorted_issues.csv
'''
# =========================================================================================================
# Predefined issue URLs to analyze (from the Github DB https://github.com/vcu-swim-lab/incivility-dataset/blob/main/dataset/issue_threads.csv)
# Load the CSV file directly from the URL
url = "https://raw.githubusercontent.com/vcu-swim-lab/incivility-dataset/main/dataset/issue_threads.csv"
data = pd.read_csv(url)
issue_urls = data['url'].tolist()

# issue_urls = issue_urls[0:50] TOOK ME 12 HOURS
# issue_urls = issue_urls[0:5]

# Given the url, return the # of comments in the issue
def get_issue_comment_count(issue_url, headers):
    try:
        response = requests.get(issue_url, headers=headers)
        if response.status_code == 200:
            issue_data = response.json()
            return issue_data.get('comments', 0)
        else:
            print(f"Error fetching {issue_url}: {response.status_code}")
            return 0
    except Exception as e:
        print(f"Exception for {issue_url}: {e}")
        return 0

# First time load of sorted CSV
def sort_issues_by_comments(issue_urls, headers, csv_path="sorted_issues.csv"):
    # Create a list of tuples (url, comment_count)
    issue_data = []
    
    # Add a progress bar (chatgpt did this bar thing and i like it lol)
    for url in tqdm(issue_urls, desc="Fetching comment counts"):
        comment_count = get_issue_comment_count(url, headers)
        issue_data.append((url, comment_count))
        
        # Add a small delay to avoid hitting rate limits
        time.sleep(0.5)  

    # Sort the list by comment count (descending)
    sorted_issues = sorted(issue_data, key=lambda x: x[1], reverse=True)
    
    # Write to CSV
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['url', 'comment_count'])  # Header
        for url, count in sorted_issues:
            writer.writerow([url, count])
    
    print(f"Sorted issues saved to {csv_path}")
    
    return sorted_issues

# Get issue links from sorted local CSV
def load_sorted_issues_from_csv(csv_path="sorted_issues.csv"):
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} not found!")
        return []
    
    sorted_issues = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:
                url, count = row[0], int(row[1])
                sorted_issues.append((url, count))
    
    print(f"Loaded {len(sorted_issues)} sorted issues from {csv_path}")
    return sorted_issues


if os.path.exists("sorted_issues.csv"):
    print("Getting links from the locally stored/sorted file")
    sorted_issues = load_sorted_issues_from_csv()
else:
    sorted_issues = sort_issues_by_comments(issue_urls, headers)

# Get just the URLs in sorted order
issue_urls = [item[0] for item in sorted_issues]

i = input("How many repos? (int or ALL): \n")
try:
    i = int(i) 
    if isinstance(i,int):
        issue_urls = issue_urls[0:i]
except:
    print("all repos being processed")

print(f"Repos Processing: {len(issue_urls)}")
# =========================================================================================================
