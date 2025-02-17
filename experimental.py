import os
import csv
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from toxicityrater import ToxicityRater
import random
from collections import defaultdict
import time

load_dotenv()

GITHUB_ACCESS_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")
HEADERS = {
    "Authorization": f"token {GITHUB_ACCESS_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}
GITHUB_API_URL = "https://api.github.com/search/repositories"
tr = ToxicityRater()

# Search repos w the given params (to be changed?)
def search_repositories():
    params = {
        "q": "stars:>6 created:>=2020-01-01",
        "sort": "stars",
        "order": "desc",
        "per_page": 20
    }

    response = requests.get(GITHUB_API_URL, params=params, headers=HEADERS)

    if response.status_code == 200:
        data = response.json()
        print("Repositories found:", len(data.get("items", [])))
        return data.get("items", [])
    else:
        print(f"API Error: {response.status_code}, {response.text}")
        return []

# Get the issues of a repo
def get_issues(owner, repo_name):
    issues = []
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
        time.sleep(1)  # Rate limiting
    return issues

# For AN issue, return the comments
def get_comments(base_url):
    comments = []
    page = 1
    while True:
        url = f'{base_url}?page={page}&per_page=100'
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            print(f"Error fetching comments: {response.status_code}")
            break
        data = response.json()
        if not data:
            break
        comments.extend(data)
        page += 1
        time.sleep(1)  # Rate limiting
    return comments

# Return ALL comments for EACH issue
def get_all_comments(issues_list):
    comments = []
    for issue in issues_list:
        comments.extend(get_comments(issue['comments_url']))
    return comments


# Calculate the toxicity using Googles Perspective API
# used CHATGPT to help with preventative measures for reaching API limit
def analyze_toxicity(comments):
    scores = []
    max_score = 0.
    requests_this_minute = 0
    last_request_time = datetime.now()
    backoff_time = 1  # Start with 1 second backoff
    max_backoff = 64  # Maximum backoff of 64 seconds
    max_retries = 5   # Maximum number of retries per comment

    for i, comment in enumerate(comments):
        if not comment.get('body'):
            continue

        retry_count = 0
        while retry_count < max_retries:
            try:
                current_time = datetime.now()
                time_diff = (current_time - last_request_time).total_seconds()

                # Reset counter if a minute has passed
                if time_diff >= 60:
                    requests_this_minute = 0
                    last_request_time = current_time
                    backoff_time = 1  # Reset backoff time after successful minute

                # If we're near the rate limit, wait
                if requests_this_minute >= 55:  # Buffer of 5 requests
                    wait_time = 60 - time_diff
                    if wait_time > 0:
                        print(f"Rate limit approaching, waiting {wait_time:.2f} seconds...")
                        time.sleep(wait_time)
                    requests_this_minute = 0
                    last_request_time = datetime.now()

                # Add a small base delay between requests
                time.sleep(0.5)

                # Process comment
                print(f"Processing comment {i+1}/{len(comments)}")
                score = tr.get_toxicity_rating(comment['body'], language='en')
                scores.append(score)
                max_score = max(max_score, score)
                requests_this_minute += 1
                
                # Successful request, break retry loop
                break

            except Exception as e:
                if "429" in str(e):  # Rate limit error
                    print(f"Rate limit exceeded. Backing off for {backoff_time} seconds...")
                    time.sleep(backoff_time)
                    backoff_time = min(backoff_time * 2, max_backoff)  # Exponential backoff
                    retry_count += 1
                    requests_this_minute = 55  # Force a wait on next iteration
                else:
                    print(f"Unexpected error analyzing comment: {str(e)}")
                    break  # Break on non-rate-limit errors

        if retry_count >= max_retries:
            print(f"Maximum retries reached for comment {i+1}. Skipping.")

    if scores:
        mean_score = sum(scores) / len(scores)
        print(f"Successfully analyzed {len(scores)} out of {len(comments)} comments")
        return mean_score, max_score
    else:
        print("No comments were successfully analyzed")
        return 0, 0
    
# Get commit detaiils
def get_commit_metrics(owner, repo_name):
    url = f'https://api.github.com/repos/{owner}/{repo_name}/commits'
    commits = []
    page = 1
    
    while True:
        response = requests.get(f'{url}?page={page}&per_page=100', headers=HEADERS)
        if response.status_code != 200 or not response.json():
            break
        commits.extend(response.json())
        page += 1
        time.sleep(1)  # Rate limiting
    
    # if therese no commits, return 0
    if not commits:
        return {
            'commit_count': 0,
            'commit_frequency': 0,
            'active_contributors_per_month': 0
        }
    
    first_commit_date = datetime.strptime(commits[-1]['commit']['author']['date'], "%Y-%m-%dT%H:%M:%SZ")
    last_commit_date = datetime.strptime(commits[0]['commit']['author']['date'], "%Y-%m-%dT%H:%M:%SZ")
    days_diff = (last_commit_date - first_commit_date).days or 1
    
    monthly_contributors = defaultdict(set)
    for commit in commits:
        date = datetime.strptime(commit['commit']['author']['date'], "%Y-%m-%dT%H:%M:%SZ")
        month_key = f"{date.year}-{date.month}"
        monthly_contributors[month_key].add(commit['commit']['author']['email'])
    
    avg_monthly_contributors = sum(len(contributors) for contributors in monthly_contributors.values()) / len(monthly_contributors) if monthly_contributors else 0
    
    return {
        'commit_count': len(commits),
        'commit_frequency': len(commits) / days_diff,
        'active_contributors_per_month': avg_monthly_contributors
    }

# Get Release Details (TA said that maybe focusing around the toxicity around RELEASES might be better?)
def get_release_metrics(owner, repo_name, toxicity_data):
    url = f'https://api.github.com/repos/{owner}/{repo_name}/releases'
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        return {
            'release_count': 0,
            'avg_pre_release_toxicity': 0,
            'avg_post_release_toxicity': 0
        }
    
    releases = response.json()
    if not releases:
        return {
            'release_count': 0,
            'avg_pre_release_toxicity': 0,
            'avg_post_release_toxicity': 0
        }

    pre_release_toxicity = []
    post_release_toxicity = []
    
    for release in releases:
        release_date = datetime.strptime(release['published_at'], "%Y-%m-%dT%H:%M:%SZ")
        
        for date, toxicity in toxicity_data:
            time_diff = (date - release_date).days
            if -14 <= time_diff < 0:
                pre_release_toxicity.append(toxicity)
            elif 0 <= time_diff < 14:
                post_release_toxicity.append(toxicity)
    
    return {
        'release_count': len(releases),
        'avg_pre_release_toxicity': sum(pre_release_toxicity) / len(pre_release_toxicity) if pre_release_toxicity else 0,
        'avg_post_release_toxicity': sum(post_release_toxicity) / len(post_release_toxicity) if post_release_toxicity else 0
    }

# Return the contributor details
def get_contributor_metrics(owner, repo_name):
    url = f'https://api.github.com/repos/{owner}/{repo_name}/contributors'
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        return {
            'avg_contributor_age_days': 0,
            'avg_contributor_repos': 0,
            'avg_contributor_commits': 0
        }
    
    contributors = response.json()
    if not contributors:
        return {
            'avg_contributor_age_days': 0,
            'avg_contributor_repos': 0,
            'avg_contributor_commits': 0
        }
    
    total_age = 0
    total_repos = 0
    total_commits = 0
    
    for contributor in contributors[:5]:
        user_url = contributor['url']
        user_response = requests.get(user_url, headers=HEADERS)
        if user_response.status_code != 200:
            continue
            
        user_data = user_response.json()
        
        if 'created_at' in user_data:
            account_age = (datetime.now() - datetime.strptime(user_data['created_at'], "%Y-%m-%dT%H:%M:%SZ")).days
            total_age += account_age
            total_repos += user_data.get('public_repos', 0)
            total_commits += contributor['contributions']
        
        time.sleep(1)  # Rate limiting
    
    contributor_count = min(5, len(contributors))
    if contributor_count == 0:
        return {
            'avg_contributor_age_days': 0,
            'avg_contributor_repos': 0,
            'avg_contributor_commits': 0
        }
    
    return {
        'avg_contributor_age_days': total_age / contributor_count,
        'avg_contributor_repos': total_repos / contributor_count,
        'avg_contributor_commits': total_commits / contributor_count
    }

# Return the project size details, including langauge (@Jordan, this might help w your idea of Rust(?) having more toxiciity )
def get_project_size_metrics(owner, repo_name):
    url = f'https://api.github.com/repos/{owner}/{repo_name}/languages'
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        return {
            'lines_of_code': 0,
            'language_count': 0,
            'primary_language': 'Unknown'
        }
    
    languages = response.json()
    
    if not languages:
        return {
            'lines_of_code': 0,
            'language_count': 0,
            'primary_language': 'Unknown'
        }
    
    return {
        'lines_of_code': sum(languages.values()),
        'language_count': len(languages),
        'primary_language': max(languages.items(), key=lambda x: x[1])[0] if languages else 'Unknown'
    }

# Get issue details (for bioth closed && open)
def get_issue_metrics(issues, comments):
    if not issues:
        return {
            'avg_time_to_resolve': 0,
            'comment_count_per_issue': 0,
            'open_issues_percentage': 0,
            'issue_labels_count': 0
        }
    
    total_resolution_time = 0
    resolved_count = 0
    total_comments = len(comments)
    open_issues = 0
    all_labels = set()
    
    for issue in issues:
        if issue['state'] == 'closed' and issue.get('closed_at'):
            created = datetime.strptime(issue['created_at'], "%Y-%m-%dT%H:%M:%SZ")
            closed = datetime.strptime(issue['closed_at'], "%Y-%m-%dT%H:%M:%SZ")
            total_resolution_time += (closed - created).days
            resolved_count += 1
        elif issue['state'] == 'open':
            open_issues += 1
            
        all_labels.update(label['name'] for label in issue['labels'])
    
    return {
        'avg_time_to_resolve': total_resolution_time / resolved_count if resolved_count else 0,
        'comment_count_per_issue': total_comments / len(issues),
        'open_issues_percentage': (open_issues / len(issues)) * 100,
        'issue_labels_count': len(all_labels)
    }

# Get repo detials
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

# Get counts of issues or contributors or PR
def get_count(field, owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/{field}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        return 0
    
    if field == "issues":
        return len([issue for issue in response.json() if "pull_request" not in issue])
    elif field in ["contributors", "pulls"]:
        return len(response.json())
    else:
        print("!!!TYPO in get_count(field, ....)")
        return 0

# Filters repos via our criteria
# CHATGPT to help wit API rate limit, process in batches. (slow, but it doesnt hit the limit..)

def filter_repositories(repositories):
    for repo in repositories:
        owner, name = repo["full_name"].split("/")
        print(f"Processing repository: {owner}/{name}")
        
        details = get_repo_details(owner, name)
        if details is None:
            continue
            
        try:
            # Get issues with some basic rate limiting
            time.sleep(1)  # Basic rate limit protection
            issues = get_issues(owner, name)
            
            # Sample issues if we have too many
            sample_size = min(50, len(issues))
            sampled_issues = random.sample(issues, sample_size) if sample_size > 0 else []
            
            # Get comments with rate limiting
            print(f"Fetching comments for {len(sampled_issues)} issues...")
            issue_comments = get_all_comments(sampled_issues)
            print(f"Found {len(issue_comments)} comments to analyze")
            
            # Process comments in smaller batches
            batch_size = 50
            all_comments = []
            for i in range(0, len(issue_comments), batch_size):
                batch = issue_comments[i:i + batch_size]
                print(f"Processing comment batch {i//batch_size + 1}/{(len(issue_comments) + batch_size - 1)//batch_size}")
                all_comments.extend(batch)
                time.sleep(1)  # Pause between batches
            
            mean_issue_toxicity, max_issue_toxicity = analyze_toxicity(all_comments)
            
            # Get remaining metrics with basic rate limiting
            time.sleep(1)
            commit_metrics = get_commit_metrics(owner, name)
            time.sleep(1)
            release_metrics = get_release_metrics(owner, name, [(datetime.now(), mean_issue_toxicity)])
            time.sleep(1)
            contributor_metrics = get_contributor_metrics(owner, name)
            time.sleep(1)
            project_metrics = get_project_size_metrics(owner, name)
            issue_metrics = get_issue_metrics(sampled_issues, all_comments)
            
            print(f"Completed processing: {owner}/{name}")
            

            # These are all of the column names that will be in our dataset
            # TODO: add mmore?
            yield {
                "repo_name": details["repo_name"],
                "repo_url": details["repo_url"],
                "created_at": details["created_at"],
                "last_update": details["last_update"],
                "issues_count": len(issues),
                "contributors_count": get_count("contributors", owner, name),
                "mean_issue_toxicity": mean_issue_toxicity,
                "max_issue_toxicity": max_issue_toxicity,
                
                "commit_count": commit_metrics['commit_count'],
                "commit_frequency": commit_metrics['commit_frequency'],
                "active_contributors_per_month": commit_metrics['active_contributors_per_month'],
                
                "release_count": release_metrics['release_count'],
                "pre_release_toxicity": release_metrics['avg_pre_release_toxicity'],
                "post_release_toxicity": release_metrics['avg_post_release_toxicity'],
                
                "avg_contributor_age_days": contributor_metrics['avg_contributor_age_days'],
                "avg_contributor_repos": contributor_metrics['avg_contributor_repos'],
                "avg_contributor_commits": contributor_metrics['avg_contributor_commits'],
                
                "lines_of_code": project_metrics['lines_of_code'],
                "language_count": project_metrics['language_count'],
                "primary_language": project_metrics['primary_language'],
                
                "avg_time_to_resolve_issues": issue_metrics['avg_time_to_resolve'],
                "comment_count_per_issue": issue_metrics['comment_count_per_issue'],
                "open_issues_percentage": issue_metrics['open_issues_percentage'],
                "issue_labels_count": issue_metrics['issue_labels_count']
            }
            
        except Exception as e:
            print(f"Error processing repository {owner}/{name}: {str(e)}")
            continue

# Load to CSV file
def save_to_csv(filtered_repos, overwrite=False):
    csv_filename = "./data/github_data.csv"
    cnt = 0

    # Ensure the data directory exists
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

    # if overwriting clear the file 
    if overwrite:
        with open(csv_filename, mode="w", encoding="utf-8", newline="") as file:
            pass

    # Write repositories one at a time
    for repo in filtered_repos:
        with open(csv_filename, mode="a", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            if cnt == 0:
                # Write headers only for the first row
                writer.writerow(repo.keys())
            writer.writerow(repo.values())
            cnt += 1
            print(f"Saved repository {cnt}: {repo['repo_name']}")

    print(f"Found {cnt} repositories -> saved to {csv_filename}")

def main():
    print("Starting GitHub repository analysis...")
    
    try:
        # Search for repositories
        print("Searching for repositories...")
        repositories = search_repositories()
        
        if not repositories:
            print("No repositories found matching the criteria.")
            return

        print(f"Found {len(repositories)} repositories to analyze.")
        
        # Process repositories and save data
        print("Processing repositories and collecting metrics...")
        filtered_repos = filter_repositories(repositories)
        save_to_csv(filtered_repos, overwrite=True)
        
        print("Analysis complete!")

    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API requests: {e}")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")

if __name__ == "__main__":
    main()