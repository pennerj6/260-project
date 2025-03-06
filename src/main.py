from config import *
from api_requests import *

from toxicity_analysis import *
from productivity_analysis import *
from release_analysis import *
from contributor_analysis import *
from discussion_analysis import *

from data_export import *

from concurrent.futures import ThreadPoolExecutor

import cProfile # for checking which part of the code is slow (its was expected: the API calls)

# Fetch issue details and comments in parallel.
def fetch_issue_details(issue_url):
    issue = get_issue_details(issue_url)
    if not issue:
        return None
    
    comments = get_issue_comments(issue_url)
    return issue_url, issue, comments

def main():
    all_results = []
    all_toxic_comments = []
    issue_details = {}
    
    # Fetch issue details and comments in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_issue_details, issue_url) for issue_url in issue_urls]
        for future in futures:
            result = future.result()
            if not result:
                continue
            issue_url, issue, comments = result  # Unpack issue_url from the result
            
            repo_url = issue['repository_url']
            repo_parts = repo_url.split('/')
            repo_owner = repo_parts[-2]
            repo_name = repo_parts[-1]
            
            # Store issue details for later use
            issue_details[issue_url] = {
                'owner': repo_owner,
                'name': repo_name,
                'number': issue['number'],
                'title': issue['title'],
                'state': issue['state'],
                'created_at': issue['created_at'],
                'updated_at': issue['updated_at']
            }
            
            # Identify toxic comments
            toxic_comments = identify_toxic_comments(comments)
            print(f"Found {len(toxic_comments)} toxic comments")
            
            if not toxic_comments:
                continue
                
            all_toxic_comments.extend(toxic_comments)
            
            # Analyze productivity change for each toxic comment
            for toxic_comment in toxic_comments:
                result = analyze_productivity(repo_owner, repo_name, toxic_comment['created_at'])
                result['issue_url'] = issue_url
                result['toxic_comment'] = toxic_comment
                all_results.append(result)
            
            # Get a control group of non-toxic issues
            non_toxic_issues = get_non_toxic_issues(repo_owner, repo_name)
            
            # Analyze issue resolution metrics
            issue_resolution_metrics = analyze_issue_resolution_metrics(
                repo_owner, repo_name,
                [issue],
                non_toxic_issues
            )
            issue_details[issue_url]['resolution_metrics'] = issue_resolution_metrics
            
            # Analyze toxicity around releases
            release_toxicity = analyze_toxicity_around_releases(repo_owner, repo_name)
            issue_details[issue_url]['release_toxicity'] = release_toxicity
    
    # Generate enhanced CSV files for analysis
    if all_results:
        # Calculate or define discussion_metrics
        discussion_metrics = {}  # Replace this with actual discussion metrics
        export_csv(all_results, all_toxic_comments, issue_details, discussion_metrics)
        export_research_summary(issue_details)

if __name__ == "__main__":
    main()
