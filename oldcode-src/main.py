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
    # Return the data associated with issue url
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
            
            # Filter out the toxic comments
            both_toxic_comments = identify_both_toxic_comments(comments)
            
            # Separate the toxic and non-toxic comments into different lists
            toxic_comments_list = [comment for comment in both_toxic_comments if comment['status'] == 'Toxic']
            non_toxic_comments_list = [comment for comment in both_toxic_comments if comment['status'] == 'Non-Toxic']

            # Count the number of toxic and non-toxic comments
            toxic_count = len(toxic_comments_list)
            non_toxic_count = len(non_toxic_comments_list)

            # Optionally, you can print the lists or use them for further processing
            print("Toxic Comments:", toxic_count)
            print("Non-Toxic Comments:", non_toxic_count)

            if not toxic_comments_list:
                continue
                
            all_toxic_comments.extend(toxic_comments_list)
            
            # Analyze productivity change for each toxic comment
            for toxic_comment in toxic_comments_list:
                result = analyze_productivity(repo_owner, repo_name, toxic_comment['created_at'])
                result['issue_url'] = issue_url
                result['toxic_comment'] = toxic_comment
                all_results.append(result)
            
            # Analyze issue resolution metrics
            issue_resolution_metrics = analyze_issue_resolution_metrics(
                repo_owner, repo_name,
                [issue], # i chnaged this to be a list of 1 so we can loop thru 1 issue at a time
                toxic_comments_list, # all the "toxic" comments in the curr issue
                non_toxic_comments_list # all the "nontoxic" comments in the curr issue (in quotes bc we determine the threshaold in config)
            )
            issue_details[issue_url]['resolution_metrics'] = issue_resolution_metrics
            
            # Analyze toxicity around releases
            release_toxicity = analyze_toxicity_around_releases(repo_owner, repo_name)
            issue_details[issue_url]['release_toxicity'] = release_toxicity
         
            # Calculate or define discussion_metrics
            discussion_metrics = analyze_discussion_activity(issue_url, toxic_comments_list)
    
    # Generate enhanced CSV files for analysis
    if all_results:
        export_csv(all_results, all_toxic_comments, issue_details, discussion_metrics)
        export_research_summary(issue_details)

if __name__ == "__main__":
    main()
