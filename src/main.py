from config import *
from api_requests import *

from toxicity_analysis import *
from productivity_analysis import *
from release_analysis import *
from contributor_analysis import *
from discussion_analysis import *

from data_export import *

def main():
    all_results = []
    all_toxic_comments = []
    issue_details = {}
    
    for issue_url in issue_urls:
        print(f"Analyzing issue: {issue_url}")
        
        # Get issue details to extract repository information
        issue = get_issue_details(issue_url)
        if not issue:
            continue
            
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
        
        # Get comments for the issue
        comments = get_issue_comments(issue_url)
        print(f"Found {len(comments)} comments")
        
        # Identify toxic comments
        toxic_comments = identify_toxic_comments(comments)
        print(f"Found {len(toxic_comments)} toxic comments")
        
        # Skip if no toxic comments found
        if not toxic_comments:
            continue
            
        all_toxic_comments.extend(toxic_comments)
        
        # Analyze productivity change for each toxic comment (original analysis)
        for toxic_comment in toxic_comments:
            print(f"\nAnalyzing impact of toxic comment ID: {toxic_comment['id']}")
            
            result = analyze_productivity(repo_owner, repo_name, toxic_comment['created_at'])
            result['issue_url'] = issue_url
            result['toxic_comment'] = toxic_comment
            
            print(f"Before: {result['before_count']} commits, After: {result['after_count']} commits")
            print(f"Percent change in commits (negative means commit count went down): {result['percent_change']:.1f}% \n")
            
            all_results.append(result)
            
            # Q1: Impact on productivity (beyond just commits)
            print(f"Analyzing broader discussion activity impact...")
            discussion_metrics = analyze_discussion_activity(issue_url, toxic_comment)
            result['discussion_metrics'] = discussion_metrics
            
        # Get a control group of non-toxic issues
        print(f"\nGetting control group of non-toxic issues for {repo_owner}/{repo_name}...")
        print("THIS DEPENDS ON THE toxicity_threshold variable.... IT is currently 0.01, for testing")
        print("toxicity_threshold is what percent is considered toxic, that's why I put it as low (so all comments are considered toxic)")
        non_toxic_issues = get_non_toxic_issues(repo_owner, repo_name)
        
        # Analyze issue resolution metrics comparing toxic vs non-toxic
        issue_resolution_metrics = analyze_issue_resolution_metrics(
            repo_owner, repo_name,
            [issue],
            non_toxic_issues
        )
        issue_details[issue_url]['resolution_metrics'] = issue_resolution_metrics
        print(f"Issue resolution comparison: toxic vs non-toxic complete")
        
        # Q2: Toxicity around releases
        print(f"Analyzing toxicity patterns around releases for {repo_owner}/{repo_name}...")
        release_toxicity = analyze_toxicity_around_releases(repo_owner, repo_name)
        issue_details[issue_url]['release_toxicity'] = release_toxicity
        
        # Q3: Contributor experience correlation
        print(f"Analyzing correlation between contributor experience and toxicity...")
        if 1 == 0:  # Temporarily disabled for testing
            contributor_analysis = analyze_contributor_experience_toxicity(repo_owner, repo_name)
            issue_details[issue_url]['contributor_analysis'] = contributor_analysis

    # Generate enhanced CSV files for analysis (i might use PBI)
    if all_results:
        export_csv(all_results, all_toxic_comments, issue_details, discussion_metrics)
        print("Generated enhanced CSV files for analysis")
        
        # Create summary of the research questions
        export_research_summary(issue_details)
        print("Generated research questions summary")

if __name__ == "__main__":
    main()


