from datetime import timedelta
from dateutil import parser

from config import *
from api_requests import *
from toxicity_analysis import *

# Get issues and filter via datarange
def get_issues_in_timeframe(repo_owner, repo_name, start_date, end_date):
    issues_url = f"{BASE_URL}/repos/{repo_owner}/{repo_name}/issues"
    
    params = {
        'state': 'all',  # Get both open and closed issues
        'per_page': 100,
        'sort': 'created',
        'direction': 'desc'  # Start with newest issues
    }
    
    all_issues = []
    page = 1
    
    while True:
        params['page'] = page
        response = make_request(issues_url, params)
        
        if not response or response.status_code != 200 or not response.json():
            break
        
        issues_page = response.json()
        
        # Check if we've gone past our date range
        oldest_in_page = parser.parse(issues_page[-1]['created_at'])
        if oldest_in_page < start_date:
            # Filter this page and stop
            for issue in issues_page:
                created_date = parser.parse(issue['created_at'])
                if start_date <= created_date <= end_date:
                    all_issues.append(issue)
            break
        
        # Filter issues by date range
        for issue in issues_page:
            created_date = parser.parse(issue['created_at'])
            if start_date <= created_date <= end_date:
                all_issues.append(issue)
        
        page += 1
        
        # Sleep to avoid rate limiting
        time.sleep(1)
    
    return all_issues

# Went to OH, TA Kunal suggest we look at the toxicity around realeases too, so ..
# Anazlyses the toxicity around releases (with our intuition of toxicity being higher right before a release, due to tension/stress/etc)
def analyze_toxicity_around_releases(repo_owner, repo_name, window_days=release_window_days):
    #release_window_days is defined in congig, to be changed
    print(f"Analyzing toxicity patterns around releases for {repo_owner}/{repo_name}...")
    
    # Get all releases for the repository
    releases_url = f"{BASE_URL}/repos/{repo_owner}/{repo_name}/releases"
    releases = get_all_pages(releases_url)
    
    if not releases:
        print(f"No releases found for {repo_owner}/{repo_name}")
        return None
    
    # Sort releases by date
    sorted_releases = sorted(releases, key=lambda x: parser.parse(x['published_at']))
    
    release_toxicity = []
    
    for release in sorted_releases:
        release_date = parser.parse(release['published_at'])
        print(f"Analyzing release {release['tag_name']} (published {release_date.strftime('%Y-%m-%d')})")
        
        # Define time windows
        before_window_start = release_date - timedelta(days=window_days)
        before_window_end = release_date
        
        # For comparison, use a "normal" period of same length before the pre-release window
        normal_window_start = before_window_start - timedelta(days=window_days)
        normal_window_end = before_window_start
        
        # Get issues and PRs active during these periods
        before_issues = get_issues_in_timeframe(repo_owner, repo_name, before_window_start, before_window_end)
        normal_issues = get_issues_in_timeframe(repo_owner, repo_name, normal_window_start, normal_window_end)
        
        # Analyze toxicity in comments during these periods
        before_toxicity = calculate_toxicity_metrics(before_issues, before_window_start, before_window_end)
        normal_toxicity = calculate_toxicity_metrics(normal_issues, normal_window_start, normal_window_end)
        
        # Skip releases with no activity
        if before_toxicity['total_comments'] == 0 and normal_toxicity['total_comments'] == 0:
            print(f"Skipping release {release['tag_name']} - no activity in analysis windows")
            continue
        
        release_info = {
            'release_tag': release['tag_name'],
            'release_date': release['published_at'],
            'pre_release_window': {
                'start_date': before_window_start.isoformat(),
                'end_date': before_window_end.isoformat(),
                'total_comments': before_toxicity['total_comments'],
                'toxic_comments': before_toxicity['toxic_comments'],
                'toxicity_percentage': before_toxicity['toxicity_percentage'],
                'avg_toxicity_score': before_toxicity['avg_toxicity_score']
            },
            'normal_window': {
                'start_date': normal_window_start.isoformat(),
                'end_date': normal_window_end.isoformat(),
                'total_comments': normal_toxicity['total_comments'],
                'toxic_comments': normal_toxicity['toxic_comments'],
                'toxicity_percentage': normal_toxicity['toxicity_percentage'],
                'avg_toxicity_score': normal_toxicity['avg_toxicity_score']
            }
        }
        
        release_toxicity.append(release_info)
    
    return release_toxicity