from config import *
from api_requests import *
from toxicity_analysis import *

from dateutil import parser
import numpy as np


# Look into the correlation between contributor experience("age") and toxicity
def analyze_contributor_experience_toxicity(repo_owner, repo_name):
    # Get all contributors to the repository
    contributors_url = f"{BASE_URL}/repos/{repo_owner}/{repo_name}/contributors"
    contributors = get_all_pages(contributors_url)
    
    contributor_metrics = []
    
    for contributor in contributors[:20]:  # Limit to top 20 to avoid excessive API calls
        username = contributor['login']
        
        # Get user details to determine account age
        user_url = f"{BASE_URL}/users/{username}"
        user_response = make_request(user_url)
        
        if user_response is None or user_response.status_code != 200:
            print(f"Error fetching user data for {username}: {user_response.status_code if user_response else 'No response'}")
            continue
            
        user_data = user_response.json()
        account_created_at = parser.parse(user_data['created_at'])
        account_age_days = (datetime.now(account_created_at.tzinfo) - account_created_at).days
        
        # Get contributions to this repository
        contributions_count = contributor['contributions']
        
        # Get all comments by this user in the repository
        user_comments = get_user_comments_in_repo(repo_owner, repo_name, username)
        
        # Skip if no comments
        if not user_comments:
            continue
        
        # Analyze toxicity in user's comments
        toxic_comments = []
        toxicity_scores = []
        
        for comment in user_comments:
            if 'body' in comment and comment['body']:
                toxicity_score = tr.get_toxicity_rating(comment['body'])
                time.sleep(1)
                toxicity_scores.append(toxicity_score)
                
                if toxicity_score > toxicity_threshold:
                    toxic_comments.append(comment)
        
        # Calculate metrics
        contributor_metrics.append({
            'username': username,
            'account_age_days': account_age_days,
            'contributions_count': contributions_count,
            'total_comments': len(user_comments),
            'toxic_comments': len(toxic_comments),
            'toxicity_percentage': (len(toxic_comments) / max(1, len(user_comments))) * 100,
            'avg_toxicity_score': sum(toxicity_scores) / max(1, len(toxicity_scores)) if toxicity_scores else 0
        })
    
    # Calculate correlations
    account_ages = [m['account_age_days'] for m in contributor_metrics]
    contribution_counts = [m['contributions_count'] for m in contributor_metrics]
    toxicity_percentages = [m['toxicity_percentage'] for m in contributor_metrics]
    
    # Only calculate correlation if we have enough data points
    if len(account_ages) > 1:
        age_toxicity_correlation = np.corrcoef(account_ages, toxicity_percentages)[0, 1]
        contribution_toxicity_correlation = np.corrcoef(contribution_counts, toxicity_percentages)[0, 1]
    else:
        age_toxicity_correlation = 0
        contribution_toxicity_correlation = 0
    
    # Bin contributors by experience level for clearer analysis
    experience_bins = {
        'new': {'toxic_percentage': [], 'count': 0},
        'intermediate': {'toxic_percentage': [], 'count': 0},
        'experienced': {'toxic_percentage': [], 'count': 0}
    }
    
    for metrics in contributor_metrics:
        if metrics['account_age_days'] < 180:  # Less than 6 months
            bin_key = 'new'
        elif metrics['account_age_days'] < 730:  # Less than 2 years
            bin_key = 'intermediate'
        else:
            bin_key = 'experienced'
            
        experience_bins[bin_key]['toxic_percentage'].append(metrics['toxicity_percentage'])
        experience_bins[bin_key]['count'] += 1
    
    # Calculate average toxicity by experience group
    for bin_key, bin_data in experience_bins.items():
        bin_data['avg_toxicity'] = sum(bin_data['toxic_percentage']) / max(1, len(bin_data['toxic_percentage']))
    
    return {
        'contributor_metrics': contributor_metrics,
        'age_toxicity_correlation': age_toxicity_correlation,
        'contribution_toxicity_correlation': contribution_toxicity_correlation,
        'experience_bins': experience_bins
    }