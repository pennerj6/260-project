import time
from dateutil import parser
import datetime
from datetime import datetime, timedelta
import numpy as np
from googleapiclient.errors import HttpError

from api_requests import *
from config import toxicity_threshold
from toxicityrater import ToxicityRater

# Initialize toxicity rater
tr = ToxicityRater()

# Given list of comments, return only the comments that are toxic
# the toxicity threshold is in config (need to change)
def identify_toxic_comments(comments):
    if not comments:
        return []
    
    # Get toxicity scores for all comments in one batch 
    comment_bodies = [comment['body'] for comment in comments if 'body' in comment]
    toxicity_scores = tr.get_toxicity_ratings(comment_bodies)
    
    # Filter out toxic comments
    toxic_comments = []
    score_index = 0
    for comment in comments:
        if 'body' in comment:
            toxicity_score = toxicity_scores[score_index]
            score_index += 1
            
            if toxicity_score > toxicity_threshold:
                toxic_comment = {
                    'id': comment['id'],
                    'user': comment['user']['login'],
                    'body': comment['body'],
                    'created_at': comment['created_at'],
                    'toxicity_score': toxicity_score,
                    'url': comment['html_url']
                }
                toxic_comments.append(toxic_comment)
    
    return toxic_comments

# Calculate toxicity metrics for comments in a set of issues within a timeframe
def calculate_toxicity_metrics(issues, start_date, end_date):
    all_comments = []
    toxic_comments = []
    total_toxicity_score = 0
    
    for issue in issues:
        comments = get_comments_in_timeframe(issue['url'], start_date, end_date)
        
        # Extract comment texts
        comment_texts = [comment['body'] for comment in comments if 'body' in comment]
        
        # Batch process all comments
        if comment_texts:
            try:
                toxicity_scores = tr.get_toxicity_ratings(comment_texts)
                
                # Process the results
                score_index = 0
                for comment in comments:
                    if 'body' in comment:
                        toxicity_score = toxicity_scores[score_index]
                        score_index += 1
                        
                        all_comments.append({
                            'comment': comment,
                            'toxicity_score': toxicity_score
                        })
                        
                        if toxicity_score > toxicity_threshold:
                            toxic_comments.append({
                                'comment': comment,
                                'toxicity_score': toxicity_score
                            })
                            total_toxicity_score += toxicity_score
                
            except Exception as e:
                print(f"Error processing comments: {e}")
    
    total_comments = len(all_comments)
    toxic_count = len(toxic_comments)
    
    return {
        'total_comments': total_comments,
        'toxic_comments': toxic_count,
        'toxicity_percentage': (toxic_count / total_comments * 100) if total_comments > 0 else 0,
        'avg_toxicity_score': (total_toxicity_score / toxic_count) if toxic_count > 0 else 0
    }

#Get comments for an issue that were created during a chosen timerange
def get_comments_in_timeframe(issue_url, start_date, end_date):
    comments_url = f"{issue_url}/comments"
    all_comments = get_all_pages(comments_url)
    
    # Filter comments by creation date
    filtered_comments = []
    for comment in all_comments:
        comment_date = parser.parse(comment['created_at'])
        if start_date <= comment_date <= end_date:
            filtered_comments.append(comment)
    
    return filtered_comments

# Get non-toxic issues for comparison
def get_non_toxic_issues(repo_owner, repo_name, count=10):
    print("Entered get_non_toxic_issues")
    issues_url = f"{BASE_URL}/repos/{repo_owner}/{repo_name}/issues"
    params = {
        'state': 'all',
        'sort': 'updated',
        'direction': 'desc',
        'per_page': 50  # Fetch only 50 issues per page
    }
    
    issues = get_all_pages(issues_url, params)
    
    non_toxic_issues = []
    
    for issue in issues:
        if 'pull_request' in issue:  # Skip PRs
            continue
            
        comments = get_issue_comments(issue['comments_url'])
        
        # Extract comment bodies
        comment_bodies = [comment['body'] for comment in comments if 'body' in comment and comment['body']]
        
        # Check if any comments are toxic
        is_toxic = False
        if comment_bodies:
            toxicity_scores = tr.get_toxicity_ratings(comment_bodies)
            is_toxic = any(score > toxicity_threshold for score in toxicity_scores)
        
        if not is_toxic:
            non_toxic_issues.append(issue)
            if len(non_toxic_issues) >= count:
                break
    
    return non_toxic_issues

# compare toxic vs nontoxic
def analyze_issue_resolution_metrics(repo_owner, repo_name, issues_with_toxicity, issues_without_toxicity):
    print("Entered analyze_issue_resolution_metrics")
    
    resolution_metrics = {
        'toxic': {
            'time_to_close': [],
            'comment_count': [],
            'abandonment_rate': 0,
        },
        'non_toxic': {
            'time_to_close': [],
            'comment_count': [],
            'abandonment_rate': 0,
        }
    }
    
    # Process issues with toxicity
    for issue in issues_with_toxicity:
        if issue['state'] == 'closed' and 'closed_at' in issue:
            created_at = parser.parse(issue['created_at'])
            closed_at = parser.parse(issue['closed_at'])
            resolution_time = (closed_at - created_at).total_seconds() / 3600  # hours
            resolution_metrics['toxic']['time_to_close'].append(resolution_time)
        else:
            # Count as abandoned if open for more than 90 days
            created_at = parser.parse(issue['created_at'])
            if (datetime.now(created_at.tzinfo) - created_at).days > 90:
                resolution_metrics['toxic']['abandonment_rate'] += 1
                
        # Get comment count
        comments_url = issue['comments_url']
        comments = get_issue_comments(comments_url)
        resolution_metrics['toxic']['comment_count'].append(len(comments))
    
    # Process issues without toxicity
    for issue in issues_without_toxicity:
        if issue['state'] == 'closed' and 'closed_at' in issue:
            created_at = parser.parse(issue['created_at'])
            closed_at = parser.parse(issue['closed_at'])
            resolution_time = (closed_at - created_at).total_seconds() / 3600  # hours
            resolution_metrics['non_toxic']['time_to_close'].append(resolution_time)
        else:
            # Count as abandoned if open for more than 90 days
            created_at = parser.parse(issue['created_at'])
            if (datetime.now(created_at.tzinfo) - created_at).days > 90:
                resolution_metrics['non_toxic']['abandonment_rate'] += 1
                
        # Get comment count
        comments_url = issue['comments_url']
        comments = get_issue_comments(comments_url)
        resolution_metrics['non_toxic']['comment_count'].append(len(comments))
    
    # Calculate final metrics
    if issues_with_toxicity:
        resolution_metrics['toxic']['abandonment_rate'] /= len(issues_with_toxicity)
    if issues_without_toxicity:
        resolution_metrics['non_toxic']['abandonment_rate'] /= len(issues_without_toxicity)
    
    # Calculate averages
    resolution_metrics['toxic']['avg_time_to_close'] = (
        sum(resolution_metrics['toxic']['time_to_close']) / len(resolution_metrics['toxic']['time_to_close'])
        if resolution_metrics['toxic']['time_to_close'] else 0
    )
    resolution_metrics['non_toxic']['avg_time_to_close'] = (
        sum(resolution_metrics['non_toxic']['time_to_close']) / len(resolution_metrics['non_toxic']['time_to_close'])
        if resolution_metrics['non_toxic']['time_to_close'] else 0
    )
    
    resolution_metrics['toxic']['avg_comment_count'] = (
        sum(resolution_metrics['toxic']['comment_count']) / len(resolution_metrics['toxic']['comment_count'])
        if resolution_metrics['toxic']['comment_count'] else 0
    )
    resolution_metrics['non_toxic']['avg_comment_count'] = (
        sum(resolution_metrics['non_toxic']['comment_count']) / len(resolution_metrics['non_toxic']['comment_count'])
        if resolution_metrics['non_toxic']['comment_count'] else 0
    )
    
    return resolution_metrics