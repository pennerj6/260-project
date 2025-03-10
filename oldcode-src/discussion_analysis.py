from config import *
from api_requests import *
from toxicity_analysis import *

from dateutil import parser
import numpy as np


# Lookign into the discussion activity before and after a toxic comment
def analyze_discussion_activity(issue_url, toxic_comment_list):
    for toxic_comment in toxic_comment_list:
        toxic_date = parser.parse(toxic_comment['created_at'])
        
        # Get all comments for the issue
        all_comments = get_issue_comments(issue_url)
        
        # Sort comments by date
        sorted_comments = sorted(all_comments, key=lambda x: parser.parse(x['created_at']))
        
        # Separate comments before and after the toxic comment
        before_comments = [c for c in sorted_comments if parser.parse(c['created_at']) < toxic_date]
        after_comments = [c for c in sorted_comments if parser.parse(c['created_at']) > toxic_date]
        
        # Calculate metrics
        metrics = {
            'toxic_comment_id': toxic_comment['id'],  # Add the toxic_comment_id here
            'before': {
                'comment_count': len(before_comments),
                'unique_participants': len(set(c['user']['login'] for c in before_comments)) if before_comments else 0,
                'avg_response_time': calculate_avg_response_time(before_comments),
                'comment_frequency': calculate_comment_frequency_before(before_comments, toxic_date)
            },
            'after': {
                'comment_count': len(after_comments),
                'unique_participants': len(set(c['user']['login'] for c in after_comments)) if after_comments else 0,
                'avg_response_time': calculate_avg_response_time(after_comments),
                'comment_frequency': calculate_comment_frequency_after(after_comments, toxic_date)
            }
        }
        
        # Calculate changes in metrics
        metrics['change'] = {
            'comment_count_change': metrics['after']['comment_count'] - metrics['before']['comment_count'],
            'unique_participants_change': metrics['after']['unique_participants'] - metrics['before']['unique_participants'],
            'response_time_change': metrics['after']['avg_response_time'] - metrics['before']['avg_response_time'],
            'comment_frequency_change': metrics['after']['comment_frequency'] - metrics['before']['comment_frequency']
        }
        
    return metrics

# Get comment frequency before toxic comment(expected to be higher than after)
def calculate_comment_frequency_before(comments, toxic_date):
    if not comments:
        return 0
    
    first_comment_date = parser.parse(comments[0]['created_at'])
    days_span = max(1, (toxic_date - first_comment_date).days)
    
    return len(comments) / days_span

# Get comment frequency after toxic comment (expected to be lower than before)
def calculate_comment_frequency_after(comments, toxic_date):
    if not comments:
        return 0
    
    last_comment_date = parser.parse(comments[-1]['created_at'])
    days_span = max(1, (last_comment_date - toxic_date).days + 1)
    
    return len(comments) / days_span

# Get average time btween comments (for testing)
def calculate_avg_response_time(comments):
    if len(comments) <= 1:
        return 0
        
    response_times = []
    sorted_comments = sorted(comments, key=lambda x: parser.parse(x['created_at']))
    
    for i in range(1, len(sorted_comments)):
        current = parser.parse(sorted_comments[i]['created_at'])
        previous = parser.parse(sorted_comments[i-1]['created_at'])
        response_time = (current - previous).total_seconds() / 3600  # hours
        response_times.append(response_time)
    
    return np.mean(response_times) if response_times else 0