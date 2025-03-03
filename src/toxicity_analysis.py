import time
from dateutil import parser
from googleapiclient.errors import HttpError

from api_requests import *
from config import toxicity_threshold
from toxicityrater import ToxicityRater

# Initialize toxicity rater
tr = ToxicityRater()

# Given list of comments, return only the comments that are toxic
# the toxicity threshold is in config (need to change)

def identify_toxic_comments(comments):
    toxic_comments = []
    for comment in comments:
        if 'body' in comment and comment['body']:
            # tr uses Perspective API to calc hte toxicitiy of comment from 0 to 1
            toxicity_score = tr.get_toxicity_rating(comment['body'])
            # threshold is determined in congif            
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
        
        for comment in comments:
            comment_text = comment['body']

            try:
                # using Perspective API
                toxicity_score = tr.get_toxicity_rating(comment_text)
                
            # used gpt to help w rate limit error handling
            except HttpError as e:
                if e.resp.status == 429:  # Rate limit error
                    print("Rate limit exceeded. Retrying after 60 seconds...")
                    time.sleep(60)  # Wait for 60 seconds before retrying
                    toxicity_score = tr.get_toxicity_rating(comment_text)  # Retry the request
                else:
                    # Handle other HTTP errors
                    print(f"An error occurred: {e}")

            # Sleep to avoid hitting the rate limit
            time.sleep(1)

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