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

