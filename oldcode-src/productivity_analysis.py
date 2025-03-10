from dateutil import parser
from datetime import timedelta
from api_requests import get_all_pages
from config import *
from toxicityrater import ToxicityRater
tr = ToxicityRater()

# I dont know if PyDriller is suitable to use bc we have to download each repo locally?
# I could be wrong^

# Get the commits of a repo between a date range
def get_repo_commits(repo_owner, repo_name, start_date, end_date):
    url = f"{BASE_URL}/repos/{repo_owner}/{repo_name}/commits"
    params = {
        'since': start_date.isoformat(),
        'until': end_date.isoformat()
    }
    
    return get_all_pages(url, params)

# Analyze the commit activity before/after a specific toxic comment 
def analyze_productivity(repo_owner, repo_name, toxic_comment_date):
    comment_date = parser.parse(toxic_comment_date)

    before_start = comment_date - timedelta(days=analysis_window_days)
    before_end = comment_date  
    
    day_of_start = comment_date
    day_of_end = comment_date + timedelta(days=1)

    after_start = comment_date + timedelta(days=1)
    after_end = comment_date + timedelta(days=analysis_window_days + 1) 

    before_commits = get_repo_commits(repo_owner, repo_name, before_start, before_end)
    day_of_commits = get_repo_commits(repo_owner, repo_name, day_of_start, day_of_end)
    after_commits = get_repo_commits(repo_owner, repo_name, after_start, after_end)

    daily_data = []

    # Process before commits by day
    for commit in before_commits:
        commit_date = parser.parse(commit['commit']['author']['date']).date()
        days_from_comment = (comment_date.date() - commit_date).days
        
        daily_data.append({
            'date': commit_date.isoformat(),
            'days_from_comment': -days_from_comment,  # Negative for days before
            'period': 'before',
            'author': commit['commit']['author']['name'],
            'email': commit['commit']['author']['email'],
            'message': commit['commit']['message'],
            'toxicity': tr.get_toxicity_ratings([commit['commit']['message']])
        })

    # Process day of commits
    for commit in day_of_commits:
        commit_date = parser.parse(commit['commit']['author']['date']).date()
        days_from_comment = (commit_date - comment_date.date()).days
        
        daily_data.append({
            'date': commit_date.isoformat(),
            'days_from_comment': days_from_comment,  # 0 for the day of
            'period': 'day_of',
            'author': commit['commit']['author']['name'],
            'email': commit['commit']['author']['email'],
            'message': commit['commit']['message'],
            'toxicity': tr.get_toxicity_ratings([commit['commit']['message']])
        })

    # Process after commits by day
    for commit in after_commits:
        commit_date = parser.parse(commit['commit']['author']['date']).date()
        days_from_comment = (commit_date - comment_date.date()).days
        
        daily_data.append({
            'date': commit_date.isoformat(),
            'days_from_comment': days_from_comment,  # Positive for days after
            'period': 'after',
            'author': commit['commit']['author']['name'],
            'email': commit['commit']['author']['email'],
            'message': commit['commit']['message'],
            'toxicity': tr.get_toxicity_ratings([commit['commit']['message']])
        })
        
    return {
        'before_count': len(before_commits),
        'day_of_count': len(day_of_commits),
        'after_count': len(after_commits),
        'daily_data': daily_data,
        'percent_change_before_to_after': ((len(after_commits) - len(before_commits)) / max(1, len(before_commits)) * 100),
        'percent_change_before_to_day_of': ((len(day_of_commits) - len(before_commits)) / max(1, len(before_commits)) * 100)
    }