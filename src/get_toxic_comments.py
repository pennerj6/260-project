from dotenv import load_dotenv
import os
import pickle
from toxicity_rater import ToxicityRater
import requests
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import spearmanr

if not 'GITHUB_ACCESS_TOKEN' in os.environ:
    load_dotenv()

GITHUB_ACCESS_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")
HEADERS = {
    "Authorization": f"token {GITHUB_ACCESS_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

def get_comments_with_toxicity(issue):
    url = issue['comments_url']
    tr = ToxicityRater()
    max_attempts = 5
    attempt = 0
    page = 0
    page_size = 100

    params={
        'per_page': page_size
    }

    comments = []
    
    while True:
        params['page'] = page
        response = requests.get(url, params=params, headers=HEADERS)
        
        if response.status_code == 200:
            data = response.json()
            comments.extend(data)
            if len(data) < page_size:
                break
            attempt = 0
            page += 1
        elif response.status_code == 403:  # Rate limit exceeded
            attempt += 1
            remaining_attempts = max_attempts - attempt
            
            if attempt < max_attempts:
                print(f"GitHub rate limit exceeded, retrying in 1 second ({remaining_attempts} attempts remaining)")
                time.sleep(1)
            else:
                print(f"Failed to retrieve comments after {max_attempts} attempts: {url}")
        else:
            print(f"Error retrieving comments, status code: {response.status_code}, URL: {url}")
            break
    for comment in comments:
        comment['toxicity_rating'] = tr.get_toxicity_rating(comment['body'])
    return comments

def get_toxic_comments(issues):
    pickle_file = '../data/toxic_comments.pkl'
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)
    comments = {}
    count = 0
    for issue in issues:
        count += 1
        print(f'Fetching comments from issue {issue['id']} ({count} of {len(issues)} done)')
        comments[issue['id']] = get_comments_with_toxicity(issue)
    with open(pickle_file, 'wb') as f:
        pickle.dump(comments, f)
    return comments

def analyze_delay_after_toxicity(issues, comments):
    previous_toxicity = np.array([])
    delays_after = np.array([])
    
    for issue in issues:
        issue_id = issue['id']
        if len(comments[issue_id]) < 2:
            continue
            
        issue_created_at = datetime.fromisoformat(issue['created_at']).timestamp()
        comment_data = np.array([(datetime.fromisoformat(c['created_at']).timestamp() - issue_created_at, 
                                 c['toxicity_rating']) for c in comments[issue_id]])
        
        sorted_indices = np.argsort(comment_data[:, 0])
        sorted_data = comment_data[sorted_indices]
        
        timestamps = sorted_data[:, 0]
        ratings = sorted_data[:, 1]
        
        delays = np.diff(timestamps)
        prev_toxicity = ratings[:-1]
        
        previous_toxicity = np.append(previous_toxicity, prev_toxicity)
        delays_after = np.append(delays_after, delays)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(previous_toxicity, delays_after, alpha=0.3)
    
    if len(previous_toxicity) > 1:
        mask = ~np.isnan(previous_toxicity) & ~np.isnan(delays_after)
        valid_toxicity = previous_toxicity[mask]
        valid_delays = delays_after[mask]
        
        z = np.polyfit(valid_toxicity, valid_delays, 1)
        p = np.poly1d(z)
        
        sorted_toxicity = np.sort(valid_toxicity)
        plt.plot(sorted_toxicity, p(sorted_toxicity), "r--", linewidth=2)
        
        corr, p_value = spearmanr(valid_toxicity, valid_delays)
        print(f'Correlation: {corr:.3f} (p={p_value:.3e})')
        plt.title(f'Previous Comment Toxicity vs. Delay Until Next Comment')
    else:
        plt.title('Previous Comment Toxicity vs. Delay Until Next Comment')
    
    if len(delays_after) > 0 and np.max(delays_after) / np.min(delays_after[np.nonzero(delays_after)]) > 100:
        plt.yscale('log')
        plt.ylabel('Delay Until Next Comment (seconds, log scale)')
    else:
        plt.ylabel('Delay Until Next Comment (seconds)')
    
    plt.xlabel('Previous Comment Toxicity')
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()

if __name__ == "__main__":
    from get_toxic_issues import get_toxic_issues
    issues = get_toxic_issues()
    comments = get_toxic_comments(issues)
    fig = analyze_delay_after_toxicity(issues, comments)
    plt.savefig('toxicity_vs_delay.png')
    plt.show()