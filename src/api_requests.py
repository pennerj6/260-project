import time
import requests
from config import *

# Connect GitHub API
def make_request(url, params=None):
    if params is None:
        params = {}
    
    # used gpt to help with rate limit issue, implemented a sleep system w  5 max retries before timeout error
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            return response
        elif response.status_code == 401:
            print(f"Unauthorized: Check your GitHub access token. Response: {response.text}")
            return None
        elif response.status_code == 403 and 'rate limit exceeded' in response.text.lower():
            # Handle rate limit
            rate_reset = int(response.headers.get('X-RateLimit-Reset', 0))
            if rate_reset > 0:
                sleep_time = max(rate_reset - time.time(), 0) + 1
                print(f"Rate limit exceeded. Waiting {sleep_time:.0f} seconds...")
                time.sleep(sleep_time)
                retry_count += 1
            else:
                # Default sleep if header not present
                sleep_time = 60 * (2 ** retry_count)
                print(f"Rate limit exceeded. Waiting {sleep_time} seconds...")
                time.sleep(sleep_time)
                retry_count += 1
        else:
            print(f"Error making request to {url}: {response.status_code}. Response: {response.text}")
            return response
    
    print(f"Max retries exceeded for {url}")
    return None

# Given the url, in config, return all the pages associated
def get_all_pages(url, params=None):
    if params is None:
        params = {}
        
    all_items = []
    page = 1
    
    while True:
        page_params = {**params, 'page': page, 'per_page': 100}
        response = make_request(url, page_params)
        
        if response is None or response.status_code != 200:
            break
            
        items = response.json()
        if isinstance(items, list):
            if not items:
                break
            all_items.extend(items)
        else:
            all_items = items
            break
            
        page += 1
        time.sleep(1)  # sleep to avoid API timeout
    
    return all_items

# Given the url of the issue, return the metadata associated
def get_issue_details(issue_url):
    response = make_request(issue_url)
    
    if response is None or response.status_code != 200:
        print(f"Error fetching issue details: {response.status_code if response else 'No response'}")
        return None
    
    return response.json()

# Given teh url of the issue, return the comments associated
def get_issue_comments(issue_url):
    if '/comments' not in issue_url:
        comments_url = f"{issue_url}/comments"
    else:
        comments_url = issue_url
    
    return get_all_pages(comments_url)