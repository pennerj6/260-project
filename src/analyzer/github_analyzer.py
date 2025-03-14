import pandas as pd
import numpy as np
import datetime
import time
import requests
import os
import logging
import random
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed # this speeds up our code BUT, im pretty sure our github api limit will get used more (thats ok tho, met with TA he said we dont need to anaylze as much repos as we thought)

from toxicity_rater import ToxicityRater
from helper import get_issue_number
from config import *

logger = logging.getLogger(__name__)


# thiso is meant to get the data from githib and get the toxicity score
class GitHubToxicityAnalyzer:
    def __init__(self, repos, github_tokens, test_mode=False):
       
        # ur gonna need to import the toxicity rater from somewhere once we find one that doesnt hit hte rate limit every 5 seconds
        # this is where we wud normally import it but i'll keep what i have so far 
        # update: i swapped perspective API for huggingface model. (i prompted gpt/reddit searched to give me a list of restApi i can use to rate toxicicty score with no limit)
        self.rater = ToxicityRater(use_sampling=True, sample_size=SAMPLE_SIZE) 
        
        self.repos = repos
        self.github_tokens = github_tokens
        self.current_token_index = 0
        self.test_mode = test_mode
        self.headers = self._get_headers()
        self.base_url = GITHUB_BASE_URL
        self.data = {
            'comments': [],
            'commits': [],
            'issues': [],
            'releases': [],
            'contributors': []
        }

        # kept getting token error, was told to implement a token chekcer/validator to make sure valid token & the ratelimit stats
        self._validate_tokens()
        

    # this validation wastes an api call so i might NOT incldude this if it isnt needed
    def _validate_tokens(self):
        valid_tokens = []
        for i, token in enumerate(self.github_tokens):
            if token is None:  
                continue
            headers = {'Authorization': f'token {token}', 'Accept': 'application/vnd.github.v3+json'}
            try:
                response = requests.get(f"{self.base_url}/rate_limit", headers=headers, timeout=API_REQUEST_TIMEOUT)
                if response.status_code == 200:
                    rate_data = response.json()
                    remaining = rate_data.get('resources', {}).get('core', {}).get('remaining', 0)
                    logger.info(f"Token {i + 1} is valid. Remaining API calls: {remaining}")
                    valid_tokens.append(token)
                else:
                    logger.error(f"Token {i + 1} validation failed: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"Error validating token {i + 1}: {str(e)}")

        if not valid_tokens:
            raise ValueError("No valid GitHub tokens available! Please check your tokens and permissions.")
        self.github_tokens = valid_tokens
        self.current_token_index = 0  
        
        # only keep the validated tokems
        self.headers = self._get_headers() 
        
    def _get_headers(self):
        return {
            'Authorization': f'Bearer {self.github_tokens[self.current_token_index]}',
            'Accept': 'application/vnd.github.v3+json'
        }

    def _rotate_token(self):
        # switch tokens around
        self.current_token_index = (self.current_token_index + 1) % len(self.github_tokens)
        self.headers = self._get_headers()
        logger.info(f"Rotated to token index: {self.current_token_index}")

    def _apply_test_mode(self, params):
        # using the test mode limitor
        # this will ALWAYS be the case fofr our project bc of time contraint (too much computation time)
        # we will still get sufficient data, just not ALL the data
        if self.test_mode:
            params['per_page'] = TEST_MODE_ITEMS_LIMIT  # its in conifg ( i put most constants that we need to change/adjust in there)
        return params

    def _paginated_request(self, url, params=None):
        # Thhis is where i got MANY issues 
        # Issues and waht the curr sol is:
        # - token keeps returning 401 error (why i did the validation aboive)
        # - tokens are not roatating properly (made rotate fxn)
        # - rate limit exceedes (added sleep/retry w help)
        
        current_token = self.github_tokens[self.current_token_index]
        logger.info(f"Making request to {url} with token {current_token[:4]}xxx{current_token[-4:] if len(current_token) > 8 else ''}")

        all_items = []
        if params is None:
            params = {}
        params = self._apply_test_mode(params)

        # Only fetch data from the last 3 months to save TIMEEEE (might reduce if still slow)
        three_months_ago = datetime.datetime.now() - datetime.timedelta(days=90)
        params['since'] = three_months_ago.strftime('%Y-%m-%d')

        retry_count = 0
        max_retries = 3
        base_delay = 1  

        while url and (not self.test_mode or len(all_items) < TEST_MODE_ITEMS_LIMIT):
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=API_REQUEST_TIMEOUT)
                logger.info(f"Response status: {response.status_code}")

                if response.status_code == 403 and 'rate limit exceeded' in response.text.lower():
                    retry_count += 1
                    if retry_count <= max_retries:
                        delay = base_delay * (2 ** (retry_count - 1))  # looked up code for exponential backoff, i might increase the retry count or time it sleeps
                        logger.warning(f"Rate limit exceeded. Retrying in {delay} seconds")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error("All tokens rate limited. Waiting 60 seconds before retry.")
                        time.sleep(60)
                        retry_count = 0
                        continue

                if response.status_code != 200:
                    logger.error(f"Error {response.status_code}: {response.text}")
                    break

                items = response.json()
                all_items.extend(items)

                if 'next' in response.links:
                    url = response.links['next']['url']
                    params = {}  
                else:
                    break

                # Respect rate limits
                remaining = int(response.headers.get('X-RateLimit-Remaining', '1'))
                if remaining < RATE_LIMIT_THRESHOLD:  # rotate token if running low on ratel imit
                    logger.info(f"Token {self.current_token_index + 1} running low on requests. Rotating.")
                    self._rotate_token()

                # gpt said sometimes api w block requests if they are sus, so make it random sleep times not all the same
                # i dont know if i fully believe that for github api, but it doesnt hurt to implement
                time.sleep(random.uniform(0.1, 0.5))

            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {str(e)}")
                retry_count += 1
                if retry_count <= max_retries:
                    time.sleep(2) 
                    continue
                else:
                    break

        return all_items


    def fetch_data(self):
        # w threads, get all needed data from github api
        # multiple requests happending at a time w threads might hit rate limit faster (thats why i made multiple github tokens)
        logger.info("Starting data fetching")
        # in class professor mentioned threading to speed things up, used gpt to help implement this w 5 workers (dont want to hit rate limit so kept it at 5, might increase)
        with ThreadPoolExecutor(max_workers=5) as executor:
            
            futures = {}

            # we need to have these run in seperate loops to avoid mem error issue
            
            # comment fetching
            for repo in self.repos:
                futures[executor.submit(self._fetch_comments, repo)] = f"comments_{repo}"
            
            # commit fetching
            for repo in self.repos:
                futures[executor.submit(self._fetch_commits, repo)] = f"commits_{repo}"
            
            # issue fetching
            for repo in self.repos:
                futures[executor.submit(self._fetch_issues, repo)] = f"issues_{repo}"
            
            # release fetching
            for repo in self.repos:
                futures[executor.submit(self._fetch_releases, repo)] = f"releases_{repo}"
            
            # contributor fetching
            for repo in self.repos:
                futures[executor.submit(self._fetch_contributors, repo)] = f"contributors_{repo}"
            
            # process data
            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    future.result()
                    logger.info(f"Completed task: {task_name}")
                except Exception as e:
                    logger.error(f"Error in task {task_name}: {e}")

        logger.info("Data fetching complete")
        
    def _fetch_comments(self, repo):
        #given a REPO get comments from issues and PRs, only a sample size from _paginated_request (the number is set in config)
        logger.info(f"Fetching comments for repository: {repo}")
        try:
            # get issue comments
            issue_comments_url = f"{self.base_url}/repos/{repo}/issues/comments"
            params = {'per_page': 100} # not sure what happends if i change to 1000
            issue_comments = self._paginated_request(issue_comments_url, params) # this should return samplesize(1000 i think) comments or less

            # get PR review comments
            pr_comments_url = f"{self.base_url}/repos/{repo}/pulls/comments" # same as issues, just request w pulls/comments
            pr_comments = self._paginated_request(pr_comments_url, params)  

            # mErge the issue and pullrequests comments
            all_comments = issue_comments + pr_comments
            # For each comment, load/process it to be used later
            for comment in all_comments:
                if 'body' in comment and comment['body']:
                    try:
                        toxicity = self.rater.get_toxicity(comment['body'])
                        if not isinstance(toxicity, dict) or 'score' not in toxicity:
                            logger.error(f"Invalid toxicity score format for comment: {comment['id']}")
                            continue

                        comment_data = {
                            'repo': repo,
                            'comment_id': comment['id'],
                            'user_id': comment['user']['id'],
                            'user_login': comment['user']['login'],
                            'created_at': comment['created_at'],
                            'updated_at': comment.get('updated_at'),
                            'body': comment['body'],
                            'toxicity': toxicity,
                            'type': 'issue_comment' if 'issue_url' in comment else 'pr_comment',
                            'issue_number': get_issue_number(comment.get('issue_url', comment.get('pull_request_url', '')))
                        }
                        self.data['comments'].append(comment_data)
                    except Exception as e:
                        logger.error(f"Error calculating toxicity for comment {comment['id']}: {e}")

            logger.info(f"Fetched {len(all_comments)} comments for repository: {repo}")

        except Exception as e:
            logger.error(f"Error fetching comments for repository {repo}: {e}")

    def _fetch_commits(self, repo):
        # Same as commets, but get commits instead (sample size amoutn)
        logger.info(f"Fetching commits for repository: {repo}")
        commits_url = f"{self.base_url}/repos/{repo}/commits"
        params = {'per_page': 100}
        commits = self._paginated_request(commits_url, params)
        logger.info(f"Fetched {len(commits)} commits for repository: {repo}")

        # for eac commit, store data in format to be used later
        for commit in commits:
            if commit.get('author') and commit.get('commit'):
                commit_data = {
                    'repo': repo,
                    'sha': commit['sha'],
                    'author_id': commit['author']['id'] if commit.get('author') else None,
                    'author_login': commit['author']['login'] if commit.get('author') else None,
                    'date': commit['commit']['author']['date'],
                    'message': commit['commit']['message']
                }
                self.data['commits'].append(commit_data)

    def _fetch_issues(self, repo):
        # same as above fxns, but w issues but we are skipping issues in pull requests
        issues_url = f"{self.base_url}/repos/{repo}/issues"
        params = {'state': 'all', 'per_page': 100}
        issues = self._paginated_request(issues_url, params)

        #For each issue, store the data in format to be used later
        for issue in issues:
            # dont want issue data form PR
            if 'pull_request' in issue:
                continue  

            issue_data = {
                'repo': repo,
                'issue_number': issue['number'],
                'title': issue['title'],
                'user_id': issue['user']['id'],
                'user_login': issue['user']['login'],
                'state': issue['state'],
                'created_at': issue['created_at'],
                'closed_at': issue.get('closed_at'),
                'comments_count': issue['comments']
            }
            self.data['issues'].append(issue_data)

    def _fetch_releases(self, repo):
        # like above, we get relase info
        releases_url = f"{self.base_url}/repos/{repo}/releases"
        params = {'per_page': 100} # i might make this a variable in congif, so if i mess around it will dynamically chage everywhere (thats probably better practice to do hat as well)
        releases = self._paginated_request(releases_url, params)

        # for each release , store data in nice format
        for release in releases:
            release_data = {
                'repo': repo,
                'id': release['id'],
                'tag_name': release['tag_name'],
                'name': release['name'],
                'created_at': release['created_at'],
                'published_at': release['published_at'],
                'author_id': release['author']['id'],
                'author_login': release['author']['login']
            }
            self.data['releases'].append(release_data)

    def _fetch_contributors(self, repo):
        # get contributor data, similar to other fxn
        '''
            # could we do all these _fetch_ fxns in 1 big function?
            # theres alot of code im repeating i just dont know how to effectivley restrucutre that (in some readings i did i know in SE theres different patterns/adapters that might apply to reduce all the code into 1 or 2 functions/classes)
        '''
        contributors_url = f"{self.base_url}/repos/{repo}/contributors"
        params = {'per_page': 100} # do we want to limit this obe?
        contributors = self._paginated_request(contributors_url, params)

        # get data for each contribior
        for contributor in contributors:
            user_url = f"{self.base_url}/users/{contributor['login']}"
            
            # for every contributor, we try max 3 times for api connection (should probably do this in above fxns too)
            retry_count = 0
            max_retries = 3
            while retry_count <= max_retries:
                user_response = requests.get(user_url, headers=self.headers, timeout=API_REQUEST_TIMEOUT)

                if user_response.status_code == 401:
                    logger.error(f"Authentication error with token {self.current_token_index + 1}")
                    self._rotate_token()
                    retry_count += 1
                    continue

                if user_response.status_code == 403 and 'rate limit exceeded' in user_response.text.lower():
                    self._rotate_token()
                    retry_count += 1
                    continue

                if user_response.status_code == 200:
                    user_data = user_response.json()
                    contributor_data = {
                        'repo': repo,
                        'user_id': contributor['id'],
                        'user_login': contributor['login'],
                        'contributions': contributor['contributions'],
                        'account_created_at': user_data.get('created_at'),
                        'public_repos': user_data.get('public_repos', 0),
                        'followers': user_data.get('followers', 0)
                    }
                    self.data['contributors'].append(contributor_data)
                    break
                else:
                    logger.error(f"Error {user_response.status_code} for user {contributor['login']}: {user_response.text}")
                    break

            # randonm sleep for rate limit
            time.sleep(random.uniform(0.1, 0.5))

    def convert_to_dataframes(self):
        # convert data to pandas df
        # i personally dont like to use pandas datafram bc idk the syntax but, professor reccommeneded it (and its well know/easyier to debug w so we will use it!)
        dataframes = {}
        for key, items in self.data.items():
            logger.info(f"Converting {len(items)} {key} items to DataFrame")
            if items:
                dataframes[key] = pd.DataFrame(items)
                
                #str to daatetime
                for col in dataframes[key].columns:
                    # might edge case for ifinstatnce not string
                    if 'date' in col or col.endswith('_at'):
                        dataframes[key][col] = pd.to_datetime(dataframes[key][col])
        
        logger.info(f"Created dataframes: {list(dataframes.keys())}")
        return dataframes
    
    