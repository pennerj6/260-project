import pandas as pd
import numpy as np
import datetime
import time
import requests
import os
import logging
import random
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed # this speeds up our code BUT, im pretty sure our github api limit will get used more (thats ok tho, sahil met with TA he said we dont need to anaylze as much repos as we thought)

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
    
    
    ''' 
    UPDATED RQ's after sahil went to office hours (big help with #2, overall before the questions i made we good, but he just helped gather thoughts and reprahse to better/clear way)

        1) Does toxic communication in OSS communities negatively affect programmer productivity, measured through commits, issue resolutions, and discussion activity?

        2) Is there any correlation between toxic communication and software releases? (Spearman correlation mentined in OH)

        3) How does the level of experience of the contributors (measured by the age of the account and previous contributions) correlate with their likelihood of engaging in toxic communication within OSS communities? (Spearman correlation)
    '''
    
    # 1) Does toxic communication in OSS communities negatively affect programmer productivity, measured through commits, issue resolutions, and discussion activity?
    def analyze_toxicity_vs_productivity(self):
        # TODO: analyze toxicity against "productivty"
        # measured through  
            # shuffled it in the code to first get comments(disucssion) , commits, then issue resolution
            # A) commits, 
            # B) issue resolutions, and 
            # C) discussion activity --------------- sahil went to OH for help on this, he said we can use COMMENTS or basic activity history(emails, etc) bc that includes non-developers (like project managers) so we can see if they have affect on toxicity (TA; say smthing laong lines of " we know XYZ have affects on team dyanmic, do they have affect on toxicity?" ill clean that sentence up later)
        # along w scatter blots/boxplots/barcharts, we can probably do Spearman correlation too

        logger.info("Analyzing toxicity vs productivity")
        results = {}

        try:
            # hasmap to pandas df
            dfs = self.convert_to_dataframes()
            
            # we should be filtering our data to ONLY keep the ones w the requored fields
            # sahil went to OH and got it approved that we can drop insufficient data for our report
            if 'comments' not in dfs or 'commits' not in dfs or 'issues' not in dfs:
                logger.error("Required data missing for toxicity vs productivity analysis")
                return results

            comments_df = dfs['comments']
            commits_df = dfs['commits']
            issues_df = dfs['issues']


            # the code that was giving me toruble:
            # reason was i was trying to access comments_df[toxicity][score]
            # and i didnt consider that comments_df[toxicity] gives a whole LIST of scores for all the comments , used ai to help me fix it and they reccommend to use a lambda gneerator which worked!
            # GET COMMENTS (toxicity part)
            comments_df['toxicity_score'] = comments_df['toxicity'].apply(
                lambda x: x['score'] if isinstance(x, dict) and 'score' in x else 0
            )
            # Filter toxic and non-toxic comments via the threshold i set in config 
            # idk what we will use non_toxic comments for (bc the ratio is like 1 million nontoxic to 1 toxic, TA said to focus on just the toxic)
            toxic_comments = comments_df[comments_df['toxicity_score'] > TOXICITY_THRESHOLD].copy()
            non_toxic_comments = comments_df[comments_df['toxicity_score'] <= TOXICITY_THRESHOLD].copy()

        # (A) COMMITS 
            # given the toxic comment, whats the toxicity look like 7 days before and 7 days after
            commit_impact_data = []
            window_days = 7  # might increase

            for _, toxic_comment in toxic_comments.iterrows():
                repo = toxic_comment['repo']
                issue_number = toxic_comment['issue_number']
                comment_date = toxic_comment['created_at']

                # Get commits 7 days before and after the toxic comment
                before_window_start = comment_date - pd.Timedelta(days=window_days)
                commits_before = commits_df[
                    (commits_df['repo'] == repo) &
                    (commits_df['date'] >= before_window_start) &
                    (commits_df['date'] <= comment_date)
                ]

                after_window_end = comment_date + pd.Timedelta(days=window_days)
                commits_after = commits_df[
                    (commits_df['repo'] == repo) &
                    (commits_df['date'] > comment_date) &
                    (commits_df['date'] <= after_window_end)
                ]

                commit_impact_data.append({
                    'repo': repo,
                    'issue_number': issue_number,
                    'comment_id': toxic_comment['comment_id'],
                    'comment_date': comment_date,
                    'toxicity': toxic_comment['toxicity'],
                    'commits_before': len(commits_before),
                    'commits_after': len(commits_after),
                    'commit_change_pct': ((len(commits_after) - len(commits_before)) / max(1, len(commits_before))) * 100
                })

            if commit_impact_data:
                results['commit_impact'] = pd.DataFrame(commit_impact_data)

        # (B) ISSUE RESOLITON - (time it takes to close issue, check toxicity compare if toxicity is correlated to resolition time )
            issue_resolution_data = []
            for _, issue in issues_df.iterrows():
                
                # only consider closed issues
                if pd.isna(issue['closed_at']):
                    continue  

                repo = issue['repo']
                issue_number = issue['issue_number']

                # Check if this issue had toxic comments
                issue_comments = comments_df[
                    (comments_df['repo'] == repo) &
                    (comments_df['issue_number'] == issue_number)
                ]
                has_toxic_comments = not issue_comments.empty and (
                    issue_comments['toxicity_score'] > TOXICITY_THRESHOLD
                ).any()

                # Calculate resolution time
                resolution_time = (issue['closed_at'] - issue['created_at']).total_seconds() / 3600  # in hours

                issue_resolution_data.append({
                    'repo': repo,
                    'issue_number': issue_number,
                    'created_at': issue['created_at'],
                    'closed_at': issue['closed_at'],
                    'resolution_time_hours': resolution_time,
                    'has_toxic_comments': has_toxic_comments,
                    'max_toxicity': issue_comments['toxicity_score'].max() if not issue_comments.empty else 0,
                    'comments_count': issue['comments_count']
                })

            if issue_resolution_data:
                results['issue_resolution'] = pd.DataFrame(issue_resolution_data)

    
        #(C) DISCUSSION ACTIVITY - for not developers
            developer_activity_data = []
            # issue solved, no more duplicate developers, usong set not list
            developers_with_toxic_exp = set()

            # for each toxic comment
            for _, comment in toxic_comments.iterrows():
                issue_num = comment['issue_number']
                repo = comment['repo']
                matching_issues = issues_df[
                    (issues_df['repo'] == repo) &
                    (issues_df['issue_number'] == issue_num)
                ]
                # get the username of all users involved in toxic comments
                if not matching_issues.empty:
                    developers_with_toxic_exp.add((repo, matching_issues.iloc[0]['user_login']))

            for repo, dev_login in developers_with_toxic_exp:
                # go thru each person and their toxic comments
                dev_toxic_comments = toxic_comments[
                    (toxic_comments['repo'] == repo) &
                    (comments_df['issue_number'].isin(
                        issues_df[(issues_df['repo'] == repo) &
                                (issues_df['user_login'] == dev_login)]['issue_number']
                    ))
                ]

                if dev_toxic_comments.empty:
                    continue # should not bethe case

                first_toxic_date = dev_toxic_comments['created_at'].min()
                last_toxic_date = dev_toxic_comments['created_at'].max()

                # Activity before first toxic interaction (30 days)
                before_period_start = first_toxic_date - pd.Timedelta(days=30)
                commits_before = commits_df[
                    (commits_df['repo'] == repo) &
                    (commits_df['author_login'] == dev_login) &
                    (commits_df['date'] >= before_period_start) &
                    (commits_df['date'] < first_toxic_date)
                ]
                issues_before = issues_df[
                    (issues_df['repo'] == repo) &
                    (issues_df['user_login'] == dev_login) &
                    (issues_df['created_at'] >= before_period_start) &
                    (issues_df['created_at'] < first_toxic_date)
                ]

                # Activity after last toxic interaction (30 days)
                after_period_end = last_toxic_date + pd.Timedelta(days=30)
                commits_after = commits_df[
                    (commits_df['repo'] == repo) &
                    (commits_df['author_login'] == dev_login) &
                    (commits_df['date'] > last_toxic_date) &
                    (commits_df['date'] <= after_period_end)
                ]
                issues_after = issues_df[
                    (issues_df['repo'] == repo) &
                    (issues_df['user_login'] == dev_login) &
                    (issues_df['created_at'] > last_toxic_date) &
                    (issues_df['created_at'] <= after_period_end)
                ]

                developer_activity_data.append({
                    'repo': repo,
                    'developer': dev_login,
                    'first_toxic_date': first_toxic_date,
                    'last_toxic_date': last_toxic_date,
                    'toxic_comments_received': len(dev_toxic_comments),
                    'commits_before': len(commits_before),
                    'commits_after': len(commits_after),
                    'issues_before': len(issues_before),
                    'issues_after': len(issues_after),
                    'commit_change_pct': ((len(commits_after) - len(commits_before)) / max(1, len(commits_before))) * 100,
                    'issue_change_pct': ((len(issues_after) - len(issues_before)) / max(1, len(issues_before))) * 100
                })

            if developer_activity_data:
                results['developer_activity'] = pd.DataFrame(developer_activity_data)
            
            return results
        except Exception as e:
            logger.error(f"Error in toxicity vs productivity analysis: {str(e)}")
            return results


    # 2) Is there any correlation between toxic communication and software releases? (Spearman correlation)
    def analyze_toxicity_vs_releases(self):
        # TODO: analyze toxicit against release
        logger.info("Analyzing toxicity vs releases")
        results = {}

        try:
            dfs = self.convert_to_dataframes()
            if 'comments' not in dfs or 'releases' not in dfs:
                logger.error("Required data missing for toxicity vs releases analysis")
                return results

            comments_df = dfs['comments']
            releases_df = dfs['releases']

            # same as other fxns 
            # GET toxicity scores
            comments_df['toxicity_score'] = comments_df['toxicity'].apply(
                lambda x: x['score'] if isinstance(x, dict) and 'score' in x else 0
            )

            if releases_df.empty:
                logger.warning("No release data available for analysis")
                return results

            # look into toxicity 2 weeks before/after reslease
            release_toxicity_data = []
            window_days = 14  

            for _, release in releases_df.iterrows():
                repo = release['repo']
                release_date = release['published_at']

                # before release
                before_window_start = release_date - pd.Timedelta(days=window_days)
                comments_before = comments_df[
                    (comments_df['repo'] == repo) &
                    (comments_df['created_at'] >= before_window_start) &
                    (comments_df['created_at'] < release_date)
                ]

                # after release
                after_window_end = release_date + pd.Timedelta(days=window_days)
                comments_after = comments_df[
                    (comments_df['repo'] == repo) &
                    (comments_df['created_at'] >= release_date) &
                    (comments_df['created_at'] <= after_window_end)
                ]

                # Calculate metrics to get rought picutre of toxicity around relaeses
                avg_toxicity_before = comments_before['toxicity_score'].mean() if not comments_before.empty else 0
                avg_toxicity_after = comments_after['toxicity_score'].mean() if not comments_after.empty else 0

                # max is mainly for testing since most repos have 1 toxic comment, the non-toxic drown the mean down
                max_toxicity_before = comments_before['toxicity_score'].max() if not comments_before.empty else 0
                max_toxicity_after = comments_after['toxicity_score'].max() if not comments_after.empty else 0

                # count of toxicity before / after release 
                toxic_comments_before = comments_before[comments_before['toxicity_score'] > TOXICITY_THRESHOLD]
                toxic_comments_after = comments_after[comments_after['toxicity_score'] > TOXICITY_THRESHOLD]

                # not neccessary, but can show pie chart or smthing showing the % increase/decrease of comments before/after
                toxic_pct_before = (len(toxic_comments_before) / len(comments_before) * 100) if not comments_before.empty else 0
                toxic_pct_after = (len(toxic_comments_after) / len(comments_after) * 100) if not comments_after.empty else 0

                release_toxicity_data.append({
                    'repo': repo,
                    'release_id': release['id'],
                    'tag_name': release['tag_name'],
                    'release_name': release['name'],
                    'release_date': release_date,
                    'comments_before': len(comments_before),
                    'comments_after': len(comments_after),
                    'avg_toxicity_before': avg_toxicity_before,
                    'avg_toxicity_after': avg_toxicity_after,
                    'max_toxicity_before': max_toxicity_before,
                    'max_toxicity_after': max_toxicity_after,
                    'toxic_comments_before': len(toxic_comments_before),
                    'toxic_comments_after': len(toxic_comments_after),
                    'toxic_pct_before': toxic_pct_before,
                    'toxic_pct_after': toxic_pct_after,
                    'toxicity_change': avg_toxicity_after - avg_toxicity_before
                })

            if release_toxicity_data:
                results['release_toxicity'] = pd.DataFrame(release_toxicity_data)

            # look into any toxicity trends over releases for each repo
            for repo in releases_df['repo'].unique():
                # for a repo, get the releases
                repo_releases = releases_df[releases_df['repo'] == repo].sort_values('published_at')

                # issue edge case, handled by making sure there was gurrentteed 2 releases in the repo
                if len(repo_releases) < 2:
                    continue  

                release_cycle_data = []
                for i in range(len(repo_releases) - 1):
                    current_release = repo_releases.iloc[i]
                    next_release = repo_releases.iloc[i + 1]

                    # Get comments between releases
                    cycle_comments = comments_df[
                        (comments_df['repo'] == repo) &
                        (comments_df['created_at'] >= current_release['published_at']) &
                        (comments_df['created_at'] < next_release['published_at'])
                    ]

                    # Calculate toxicity metrics so we can see if there was major toxicity changes
                    
                    avg_toxicity = cycle_comments['toxicity_score'].mean() if not cycle_comments.empty else 0
                    toxic_comments = cycle_comments[cycle_comments['toxicity_score'] > TOXICITY_THRESHOLD]
                    # - oercebt means decreased, + increaed, 0 same
                    toxic_pct = (len(toxic_comments) / len(cycle_comments) * 100) if not cycle_comments.empty else 0

                    # duraction of the release (time from release 1 to release 2)
                    cycle_duration_days = (next_release['published_at'] - current_release['published_at']).total_seconds() / (3600 * 24)

                    release_cycle_data.append({
                        'repo': repo,
                        'from_release': current_release['tag_name'],
                        'to_release': next_release['tag_name'],
                        'cycle_start': current_release['published_at'],
                        'cycle_end': next_release['published_at'],
                        'cycle_duration_days': cycle_duration_days,
                        'comments_count': len(cycle_comments),
                        'avg_toxicity': avg_toxicity,
                        'toxic_comments': len(toxic_comments),
                        'toxic_pct': toxic_pct
                    })

                if release_cycle_data:
                    if 'release_cycles' not in results:
                        results['release_cycles'] = pd.DataFrame(release_cycle_data)
                    else:
                        results['release_cycles'] = pd.concat([results['release_cycles'], pd.DataFrame(release_cycle_data)])

            return results


        except Exception as e:
            logger.error(f"Error in toxicity vs releases analysis: {str(e)}")
            return results

    
    # 3) How does the level of experience of the contributors (measured by the age of the account and previous contributions) correlate with their likelihood of engaging in toxic communication within OSS communities? (Spearman correlation)
    def analyze_experience_vs_toxicity(self):
        # can use spearman correlation to see if there is a correlation between the account age and toxic communication
        # age can be determined by their actual github age, or how long they were in the repo, not sure which is easier yet
        #
        logger.info("Analyzing experience vs toxicity")
        results = {}

        try:
            dfs = self.convert_to_dataframes()
            if 'comments' not in dfs or 'contributors' not in dfs:
                logger.error("Required data missing for experience vs toxicity analysis")
                return results

            comments_df = dfs['comments']
            contributors_df = dfs['contributors']
            
            # Extract toxicity score to a new column
            comments_df['toxicity_score'] = comments_df['toxicity'].apply(
                lambda x: x['score'] if isinstance(x, dict) and 'score' in x else 0
            )

            # go thru each contribuor (ALL contribuores, non toxic AND toxic comments)
            contributor_toxicity_data = []
            for _, contributor in contributors_df.iterrows():
                repo = contributor['repo']
                user_login = contributor['user_login']
                account_created_at = contributor['account_created_at']

                # Get all (all meaning the sample size) comments by this contributor
                user_comments = comments_df[
                    (comments_df['repo'] == repo) &
                    (comments_df['user_login'] == user_login)
                ]

                if user_comments.empty:
                    continue

                # baisc metric calc
                avg_toxicity = user_comments['toxicity_score'].mean()
                max_toxicity = user_comments['toxicity_score'].max()
                toxic_comments = user_comments[user_comments['toxicity_score'] > TOXICITY_THRESHOLD]
                toxic_pct = (len(toxic_comments) / len(user_comments) * 100)

                # For each comment, get theoir account age 
                for _, comment in user_comments.iterrows():
                    comment_date = comment['created_at']
                    account_age_days = (comment_date - account_created_at).total_seconds() / (3600 * 24)

                    contributor_toxicity_data.append({
                        'repo': repo,
                        'user_login': user_login,
                        'user_id': contributor['user_id'],
                        'comment_id': comment['comment_id'],
                        'comment_date': comment_date,
                        'account_created_at': account_created_at,
                        'account_age_days': account_age_days,
                        'toxicity_score': comment['toxicity_score'],
                        'public_repos': contributor['public_repos'],
                        'followers': contributor['followers'],
                        'contributions': contributor['contributions']
                    })

            if contributor_toxicity_data:
                results['contributor_toxicity'] = pd.DataFrame(contributor_toxicity_data)

                # Creating experience brackets for analysis, which in my opinion isnt necessary but gpt reccommened it and i think its nice to have (might be easier to make visuals from)
                exp_df = results['contributor_toxicity'].copy()
                exp_df['experience_bracket'] = pd.cut(
                    exp_df['account_age_days'],
                    bins=[0, 90, 365, 365 * 2, 365 * 5, float('inf')],
                    labels=['<3 months', '3-12 months', '1-2 years', '2-5 years', '>5 years']
                )

                # group by the experience bracket
                exp_summary = exp_df.groupby('experience_bracket').agg({
                    'toxicity_score': ['mean', 'std', 'count'],  
                    'user_login': 'nunique'
                })
                exp_summary.columns = ['avg_toxicity', 'std_toxicity', 'comment_count', 'unique_users']
                exp_summary = exp_summary.reset_index()
                results['experience_summary'] = exp_summary

                contrib_df = results['contributor_toxicity'].copy()
                contrib_df['contribution_bracket'] = pd.cut(
                    contrib_df['contributions'],
                    bins=[0, 10, 50, 100, 500, float('inf')],
                    labels=['<10', '10-50', '50-100', '100-500', '>500']
                )

                # gorup by the contribution level
                contrib_summary = contrib_df.groupby('contribution_bracket').agg({
                    'toxicity_score': ['mean', 'std', 'count'],  # Use toxicity_score instead of toxicity
                    'user_login': 'nunique'
                })
                contrib_summary.columns = ['avg_toxicity', 'std_toxicity', 'comment_count', 'unique_users']
                contrib_summary = contrib_summary.reset_index()
                results['contribution_summary'] = contrib_summary

                # first-time contributors versus experienced contributors
                first_time_data = []
                for repo in contributors_df['repo'].unique():
                    repo_contributors = contributors_df[contributors_df['repo'] == repo]

                    for _, contributor in repo_contributors.iterrows():
                        user_login = contributor['user_login']

                        # Get all comments by this user
                        user_comments = comments_df[
                            (comments_df['repo'] == repo) &
                            (comments_df['user_login'] == user_login)
                        ]

                        if user_comments.empty:
                            continue

                        # Determine if first-time contributor (low contribution count)
                        is_first_time = contributor['contributions'] <= 5

                        avg_toxicity = user_comments['toxicity_score'].mean()
                        max_toxicity = user_comments['toxicity_score'].max()
                        toxic_comments = user_comments[user_comments['toxicity_score'] > TOXICITY_THRESHOLD]
                        toxic_pct = (len(toxic_comments) / len(user_comments) * 100)

                        first_time_data.append({
                            'repo': repo,
                            'user_login': user_login,
                            'is_first_time': is_first_time,
                            'contributions': contributor['contributions'],
                            'followers': contributor['followers'],
                            'public_repos': contributor['public_repos'],
                            'comments_count': len(user_comments),
                            'avg_toxicity': avg_toxicity,
                            'max_toxicity': max_toxicity,
                            'toxic_comments': len(toxic_comments),
                            'toxic_pct': toxic_pct
                        })

                if first_time_data:
                    results['first_time_vs_experienced'] = pd.DataFrame(first_time_data)

            return results
        
        except Exception as e:
            logger.error(f"Error in experience vs toxicity analysis: {str(e)}")
            return results

    def save_results_to_csv(self, results_dict, base_filename):
        try:
            os.makedirs(os.path.dirname(base_filename), exist_ok=True)
            for key, df in results_dict.items():
                if df is not None and not df.empty:
                    filename = f"{base_filename}_{key}.csv"
                    df.to_csv(filename, index=False)
                    logger.info(f"Saved {len(df)} rows to {filename}")
        except Exception as e:
            logger.error(f"Error saving results to CSV: {str(e)}")

    # get data and save EACH to CSV
    def fetch_data(self):        
        logger.info("Starting incremental data fetching")
        
        for repo in self.repos:
            logger.info(f"Processing repository: {repo}")
            
            # Clear previous data for this repo
            for key in self.data:
                self.data[key] = [item for item in self.data[key] if item.get('repo') != repo]
            
            try:
                self._fetch_comments(repo)
                logger.info(f"Completed fetching comments for {repo}")
                
                self._fetch_commits(repo)
                logger.info(f"Completed fetching commits for {repo}")
                
                self._fetch_issues(repo)
                logger.info(f"Completed fetching issues for {repo}")
                
                self._fetch_releases(repo)
                logger.info(f"Completed fetching releases for {repo}")
                
                self._fetch_contributors(repo)
                logger.info(f"Completed fetching contributors for {repo}")
                
                # Save to CSV file, this will create ALOT of CSV files which is fine bc i need to see the data genreatedfor testing/debugging
                temp_dfs = self.convert_to_dataframes()
                for key, df in temp_dfs.items():
                    # Filter for just this repo
                    repo_df = df[df['repo'] == repo]
                    if not repo_df.empty:
                        os.makedirs(DATA_DIR, exist_ok=True)
                        filename = f"{DATA_DIR}/{repo.replace('/', '_')}_{key}.csv"
                        repo_df.to_csv(filename, index=False)
                        logger.info(f"Saved {len(repo_df)} {key} rows for {repo}")
                
                # Run analysis just for this repo
                logger.info(f"Running analysis for {repo}")
                
                # Filter data to only include this repo
                repo_data = {key: [item for item in items if item.get('repo') == repo] 
                            for key, items in self.data.items()}
                
                # temp analyzer to hold curr repo data
                temp_analyzer = GitHubToxicityAnalyzer([repo], self.github_tokens, test_mode=False)
                temp_analyzer.data = repo_data
                
                # Run and save analyses
                toxicity_vs_productivity = temp_analyzer.analyze_toxicity_vs_productivity()
                temp_analyzer.save_results_to_csv(toxicity_vs_productivity, f"{RESULTS_DIR}/{repo.replace('/', '_')}_toxicity_vs_productivity")
                
                toxicity_vs_releases = temp_analyzer.analyze_toxicity_vs_releases()
                temp_analyzer.save_results_to_csv(toxicity_vs_releases, f"{RESULTS_DIR}/{repo.replace('/', '_')}_toxicity_vs_releases")
                
                experience_vs_toxicity = temp_analyzer.analyze_experience_vs_toxicity()
                temp_analyzer.save_results_to_csv(experience_vs_toxicity, f"{RESULTS_DIR}/{repo.replace('/', '_')}_experience_vs_toxicity")
                
                logger.info(f"Completed all processing for {repo}")

            except Exception as e:
                logger.error(f"Error processing repository {repo}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info("Incremental data fetching and analysis complete")

        # after all repos are processed, combine results
        try:
            # this is where the "nice" data will be with ALL repos instead of individual
            self._combine_results()
        except Exception as e:
            logger.error(f"Error combining results: {str(e)}")


    def _combine_results(self):
        logger.info("Combining results from all repositories")
        
        # make folders if not there
        data_dir = "data_combined"
        results_dir = "results"
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Combine raw data files
        data_types = ['comments', 'commits', 'issues', 'releases', 'contributors']
        for data_type in data_types:
            combined_df = None
            pattern = f"*_{data_type}.csv"
            files = glob.glob(os.path.join(data_dir, pattern))
            
            for file in files:
                try:
                    df = pd.read_csv(file)
                    if combined_df is None:
                        combined_df = df
                    else:
                        combined_df = pd.concat([combined_df, df], ignore_index=True)
                except Exception as e:
                    logger.error(f"Error reading {file}: {str(e)}")
            
            if combined_df is not None and not combined_df.empty:
                combined_df.to_csv(os.path.join(data_dir, f"combined_{data_type}.csv"), index=False)
                logger.info(f"Created combined data file for {data_type} with {len(combined_df)} rows")
        
        # Combine analysis results
        analysis_types = [
            "toxicity_vs_productivity_commit_impact",
            "toxicity_vs_productivity_issue_resolution",
            "toxicity_vs_productivity_developer_activity",
            
            "toxicity_vs_releases_release_toxicity",
            "toxicity_vs_releases_release_cycles",
            
            "experience_vs_toxicity_contributor_toxicity",
            "experience_vs_toxicity_experience_summary",
            "experience_vs_toxicity_contribution_summary",
            "experience_vs_toxicity_first_time_vs_experienced"
        ]
        
        for analysis_type in analysis_types:
            combined_df = None
            pattern = f"*_{analysis_type}.csv"
            files = glob.glob(os.path.join(results_dir, pattern))
            
            for file in files:
                try:
                    df = pd.read_csv(file)
                    if combined_df is None:
                        combined_df = df
                    else:
                        combined_df = pd.concat([combined_df, df], ignore_index=True)
                except Exception as e:
                    logger.error(f"Error reading {file}: {str(e)}")
            
            if combined_df is not None and not combined_df.empty:
                combined_df.to_csv(os.path.join(results_dir, f"combined_{analysis_type}.csv"), index=False)
                logger.info(f"Created combined results file for {analysis_type} with {len(combined_df)} rows")
