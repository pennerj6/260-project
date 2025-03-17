import pandas as pd
import numpy as np

import time
import requests
import os
import logging
import random
import glob
from datetime import datetime, timedelta

from concurrent.futures import ThreadPoolExecutor, as_completed # this speeds up our code BUT, im pretty sure our github api limit will get used more (thats ok tho, sahil met with TA he said we dont need to anaylze as much repos as we thought)

from toxicity_rater import ToxicityRater
from helper import * #get_issue_number
from config import *

import scipy.stats as stats
from scipy.stats import spearmanr

from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)


# =======================================================================================================================================
def get_data_main(repos):
    '''
    GLOBAL VARIABLES 
    GLOBAL VARIABLES 
    GLOBAL VARIABLES 
    GLOBAL VARIABLES 
    '''
    global github_tokens
    global current_token_index
    global all_data
    global curr_data
    global analysis_window_days
    global GITHUB_GITHUB_BASE_URL
    global rater

    # Get GitHub tokens from env to be cycled (somewhat of a solution to the rate limit issue)
    # idk why but i need to call each variable separtely liek this otherwise theres a parsing error
    x = os.getenv('GITHUB_ACCESS_TOKEN_1')
    y = os.getenv('GITHUB_ACCESS_TOKEN_2')
    z = os.getenv('GITHUB_ACCESS_TOKEN_3')
    github_tokens = [x,y,z]

    current_token_index = 0
    all_data = {
        'comments': [],
        'commits': [],
        'issues': [],
        'releases': [],
        'contributors': []
    }

    curr_data = {
        'comments': [],
        'commits': [],
        'issues': [],
        'releases': [],
        'contributors': []
    }

    analysis_window_days = 30


    GITHUB_GITHUB_BASE_URL = "https://api.github.com"

    rater = ToxicityRater()  # Fixed typo in sample_size (was 10a000)
    # rater = ToxicityRater(use_sampling=True, sample_size=10000)  # Fixed typo in sample_size (was 10a000)

    '''
    GLOBAL VARIABLES 
    GLOBAL VARIABLES 
    GLOBAL VARIABLES 
    GLOBAL VARIABLES 
    '''
    
    # =======================================================================================================================================

    # Switch to next api token (professor mentioned this way of getting by the github rate limit early in the quarter, like week 2)
    def switch_api_token():
        global github_tokens
        global current_token_index
        global all_data
        global curr_data
        global analysis_window_days
        global GITHUB_GITHUB_BASE_URL
        global rater

        # switch tokens index around to next one in the cycle
        current_token_index = (current_token_index + 1) % len(github_tokens)    
        curr_token = github_tokens[current_token_index]

        headers = {
            'Authorization': f'Bearer {curr_token}',
            'Accept': 'application/vnd.github.v3+json'
        }

        logger.info(f"Rotated to token index: {current_token_index}")

        # return the new header which gets used ever time we connect to github api
        return headers

    def github_api_request(url):
        global github_tokens
        global current_token_index
        global all_data
        global curr_data
        global analysis_window_days
        global GITHUB_GITHUB_BASE_URL
        global rater
        headers = switch_api_token()
        
        # Kept having problems with this function so had to rewrite it a few times, gpt helped w debingg
        # TODO refactor this if I have time (dont rlly need to tbh)
        
        # given link, return the resonse associated (could be for issues page, comments, etc..)
        # need to rotate the api tokens (i made 3 for me, going to use gpt to help implement api rotation)
        current_token = github_tokens[current_token_index]
        
        # TODO: instead of 3moths ago, we should probly have it from the created date of the toxic comment in all_roxic_repos.csv
        # three_months_ago = datetime.datetime.now() - datetime.timedelta(days=90)
        three_months_ago = datetime.now() - timedelta(days=90)

        params = {
            'per_page' : 100, # 100 issues (or comments...) perpage
            # 'since': three_months_ago
        }

        all_items = []
        retry_count = 0
        max_retries = 3
        base_delay = 1  

        while url and len(all_items) < 1000:
            try:
                response = requests.get(url, headers=headers, params=params, timeout=30)
                # Rate limit check - switch tokens or wait if needed
                if response.status_code == 403 and 'rate limit exceeded' in response.text.lower():
                    retry_count += 1
                    if retry_count <= max_retries:
                        delay = base_delay * (2 ** (retry_count - 1))
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
                # all_items.extend(items)
                if isinstance(items, list):
                    all_items.extend(items)
                else:
                    all_items.append(items)

                if 'next' in response.links:
                    url = response.links['next']['url']
                    params = {}  # should i comment this out?
                else:
                    break

                remaining = int(response.headers.get('X-RateLimit-Remaining', '1'))
                if remaining < RATE_LIMIT_THRESHOLD:
                    headers = switch_api_token()

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

    def get_comments(repo):
        global github_tokens
        global current_token_index
        global all_data
        global curr_data
        global analysis_window_days
        global GITHUB_GITHUB_BASE_URL
        global rater
        try:
            # params = {'per_page': 100} # not sure what happends if i change to 1000 (gpt said this means it gets 100 per api request, to not miss out on data i have to implement osmething called 'pagintation' TODO will look into later)
            
            # Make hte comment links in variables 
            issue_comments_url = f"{GITHUB_BASE_URL}/repos/{repo}/issues/comments"
            pr_comments_url = f"{GITHUB_BASE_URL}/repos/{repo}/pulls/comments" # same as issues, just request w pulls/comments

            # use github api to return the response of whatever the link gives (comments for issues && pr)
            issue_comments = github_api_request(issue_comments_url) # this should return samplesize(1000 i think) comments or less
            pr_comments = github_api_request(pr_comments_url)  
            
            # Merge the issue and pr comments
            all_comments = issue_comments + pr_comments
            
            # For each comment, process it into hasmap 
            curr_data['comments'] = []
            for comment in all_comments:
                c_type = ''
                issue_num = 0
                if 'issue_url' in comment: 
                    c_type = 'issue_comment' 
                    issue_num = get_issue_number(comment['issue_url'])
                elif 'pull_request_url' in comment: 
                    c_type = 'pr_comment' 
                    issue_num = get_issue_number(comment['pull_request_url'])
            
                
                # kept getting error here and gpt resolved it w this. 
                # BUT dont these 2 things mean the same thing? (like cant we just do 'if comment['body])
                if 'body' in comment and comment['body']:   
                    try:
                        # get the toxicity score of the comment
                        toxicity = rater.get_toxicity(comment['body'])

                        comment_data = {
                            'repo': repo,
                            'comment_id': comment['id'],
                            'user_id': comment['user']['id'],
                            'user_login': comment['user']['login'],
                            'created_at': comment['created_at'],
                            'updated_at': comment.get('updated_at'),
                            'body': comment['body'],
                            'toxicity': toxicity,
                            'type': c_type,
                            'issue_number': issue_num
                        }
                        curr_data['comments'].append(comment_data)
                        
                        all_data['comments'].append(comment_data)

                    except Exception as e:
                        logger.error(f"Error calculating toxicity for comment {comment['id']}: {e}")

            
            logger.info(f"Fetched {len(all_comments)} comments for repository: {repo}")

        except Exception as e:
            logger.error(f"Error fetching comments for repository {repo}: {e}")

    def get_commits(repo):
        global github_tokens
        global current_token_index
        global all_data
        global curr_data
        global analysis_window_days
        global GITHUB_GITHUB_BASE_URL
        global rater
        try:
            # Make the commits URL
            commits_url = f"{GITHUB_BASE_URL}/repos/{repo}/commits"
            
            # Use GitHub API to fetch commits
            commits = github_api_request(commits_url)
            
            # For each commit, process it into hashmap
            curr_data['commits'] = []
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
                    curr_data['commits'].append(commit_data)
                    
                    all_data['commits'].append(commit_data)
                    
            logger.info(f"Fetched {len(commits)} commits for repository: {repo}")
            
        except Exception as e:
            logger.error(f"Error fetching commits for repository {repo}: {e}")

    def get_issues(repo):
        global github_tokens
        global current_token_index
        global all_data
        global curr_data
        global analysis_window_days
        global GITHUB_GITHUB_BASE_URL
        global rater

        try:
            # Make the issues URL
            issues_url = f"{GITHUB_BASE_URL}/repos/{repo}/issues"
            
            # Use GitHub API to fetch issues
            issues = github_api_request(issues_url)
            
            # For each issue, process it into hashmap
            curr_data['issues'] = []
            for issue in issues:
                # Don't want issue data from PR
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
                curr_data['issues'].append(issue_data)
                
                all_data['issues'].append(issue_data)
                
            logger.info(f"Fetched {len(issues)} issues for repository: {repo}")
            
        except Exception as e:
            logger.error(f"Error fetching issues for repository {repo}: {e}")

    def get_releases(repo):
        global github_tokens
        global current_token_index
        global all_data
        global curr_data
        global analysis_window_days
        global GITHUB_GITHUB_BASE_URL
        global rater        
        try:
            # Make the releases URL
            releases_url = f"{GITHUB_BASE_URL}/repos/{repo}/releases"
            
            # Use GitHub API to fetch releases
            releases = github_api_request(releases_url)
            
            # For each release, process it into hashmap
            curr_data['releases'] = []
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
                curr_data['releases'].append(release_data)
            
                all_data['releases'].append(release_data)
                
            logger.info(f"Fetched {len(releases)} releases for repository: {repo}")
            
        except Exception as e:
            logger.error(f"Error fetching releases for repository {repo}: {e}")

    def get_contributors(repo):
        global github_tokens
        global current_token_index
        global all_data
        global curr_data
        global analysis_window_days
        global GITHUB_GITHUB_BASE_URL
        global rater
        try:
            # Make the contributors URL
            contributors_url = f"{GITHUB_BASE_URL}/repos/{repo}/contributors"
            
            # Use GitHub API to fetch contributors
            contributors = github_api_request(contributors_url)
            
            # For each contributor, process it into hashmap
            curr_data['contributors'] = []
            for contributor in contributors:
                try:
                    x = contributor['login']
                    # Make the user URL for additional user data
                    user_url = f"{GITHUB_BASE_URL}/users/{x}"
                    # user_url = f"{GITHUB_BASE_URL}/users/{contributor}"
                    
                    # Fetch user data
                    user_data = github_api_request(user_url)[0]
                    print(user_data)
                    
                    contributor_data = {
                        'repo': repo,
                        'user_id': contributor['id'],
                        'user_login': contributor['login'],
                        'contributions': contributor['contributions'],
                        'account_created_at': user_data['created_at'], # user_data.get('created_at'),
                        'public_repos': user_data['public_repos'], #user_data.get('public_repos', 0),
                        'followers': user_data['followers']#user_data.get('followers', 0)
                    }
                    curr_data['contributors'].append(contributor_data)
                    
                    all_data['contributors'].append(contributor_data)
                    
                except Exception as e:
                    logger.error(f"Error processing contributor {contributor['login']}: {e}")
                    continue
                    
            logger.info(f"Fetched {len(contributors)} contributors for repository: {repo}")
            
        except Exception as e:
            logger.error(f"Error fetching contributors for repository {repo}: {e}")



    ''' 
    UPDATED RQ's after sahil went to office hours (big help with #2, overall before the questions i made we good, but he just helped gather thoughts and reprahse to better/clear way)

        1) Does toxic communication in OSS communities negatively affect programmer productivity, measured through commits, issue resolutions, and discussion activity?

        2) Is there any correlation between toxic communication and software releases? (Spearman correlation mentined in OH)

        3) How does the level of experience of the contributors (measured by the age of the account and previous contributions) correlate with their likelihood of engaging in toxic communication within OSS communities? (Spearman correlation)
    '''




    # get data and save EACH to CSV
    def fetch_data(repos): 
        global github_tokens
        global current_token_index
        global all_data
        global curr_data
        global analysis_window_days
        global GITHUB_GITHUB_BASE_URL
        global rater               
        for repo in repos:
            logger.info(f"Getting data for repo: {repo}")
            try:            
                '''
                For each repo need to get (sample size to save computation al time, which is ok TA said just need to mention it):
                    A) comments
                    B) commits
                    C) issues
                    D) releases
                    E) Contributors

                    the fxns below arent foing to return anythign, they will just append to this gobale variable
                    all_data = {            all the data will get appended here for all repos
                        'comments': [],
                        'commits': [],
                        'issues': [],
                        'releases': [],
                        'contributors': []
                        }

                        curr_data = {                   this only contains data for the CURR repo
                            'comments': [],
                            'commits': [],
                            'issues': [],
                            'releases': [],
                            'contributors': []
                        }


                '''

                # A) comments
                get_comments(repo)
                
                # B) commits
                get_commits(repo)
                
                # C) issues
                get_issues(repo)
                
                # D) releases
                get_releases(repo)
                
                # E) Contributors
                get_contributors(repo)

                
                for file in all_data:
                    
                    # file -> comments , [{}, {},{}]
                    
                    save_csv(file, all_data[file])
                    
                print("")
            except Exception as e:
                logger.error(f"Error processing repository {repo}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info("All repos data fetched and is sotred in global hashmap data")

        # after all repos are processed, combine results
        # try:
        #     # this is where the "nice" data will be with ALL repos instead of individual
        #     mash_data()
        # except Exception as e:
        #     logger.error(f"Error combining results: {str(e)}")



    # originally, in main.py I ran fetch_data(repos) and that ideally would executed everything in get_data.py
    # but, i did not know the global variables will not be recognized (since they were not called on)
    # so i just put all of the code & global variables in 1 GIANT "main" function
    # so in main.py I call get_data_main and it runs this file as if i run the whole file not jsut the fxn so global var are processed
    fetch_data(repos)