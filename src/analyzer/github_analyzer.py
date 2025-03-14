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


