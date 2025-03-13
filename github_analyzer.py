import os
import gc
import json
import time
import logging
import datetime
import requests
import numpy as np
import pandas as pd
import dask.dataframe as dd
from io import BytesIO
from collections import defaultdict
from tqdm import tqdm
import gzip
import traceback
from config import DEFAULT_CONFIG, INITIAL_REQUIRED_COLUMNS, REQUIRED_COLUMNS
from toxicity_rater import ToxicityRater
import psutil


logger = logging.getLogger(__name__)


# Instead of using Github API directly for data and rinning into api rate limit issues
#Get data from a GH Archive, which has the data stored already, i just need:
#  fetch it 
#  clean it (remove any data that is hard to work with like missing,null values, etc)
#  make calculations to answer RQ 1 to 3 (like correlation between variablwes)
class GitHubArchiveAnalyzer:
    def __init__(self, start_date, end_date, toxicity_threshold=0.5):
        self.start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        self.toxicity_threshold = toxicity_threshold

        # Data storage
        self.comments_data = []
        self.commits_data = []
        self.issues_data = []
        self.releases_data = []
        self.repos_data = {}
        self.users_data = {}

        # Maps for relationships
        self.issue_to_comments = defaultdict(list)
        self.repo_to_issues = defaultdict(list)
        self.repo_to_commits = defaultdict(list)
        self.repo_to_releases = defaultdict(list)

        # GitHub API token which we do NOT use anymore bc of GHArchice
        # self.github_token = env.loadenv('GITHUB_TOKEN', '')

        # someone reccommend caching for API calls, which wont be needed anymore but i will keep it here if we implement something later
        self.user_cache = {}
        self.cache_file = os.path.join(DEFAULT_CONFIG['output_dir'], 'user_cache.json')
        self._load_user_cache()

    def _load_user_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.user_cache = json.load(f)
                logger.info(f"Loaded user cache with {len(self.user_cache)} entries")
            except Exception as e:
                logger.error(f"Error loading user cache: {str(e)}")
                self.user_cache = {}

    def _save_user_cache(self):
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.user_cache, f)
            logger.info(f"Saved user cache with {len(self.user_cache)} entries")
        except Exception as e:
            logger.error(f"Error saving user cache: {str(e)}")

    def generate_urls(self):
        """get all URLs for GHArchive given the start/end date"""
        urls = []
        current_date = self.start_date
        hours = DEFAULT_CONFIG['hours']
        
        # for each date in the range in config
        #   we will generate urls for each hour for each date
        while current_date <= self.end_date:
            # Only generate URL for the specified hour (only 1 like "7")
            # Generate URLs for all hours (0 to 24)
            # NOW we only look at hours between 6am to 10pm (i think, its in config)
            for hour in hours:
                date_str = current_date.strftime('%Y-%m-%d')
                url = f"https://data.gharchive.org/{date_str}-{hour}.json.gz"
                urls.append(url)
            current_date += datetime.timedelta(days=1)
        return urls

    
    def fetch_and_process_data(self, url):
        """Fetch and process data from only ONE GitHub Archive URL at a time"""
        try:
            # before, i was doing 2 print statements (1 for the current date/time and another for the variable i wanted to print. this was to see how long each chunk takes to load, funnily enough i used gpt for helping debug code and it did it in secodns (hence all the logger statements not print lol)
            logger.info(f"Fetching {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            logger.info(f"Fetched {url} successfully.")

            # was reccommeded to fetch data in CHUNKS to vaoid the application memeory error issue
            chunks = []
            record_count = 0
            filtered_count = 0
            sample_size = DEFAULT_CONFIG['sample_size'] # 1000 for now
            
            with gzip.GzipFile(fileobj=BytesIO(response.content)) as f:
                for line in f:
                    try:
                        # Limit how many records we will actually look at to save TIME and my patience
                        if record_count >= sample_size:
                            logger.info(f"Reached sample_size limit of {sample_size} records.")
                            break

                        record = json.loads(line)
                        record_count += 1

                        # Flatten the record to be able to access comment aka body like this "payload.comment.body"
                        flattened_record = self._flatten_dict(record)

                        # CLEANing, only keep the data that is good (no missing columns, there are comments, etc..)
                        if 'type' in flattened_record and flattened_record['type'] in ['IssueCommentEvent', 'PushEvent', 'IssuesEvent', 'ReleaseEvent']:
                            # Check if the current record has all required fields based onthe type
                            is_valid = True

                            # All events should have these basic fields 
                            if not all(key in flattened_record for key in ['id', 'type', 'created_at', 'actor.login']):
                                is_valid = False

                            # IssueCommentEvent, make sure there is a comment (body)
                            if is_valid and flattened_record['type'] == 'IssueCommentEvent':
                                # TODO: mayber we can check the toxicity score here??
                                # and drop it if it is below hte toxicity threshold
                                if 'payload.comment.body' not in flattened_record:
                                    is_valid = False

                            # IssuesEvent, make sure there is a issue # and action (i think action is a dictionary w other fields within)
                            if is_valid and flattened_record['type'] == 'IssuesEvent':
                                if not ('payload.issue.number' in flattened_record and
                                        'payload.action' in flattened_record):
                                    is_valid = False

                            # PushEvent, make sure REPO NAME is there
                            if is_valid and flattened_record['type'] == 'PushEvent':
                                if 'repo.name' not in flattened_record:
                                    is_valid = False

                            # if it is a valid record, add it to the batch of data to be processed later
                            if is_valid:
                                # Filter the record to keep only the required columns


                                    # INITIAL_REQUIRED_COLUMNS = [
                                    #     'id', 'type', 'created_at', 'actor.login', 'payload.comment.body', 
                                    #     'payload.issue.number', 'payload.action', 'repo.name'
                                    # ]
                                    # TOXICITY_COLUMNS = [
                                    #     'toxicity_score', 'is_toxic'

                                '''
                                    INITIAL_REQUIRED_COLUMNS = ['name', 'age', 'city']
                                    flattened_record = {'name': 'Alice', 'city': 'Wonderland'}
                                    filtered_record = {key: flattened_record[key] for key in INITIAL_REQUIRED_COLUMNS if key in flattened_record}
                                        returns->{'name': 'Alice', 'city': 'Wonderland'}
                                    For the LOC after this print block, i was checking to see if ALL the INITIAL_REQUIRED_COLUMNS needed to be there
                                    they do not, which is what we want. we only require 1 to be there which is what the code does
                                '''
                                filtered_record = {key: flattened_record[key] for key in INITIAL_REQUIRED_COLUMNS if key in flattened_record}
                                
                                chunks.append(filtered_record)
                                filtered_count += 1

                            
                            # Process in batches to avoid memory error (gpt helped w batching/errors w mem)
                            if len(chunks) >= 5000:
                                chunk_df = pd.DataFrame(chunks)
                                yield chunk_df
                                chunks = []
                                # garbage collection, had to look it up, it frees memory up a bit which helps the issue i had a bit
                                gc.collect()

                    except Exception as e:
                        logger.error(f"Error parsing line: {str(e)[:100]}...")
                        continue

                # Process any remaining records
                if chunks:
                    chunk_df = pd.DataFrame(chunks)
                    yield chunk_df
            
            # its alright if we dont process EVERYTHING, theres more than enough data to where that doesnt matter
            # also, i go to OH frequently and asked that question at least 3 times lol and they said its ok for use to drop data/outliers/etc... (so long as we mention it breifly)
            logger.info(f"Processed {record_count} records, kept {filtered_count} valid records from {url}.")

        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            yield pd.DataFrame()  # Return empty DataFrame on error

    # this was something i found that flattens a dict to the structure we want (payload.comment.body was the specific issue i had, but i am sure there will be more instances later)    
    def _flatten_dict(self, d, parent_key='', sep='.'):
        """Given a nested dict flatten it"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    

    def collect_data(self):
        """Collect data from GHArchive"""
        raw_data_path = os.path.join(DEFAULT_CONFIG['output_dir'], 'raw_data_chunks')
        os.makedirs(raw_data_path, exist_ok=True)

        # took the idea of "caching" so i dont have to reload the data EVERYTIME
        # NOTE, delete the "output" file everytime if you make changes to the code
        # just so it can sync with your changes
        if os.path.exists(os.path.join(raw_data_path, '_SUCCESS')):
            logger.info("Loading pre-processed raw data chunks...")
            self.ddf = dd.read_csv(os.path.join(raw_data_path, '*.csv'))  # Read all CSV files
            # Ensure required columns are present
            missing_columns = [col for col in INITIAL_REQUIRED_COLUMNS if col not in self.ddf.columns]
            if missing_columns:
                logger.warning(f"Missing required columns in pre-processed data: {missing_columns}. Dropping records.")
                self.ddf = self.ddf.dropna(subset=INITIAL_REQUIRED_COLUMNS)
            print("DEBUG: Actual columns in self.ddf:")
            print(self.ddf.columns.tolist())
        else:
            logger.info("Fetching/processing the raw data in chunks")
            urls = self.generate_urls()

            # Process URLs in smaller batches to avoid the memory issues i was having
            chunk_size = DEFAULT_CONFIG['chunk_size']  # Process N URLs at a time
            
            for i in range(0, len(urls), chunk_size):
                url_batch = urls[i:i + chunk_size]
                logger.info(f"Processing URL batch {i // chunk_size + 1}/{(len(urls) + chunk_size - 1) // chunk_size}")

                batch_dfs = []

                # Processs each url in the batch sequentially to avoid meme spikes (given chunk of data, we are processing each chunk in batches)
                for url in url_batch:
                    for chunk_df in self.fetch_and_process_data(url):
                        if not chunk_df.empty:
                            # Drop records with missing required columns (use INITIAL_REQUIRED_COLUMNS) # in OH said its ok to do so
                            chunk_df = chunk_df.dropna(subset=INITIAL_REQUIRED_COLUMNS)
                            batch_dfs.append(chunk_df)

                    # garbage collection frees up memory 
                    gc.collect()

                # Combine batch results and save to disk
                if batch_dfs:
                    try:
                        batch_df = pd.concat(batch_dfs, ignore_index=True)
                        batch_df.to_csv(os.path.join(raw_data_path, f'batch_{i // chunk_size}.csv'), index=False)

                        # clear memory
                        del batch_dfs, batch_df
                        gc.collect()
                    except Exception as e:
                        logger.error(f"Error saving batch data: {str(e)}")

                # Create _SUCCESS flag file (idk what the point of this gpt made success file) # can we not just do a print/logger statment?
                with open(os.path.join(raw_data_path, '_SUCCESS'), 'w') as f:
                    f.write('Success')

            # Load the saved chunks
            try:
                self.ddf = dd.read_csv(os.path.join(raw_data_path, '*.csv'))  # Read all CSV files
                
                # Droping records with missing required columns (use INITIAL_REQUIRED_COLUMNS since the other columns arent created here (toxiciity) )
                self.ddf = self.ddf.dropna(subset=INITIAL_REQUIRED_COLUMNS)
                if 1==0:
                    print("DEBUG: Actual columns in self.ddf:")
                    print(self.ddf.columns.tolist())

                    logger.info(f"Data collection complete. Total partitions: {self.ddf.npartitions}")
                    print("Sample of Main DataFrame:")
                    print(self.ddf.head())
                    print("Columns in DataFrame:")
                    print(self.ddf.columns)  # Print all columns in the DataFrame
            
            except Exception as e:
                logger.error(f"Error loading processed data: {str(e)}")
                # Fallback to create an empty DataFrame with expected schema
                self.ddf = dd.from_pandas(pd.DataFrame(columns=INITIAL_REQUIRED_COLUMNS), npartitions=1)
                
    
    def process_toxicity(self, rater, batch_size=32):
        """Process toxicity scores for comments"""
        toxicity_scores_path = os.path.join(DEFAULT_CONFIG['output_dir'], 'toxicity_scores.csv')

        # if the "cache" csv exists then use that
        if os.path.exists(toxicity_scores_path):
            # note, we only ever care about if it exists already for debugging the visulaizations code (since that changes frequenly but the data is the same)
            logger.info("Loading toxicity scores from FOLDER")
            try:
                toxicity_scores_df = pd.read_csv(toxicity_scores_path)
                # dont need to do anything since it exists already 
                # print("Sample of Toxicity Scores:")
                # print(toxicity_scores_df.head())

            except Exception as e:
                logger.error(f"Error loading toxicity scores: {str(e)}")
        else:
            logger.info("Computing toxicity scores")

            try:
                meta = pd.DataFrame({
                    'id': pd.Series(dtype='object'),
                    'type': pd.Series(dtype='object'),
                    'created_at': pd.Series(dtype='object'),
                    'actor.login': pd.Series(dtype='object'),
                    'payload.comment.body': pd.Series(dtype='object'),
                    'payload.issue.number': pd.Series(dtype='float64'),
                    'payload.action': pd.Series(dtype='object'),
                    'repo.name': pd.Series(dtype='object')
                })

                def filter_comments(df):
                    df = df.dropna(subset=INITIAL_REQUIRED_COLUMNS)
                    # sort it to the issue comments
                    df = df[df['type'] == 'IssueCommentEvent']
                    df = df[meta.columns] 
                    return df

                filtered_ddf = self.ddf.map_partitions(filter_comments, meta=meta)
                comment_events = filtered_ddf.compute()

                if not comment_events.empty:
                    comments = comment_events['payload.comment.body'].fillna('').tolist()

                    if comments:
                        available_mem = psutil.virtual_memory().available   # kept having mem issues, AI gave me this work around for finding the optimal chuincl size
                        chunk_size = min(5000, max(1000, int(available_mem / (5 * 1024 * 1024))))
                        logger.info(f"Processing comments in chunks of {chunk_size}")

                        all_scores = []
                        for i in range(0, len(comments), chunk_size):
                            chunk = comments[i:i + chunk_size]
                            logger.info(f"Processing comment chunk {i // chunk_size + 1}/{(len(comments) + chunk_size - 1) // chunk_size}")
                            # get toxicity score in batches
                            chunk_scores = rater.get_toxicity_ratings(chunk, batch_size=batch_size)
                            all_scores.extend(chunk_scores)

                            gc.collect()

                        toxicity_df = pd.DataFrame({
                            'id': comment_events['id'],
                            'toxicity_score': all_scores
                        })
                        # if the toxicity score is >= threshold, mark bool as True for is_toxic (this might not be needed if we are going to force data to be threshold or above aka all toxic data)
                        toxicity_df['is_toxic'] = toxicity_df['toxicity_score'] >= self.toxicity_threshold

                        # laod to CSV
                        toxicity_df.to_csv(toxicity_scores_path, index=False)

                        toxicity_dict = dict(zip(toxicity_df['id'], zip(toxicity_df['toxicity_score'], toxicity_df['is_toxic'])))

                        def add_toxicity(df):
                            df = df.copy()
                            df['toxicity_score'] = 0.0
                            df['is_toxic'] = False

                            for idx, row in df.iterrows():
                                if row.get('id') in toxicity_dict:
                                    score, is_toxic = toxicity_dict[row['id']]
                                    df.at[idx, 'toxicity_score'] = score
                                    df.at[idx, 'is_toxic'] = is_toxic

                            return df

                        self.ddf = self.ddf.map_partitions(add_toxicity)

                        logger.info("Toxicity processing completed successfully")
                    else:
                        logger.warning("No comments found to process.")
                        self.ddf = self.ddf.assign(toxicity_score=0.0)
                        self.ddf = self.ddf.assign(is_toxic=False)
                else:
                    logger.warning("No comment events found. Skipping toxicity processing.")
                    self.ddf = self.ddf.assign(toxicity_score=0.0)
                    self.ddf = self.ddf.assign(is_toxic=False)
            except Exception as e:
                logger.error(f"Error during toxicity processing: {str(e)}")
                logger.error(traceback.format_exc())
                self.ddf = self.ddf.assign(toxicity_score=0.0)
                self.ddf = self.ddf.assign(is_toxic=False)



                            
    # gpt reccommende we do something like this
    # we have the username so we can use github API to get their account AGE details
    # MIGHT consider this so i will leave it commented, CONS: this will be slow computationally i think (and rate limit  issues)
    '''
    def _fetch_user_account_age(self, user):
       """Fetch account age for a GitHub user with caching."""
        # Check cache first
        if user in self.user_cache:
            return self.user_cache[user]

        # If not in cache and we have a token, fetch from API
        if self.github_token:
            try:
                headers = {'Authorization': f'token {self.github_token}'}
                response = requests.get(f'https://api.github.com/users/{user}', headers=headers)
                if response.status_code == 200:
                    user_data = response.json()
                    created_at = user_data.get('created_at', '')
                    if created_at:
                        account_age = (datetime.datetime.now() - datetime.datetime.strptime(created_at, '%Y-%m-%dT%H:%M:%SZ')).days
                        # Cache the result
                        self.user_cache[user] = account_age
                        self._save_user_cache()
                        return account_age
            except Exception as e:
                logger.error(f"Error fetching user data for {user}: {str(e)[:100]}...")

        # Default value if we can't get the real age
        return 0
    '''
    def analyze_toxicity_productivity_correlation(self):
        """Analyze how toxic communication affects productivity."""
        productivity_path = os.path.join(DEFAULT_CONFIG['output_dir'], 'productivity_analysis.csv')

        if os.path.exists(productivity_path):
            logger.info("Loading productivity analysis results from disk...")
            try:
                productivity_df = pd.read_csv(productivity_path)
            except Exception as e:
                logger.error(f"Error loading productivity analysis: {str(e)}")
                productivity_df = pd.DataFrame()
        else:
            logger.info("Analyzing toxicity-productivity correlation...")

            try:
                columns = ['type', 'created_at', 'payload.issue.number', 'is_toxic']
                available_columns = [col for col in columns if col in self.ddf.columns]
                meta = pd.DataFrame(columns=available_columns)

                def subset_columns(df):
                    return df[available_columns]

                column_subset = self.ddf.map_partitions(subset_columns, meta=meta)
                df = column_subset.compute()

                # Save to CSV
                df.to_csv(productivity_path, index=False)

            except Exception as e:
                logger.error(f"Error in productivity analysis: {str(e)}")
                productivity_df = pd.DataFrame()

    def analyze_toxicity_release_correlation(self):
        """Analyze correlation between toxic communication and software releases."""
        releases_path = os.path.join(DEFAULT_CONFIG['output_dir'], 'releases_analysis.csv')

        if os.path.exists(releases_path):
            logger.info("Loading release analysis from disk...")
            try:
                releases_df = pd.read_csv(releases_path)
            except Exception as e:
                logger.error(f"Error loading release analysis: {str(e)}")
        else:
            logger.info("Analyzing toxicity-release correlation...")

            try:
                columns = ['type', 'created_at', 'is_toxic']
                column_subset = self.ddf[columns]
                df = column_subset.compute()

                if 'ReleaseEvent' in df['type'].unique():
                    releases = df[df['type'] == 'ReleaseEvent']
                    releases.to_csv(os.path.join(DEFAULT_CONFIG['output_dir'], 'releases.csv'), index=False)
                else:
                    logger.warning("No 'ReleaseEvent' records found. Skipping release analysis.")
                    releases = pd.DataFrame()

                toxicity_rates = df.groupby('created_at')['is_toxic'].mean()
                toxicity_rates.to_csv(os.path.join(DEFAULT_CONFIG['output_dir'], 'toxicity_rates.csv'), index=True)

                if not releases.empty or not isinstance(toxicity_rates, pd.Series) or not toxicity_rates.empty:
                    results = []
                    if not releases.empty:
                        results.append(releases)
                    if not isinstance(toxicity_rates, pd.DataFrame):
                        results.append(toxicity_rates.reset_index())

                    if results:
                        combined_df = pd.concat(results, ignore_index=True)
                        combined_df.to_csv(releases_path, index=False)

            except Exception as e:
                logger.error(f"Error in release analysis: {str(e)}")

    def analyze_experience_toxicity_correlation(self):
        """Analyze correlation between contributor experience and toxic communication."""
        experience_path = os.path.join(DEFAULT_CONFIG['output_dir'], 'experience_analysis.csv')

        if os.path.exists(experience_path):
            logger.info("Loading experience analysis from disk...")
            try:
                experience_df = pd.read_csv(experience_path)
            except Exception as e:
                logger.error(f"Error loading experience analysis: {str(e)}")
        else:
            logger.info("Analyzing experience-toxicity correlation...")

            try:
                required_columns = ['actor.login', 'is_toxic']
                missing_columns = [col for col in required_columns if col not in self.ddf.columns]

                if missing_columns:
                    logger.warning(f"Missing columns: {missing_columns}. Skipping experience-toxicity analysis.")
                    return

                column_subset = self.ddf[required_columns]
                df = column_subset.compute()

                # Save to CSV
                df.to_csv(experience_path, index=False)

            except Exception as e:
                logger.error(f"Error in experience analysis: {str(e)}")

    def save_results(self, output_dir):
        """Save all collected data and analysis results to files."""
        logger.info(f"Saving results to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        try:
            self.ddf.columns = self.ddf.columns.astype(str)
            column_list = list(self.ddf.columns)
            ddf_ordered = self.ddf[column_list]
            df_ordered = ddf_ordered.compute()
            ddf_final = dd.from_pandas(df_ordered, npartitions=DEFAULT_CONFIG['dask_partitions'])

            logger.info("Saving main DataFrame to CSV...")
            ddf_final.to_csv(
                os.path.join(output_dir, 'github_data.csv'),
                index=True
            )
            logger.info("Save complete.")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

    def generate_specific_csv_files(self, output_dir):
        """Generate specific CSV files for detailed analysis."""
        logger.info("Generating specific CSV files for analysis...")
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            df = self.ddf.compute()
            
            # 1. Generate commit_freq.csv
            logger.info("Generating commit frequency data...")
            if 'type' in df.columns:
                commit_data = df[df['type'] == 'PushEvent'].copy()
                if not commit_data.empty and 'created_at' in commit_data.columns:
                    if isinstance(commit_data['created_at'].iloc[0], str):
                        commit_data['date'] = commit_data['created_at'].str.split('T').str[0]
                    else:
                        commit_data['date'] = commit_data['created_at'].dt.date
                    
                    if 'repo.name' in commit_data.columns:
                        commit_freq = commit_data.groupby(['date', 'repo.name']).size().reset_index(name='commit_count')
                        commit_freq.to_csv(os.path.join(output_dir, 'commit_freq.csv'), index=False)
                        logger.info(f"Saved commit_freq.csv with {len(commit_freq)} records")
                    else:
                        logger.warning("Column 'repo.name' not found. Using date only for commit_freq.csv")
                        print(commit_data.columns)
                        
                        commit_freq = commit_data.groupby('date').size().reset_index(name='commit_count')
                        commit_freq.to_csv(os.path.join(output_dir, 'commit_freq.csv'), index=False)
                else:
                    logger.warning("No push events found or missing 'created_at' column. Creating empty commit_freq.csv")
                    pd.DataFrame(columns=['date', 'repo', 'commit_count']).to_csv(
                        os.path.join(output_dir, 'commit_freq.csv'), index=False)
            else:
                logger.warning("Column 'type' not found. Creating empty commit_freq.csv")
                pd.DataFrame(columns=['date', 'repo', 'commit_count']).to_csv(
                    os.path.join(output_dir, 'commit_freq.csv'), index=False)
            
            # 2. Generate issue_resolution_times.csv
            logger.info("Generating issue resolution times data...")
            if 'type' in df.columns and 'payload.issue.number' in df.columns:
                issue_opened = df[
                    (df['type'] == 'IssuesEvent') & 
                    (df['payload.action'] == 'opened')
                ].copy()
                
                issue_closed = df[
                    (df['type'] == 'IssuesEvent') & 
                    (df['payload.action'] == 'closed')
                ].copy()
                
                if not issue_opened.empty and not issue_closed.empty:
                    issue_opened = issue_opened[['payload.issue.number', 'created_at', 'repo.name']]
                    issue_opened.columns = ['issue_number', 'opened_at', 'repo']
                    
                    issue_closed = issue_closed[['payload.issue.number', 'created_at', 'repo.name']]
                    issue_closed.columns = ['issue_number', 'closed_at', 'repo']
                    
                    issue_times = pd.merge(
                        issue_opened, issue_closed, 
                        on=['issue_number', 'repo'], 
                        how='inner'
                    )
                    
                    if not issue_times.empty:
                        issue_times['opened_at'] = pd.to_datetime(issue_times['opened_at'])
                        issue_times['closed_at'] = pd.to_datetime(issue_times['closed_at'])
                        issue_times['resolution_time_hours'] = (
                            issue_times['closed_at'] - issue_times['opened_at']
                        ).dt.total_seconds() / 3600
                        
                        issue_times = issue_times[issue_times['resolution_time_hours'] >= 0]
                        
                        issue_times.to_csv(os.path.join(output_dir, 'issue_resolution_times.csv'), index=False)
                        logger.info(f"Saved issue_resolution_times.csv with {len(issue_times)} records")
                    else:
                        logger.warning("No matching issues found. Creating empty issue_resolution_times.csv")
                        pd.DataFrame(columns=['issue_number', 'repo', 'opened_at', 'closed_at', 'resolution_time_hours']).to_csv(
                            os.path.join(output_dir, 'issue_resolution_times.csv'), index=False)
                else:
                    logger.warning("No issue events found. Creating empty issue_resolution_times.csv")
                    pd.DataFrame(columns=['issue_number', 'repo', 'opened_at', 'closed_at', 'resolution_time_hours']).to_csv(
                        os.path.join(output_dir, 'issue_resolution_times.csv'), index=False)
            else:
                logger.warning("Required columns for issue resolution not found. Creating empty issue_resolution_times.csv")
                pd.DataFrame(columns=['issue_number', 'repo', 'opened_at', 'closed_at', 'resolution_time_hours']).to_csv(
                    os.path.join(output_dir, 'issue_resolution_times.csv'), index=False)
            
            # 3. Generate toxic_comments.csv
            logger.info("Generating toxic comments data...")
            if 'is_toxic' in df.columns and 'payload.comment.body' in df.columns:
                toxic_comments = df[df['is_toxic'] == True].copy()

                if not toxic_comments.empty:
                    columns_to_keep = ['created_at', 'actor.login', 'payload.comment.body', 'toxicity_score', 'repo.name']
                    columns_to_keep = [col for col in columns_to_keep if col in toxic_comments.columns]
                    
                    toxic_comments = toxic_comments[columns_to_keep]
                    toxic_comments.to_csv(os.path.join(output_dir, 'toxic_comments.csv'), index=False)
                    logger.info(f"Saved toxic_comments.csv with {len(toxic_comments)} records")
                else:
                    logger.warning("No toxic comments found. Creating empty toxic_comments.csv")
                    pd.DataFrame(columns=['created_at', 'user', 'comment', 'toxicity_score', 'repo']).to_csv(
                        os.path.join(output_dir, 'toxic_comments.csv'), index=False)
            else:
                logger.warning("Required columns for toxic comments not found. Creating empty toxic_comments.csv")
                pd.DataFrame(columns=['created_at', 'user', 'comment', 'toxicity_score', 'repo']).to_csv(
                    os.path.join(output_dir, 'toxic_comments.csv'), index=False)
            
            # 4. Generate user_metrics.csv
            logger.info("Generating user metrics data...")
            if 'actor.login' in df.columns:
                user_activity = df.groupby('actor.login').agg({
                    'id': 'count',  # Total activity count
                }).reset_index()
                user_activity.columns = ['user', 'activity_count']
                
                if 'type' in df.columns:
                    commit_counts = df[df['type'] == 'PushEvent'].groupby('actor.login').size().reset_index(name='commit_count')
                    commit_counts.columns = ['user', 'commit_count']
                    user_activity = pd.merge(user_activity, commit_counts, on='user', how='left')
                    user_activity['commit_count'] = user_activity['commit_count'].fillna(0).astype(int)
                
                comment_counts = df[df['type'] == 'IssueCommentEvent'].groupby('actor.login').size().reset_index(name='comment_count')
                comment_counts.columns = ['user', 'comment_count']
                user_activity = pd.merge(user_activity, comment_counts, on='user', how='left')
                user_activity['comment_count'] = user_activity['comment_count'].fillna(0).astype(int)
                
                user_activity.to_csv(os.path.join(output_dir, 'user_metrics.csv'), index=False)
                logger.info(f"Saved user_metrics.csv with {len(user_activity)} records")
            else:
                logger.warning("Column 'actor.login' not found. Creating empty user_metrics.csv")
                pd.DataFrame(columns=['user', 'activity_count', 'commit_count', 'comment_count']).to_csv(
                    os.path.join(output_dir, 'user_metrics.csv'), index=False)
            
            # 5. Generate toxicity_by_user.csv
            logger.info("Generating toxicity by user data...")
            if 'actor.login' in df.columns and 'is_toxic' in df.columns:
                comments_df = df[df['type'] == 'IssueCommentEvent'].copy()
                
                if not comments_df.empty:
                    user_toxicity = comments_df.groupby('actor.login').agg({
                        'is_toxic': ['count', 'sum']  # Count of comments and sum of toxic flags
                    }).reset_index()
                    
                    user_toxicity.columns = ['user', 'total_comments', 'toxic_comments']
                    
                    user_toxicity['toxicity_ratio'] = user_toxicity['toxic_comments'] / user_toxicity['total_comments']
                    
                    user_toxicity = user_toxicity[user_toxicity['total_comments'] > 0]
                    
                    user_toxicity.to_csv(os.path.join(output_dir, 'toxicity_by_user.csv'), index=False)
                    logger.info(f"Saved toxicity_by_user.csv with {len(user_toxicity)} records")
                else:
                    logger.warning("No comments found. Creating empty toxicity_by_user.csv")
                    pd.DataFrame(columns=['user', 'total_comments', 'toxic_comments', 'toxicity_ratio']).to_csv(
                        os.path.join(output_dir, 'toxicity_by_user.csv'), index=False)
            else:
                logger.warning("Required columns for toxicity by user not found. Creating empty toxicity_by_user.csv")
                pd.DataFrame(columns=['user', 'total_comments', 'toxic_comments', 'toxicity_ratio']).to_csv(
                    os.path.join(output_dir, 'toxicity_by_user.csv'), index=False)
            
            logger.info("All specific CSV files generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating specific CSV files: {str(e)}")
            for file_name in ['commit_freq.csv', 'issue_resolution_times.csv', 'toxic_comments.csv', 
                            'user_metrics.csv', 'toxicity_by_user.csv']:
                pd.DataFrame().to_csv(os.path.join(output_dir, file_name), index=False)
                logger.error(f"Created empty {file_name} due to error")
    
    def generate_summary_report(self, output_dir):
        """Generate a summary report of the analysis."""
        logger.info("Generating summary report")

        try:
            df = self.ddf.compute()
            total_records = len(df)
            avg_toxicity = 0.0
            toxic_comments_count = 0
            
            if 'toxicity_score' in df.columns:
                avg_toxicity = df['toxicity_score'].mean() if not df['toxicity_score'].empty else 0.0
                
            if 'is_toxic' in df.columns:
                toxic_comments_count = df['is_toxic'].sum() if not df['is_toxic'].empty else 0
            
            report = {
                'total_records': total_records,
                'toxic_comments': int(toxic_comments_count),
                'average_toxicity': float(avg_toxicity)
            }
            
            with open(os.path.join(output_dir, 'summary_report.json'), 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info("Summary report generated")
            
        except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            report = {
                'total_records': 0,
                'toxic_comments': 0,
                'average_toxicity': 0.0,
                'error': str(e)
            }
            
            with open(os.path.join(output_dir, 'summary_report.json'), 'w') as f:
                json.dump(report, f, indent=2)
 