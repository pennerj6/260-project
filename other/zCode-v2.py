import os
import gc
import json
import time
import logging
import datetime
import argparse
import requests
import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask.bag as db
import psutil
from io import BytesIO
from collections import defaultdict
from tqdm import tqdm
from joblib import Parallel, delayed
from transformers import pipeline, AutoTokenizer
import torch
import multiprocessing
import gzip

import ast


# Default configuration parameters
DEFAULT_CONFIG = {
    'start_date': '2023-01-01',
    'end_date': '2023-01-01',
    'start_hour': 15,  # Add start hour for testing (8 AM)
    'output_dir': 'output',
    'toxicity_threshold': 0.04,  # Threshold for classifying comments as toxic
    'use_sampling': True,
    'sample_size': 10000,
    'max_workers': 5,  # Reduced to avoid overwhelming the system
    'dask_partitions': 10,  # Number of partitions for Dask DataFrame
    'batch_size': 32,  # Batch size for model inference
    'chunk_size': 3,  # Number of URLs to process in a batch
    'memory_limit_factor': 0.7  # Percentage of total memory to allocate to Dask
}

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ToxicityRater:
    def __init__(self, use_sampling=False, sample_size=None):
        self.model_name = "unitary/toxic-bert"
        self.device = -1  # Force CPU
        logger.info(f"Using device: CPU")

        logger.info(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        # For CPU optimization - use efficient pipeline settings
        self.model = pipeline(
            "text-classification",
            model=self.model_name,
            device=self.device,
            framework="pt"
        )

        self.max_length = 512
        self.use_sampling = use_sampling
        self.sample_size = sample_size

        # Warm up the model
        logger.info("Warming up model...")
        _ = self.model(["This is a warm-up text"])
        gc.collect()

    @torch.no_grad()
    def get_toxicity_ratings(self, comments, batch_size=32):
        """Process a list of comments and return toxicity scores with memory efficiency."""
        if not comments:
            return []

        start_time = time.time()
        logger.info(f"Processing {len(comments)} comments...")

        if self.use_sampling and len(comments) > self.sample_size:
            logger.info(f"Sampling {self.sample_size} comments from {len(comments)} total")
            indices = np.random.choice(len(comments), self.sample_size, replace=False)
            sampled_comments = [comments[i] for i in indices]
            results = np.zeros(len(comments))
            sample_results = self._process_comments(sampled_comments, batch_size)
            for idx, result_idx in enumerate(indices):
                results[result_idx] = sample_results[idx]
            return results.tolist()
        else:
            return self._process_comments(comments, batch_size)

    def _process_comments(self, comments, batch_size):
        """Process all comments using batching with CPU optimization."""
        # Use 75% of available CPU cores for optimal performance
        n_jobs = max(1, int(multiprocessing.cpu_count() * 0.75))
        logger.info(f"Using {n_jobs} CPU cores for processing")

        # Break into smaller batches to avoid memory issues
        num_batches = (len(comments) + batch_size - 1) // batch_size
        batches = [comments[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]

        # Process batches with progress reporting
        all_scores = []
        # Use smaller chunks for parallel processing to avoid memory spikes
        chunk_size = 10  # Process 10 batches at a time
        for i in range(0, len(batches), chunk_size):
            chunk_batches = batches[i:i + chunk_size]
            logger.info(f"Processing batch chunk {i // chunk_size + 1}/{(len(batches) + chunk_size - 1) // chunk_size}")

            chunk_scores = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(self._process_batch)(batch)
                for batch in tqdm(chunk_batches, desc=f"Processing Batches {i}-{i + len(chunk_batches) - 1}")
            )
            all_scores.extend(chunk_scores)

            # Force garbage collection after each chunk
            gc.collect()

        # Flatten the scores
        return [score for batch_scores in all_scores for score in batch_scores]

    def _process_batch(self, batch):
        """Process a single batch of comments."""
        if not "".join(batch).strip():
            return [0.0] * len(batch)
        try:
            results = self.model(batch, truncation=True, max_length=self.max_length)
            return [result['score'] if result['label'] == 'toxic' else 0 for result in results]
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)[:100]}...")
            return [0.0] * len(batch)

    def test_toxicity(self, sentence):
        """Test the toxicity checker with a given sentence."""
        result = self.model([sentence], truncation=True, max_length=self.max_length)
        return result


class GitHubArchiveAnalyzer:
    def __init__(self, start_date, end_date, start_hour=None, toxicity_threshold=0.5):
        self.start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        self.start_hour = start_hour  # Add start hour
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

        # GitHub API token
        self.github_token = os.environ.get('GITHUB_TOKEN', '')

        # Cache for API calls
        self.user_cache = {}
        self.cache_file = os.path.join(DEFAULT_CONFIG['output_dir'], 'user_cache.json')
        self._load_user_cache()

    def _load_user_cache(self):
        """Load user cache from disk if available."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.user_cache = json.load(f)
                logger.info(f"Loaded user cache with {len(self.user_cache)} entries")
            except Exception as e:
                logger.error(f"Error loading user cache: {str(e)}")
                self.user_cache = {}

    def _save_user_cache(self):
        """Save user cache to disk."""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.user_cache, f)
            logger.info(f"Saved user cache with {len(self.user_cache)} entries")
        except Exception as e:
            logger.error(f"Error saving user cache: {str(e)}")

    def generate_urls(self):
        """Generate URLs for a specific hour between start and end date."""
        urls = []
        current_date = self.start_date
        while current_date <= self.end_date:
            # Only generate URL for the specified hour
            if self.start_hour is not None:
                date_str = current_date.strftime('%Y-%m-%d')
                url = f"https://data.gharchive.org/{date_str}-{self.start_hour}.json.gz"
                urls.append(url)
            else:
                # Generate URLs for all hours (original behavior)
                for hour in range(24):
                    date_str = current_date.strftime('%Y-%m-%d')
                    url = f"https://data.gharchive.org/{date_str}-{hour}.json.gz"
                    urls.append(url)
            current_date += datetime.timedelta(days=1)
        return urls

    def fetch_and_process_data(self, url):
        """Fetch and process data from a single GitHub Archive URL using streaming."""
        try:
            logger.info(f"Fetching {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            logger.info(f"Fetched {url} successfully.")

            chunks = []
            record_count = 0
            filtered_count = 0

            with gzip.GzipFile(fileobj=BytesIO(response.content)) as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if record_count == 0:
                            print(record)  # Debug: Print the first record
                        record_count += 1

                        # Flatten the record
                        flattened_record = self._flatten_dict(record)

                        # Early filtering - only keep events we need
                        if 'type' in flattened_record and flattened_record['type'] in ['IssueCommentEvent', 'PushEvent', 'IssuesEvent', 'ReleaseEvent']:
                            # Check if the record has all required fields based on its type
                            is_valid = True

                            # Common required fields for all events
                            if not all(key in flattened_record for key in ['id', 'type', 'created_at', 'actor.login']):
                                is_valid = False

                            # For IssueCommentEvent, ensure we have comment data
                            if is_valid and flattened_record['type'] == 'IssueCommentEvent':
                                if 'payload.comment.body' not in flattened_record:
                                    logger.warning(f"Missing 'payload.comment.body' in record: {flattened_record}")
                                    is_valid = False

                            # For IssuesEvent, ensure we have issue number and action
                            if is_valid and flattened_record['type'] == 'IssuesEvent':
                                if not ('payload.issue.number' in flattened_record and
                                        'payload.action' in flattened_record):
                                    is_valid = False

                            # For PushEvent, ensure we have repository info
                            if is_valid and flattened_record['type'] == 'PushEvent':
                                if 'repo.name' not in flattened_record:
                                    is_valid = False

                            # Add the record if it passes all checks
                            if is_valid:
                                chunks.append(flattened_record)
                                filtered_count += 1

                            # Process in batches to avoid memory explosion
                            if len(chunks) >= 5000:
                                chunk_df = pd.DataFrame(chunks)
                                print("Sample of Processed Data:")
                                print(chunk_df.head())
                                yield chunk_df
                                chunks = []

                                # Force garbage collection
                                gc.collect()

                    except Exception as e:
                        logger.error(f"Error parsing line: {str(e)[:100]}...")
                        continue

                # Process any remaining records
                if chunks:
                    chunk_df = pd.DataFrame(chunks)
                    print("Sample of Processed Data:")
                    print(chunk_df.head())
                    yield chunk_df

            logger.info(f"Processed {record_count} records, kept {filtered_count} valid records from {url}.")

        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            yield pd.DataFrame()  # Return empty DataFrame on error                 

    def _flatten_dict(self, d, parent_key='', sep='.'):
        """Recursively flatten a nested dictionary, ensuring consistent column naming."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    
    def collect_data(self):
        """Collect data from GitHub Archive using optimized streaming."""
        raw_data_path = os.path.join(DEFAULT_CONFIG['output_dir'], 'raw_data_chunks')
        os.makedirs(raw_data_path, exist_ok=True)

        if os.path.exists(os.path.join(raw_data_path, '_SUCCESS')):
            logger.info("Loading pre-processed raw data chunks...")
            self.ddf = dd.read_parquet(raw_data_path)
            print("DEBUG: Actual columns in self.ddf:")
            print(self.ddf.columns.tolist())
        else:
            logger.info("Fetching and processing raw data in chunks...")
            urls = self.generate_urls()

            # Process URLs in smaller batches to avoid memory issues
            chunk_size = DEFAULT_CONFIG['chunk_size']  # Process N URLs at a time
            for i in range(0, len(urls), chunk_size):
                url_batch = urls[i:i + chunk_size]
                logger.info(f"Processing URL batch {i // chunk_size + 1}/{(len(urls) + chunk_size - 1) // chunk_size}")

                # Initialize batch_dfs list BEFORE entering the URL loop
                batch_dfs = []

                # Process each URL in the batch sequentially to avoid memory spikes
                for url in url_batch:
                    for chunk_df in self.fetch_and_process_data(url):
                        if not chunk_df.empty:
                            batch_dfs.append(chunk_df)

                    # Force garbage collection after each URL
                    gc.collect()

                # Combine batch results and save to disk
                if batch_dfs:
                    try:
                        batch_df = pd.concat(batch_dfs, ignore_index=True)
                        batch_dd = dd.from_pandas(batch_df, npartitions=DEFAULT_CONFIG['dask_partitions'])
                        batch_dd.to_parquet(os.path.join(raw_data_path, f'batch_{i // chunk_size}.parquet'))

                        # Clear memory
                        del batch_dfs, batch_df, batch_dd
                        gc.collect()
                    except Exception as e:
                        logger.error(f"Error saving batch data: {str(e)}")

                # Create _SUCCESS flag file
                with open(os.path.join(raw_data_path, '_SUCCESS'), 'w') as f:
                    f.write('Success')

                # Load the saved chunks
                try:
                    self.ddf = dd.read_parquet(raw_data_path)
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
                    self.ddf = dd.from_pandas(pd.DataFrame(columns=['type', 'created_at', 'actor.login', 'payload.comment.body']),
                                            npartitions=1)
                    
    def process_toxicity(self, rater, batch_size=32):
        """Process toxicity scores for comments with memory efficiency."""
        toxicity_scores_path = os.path.join(DEFAULT_CONFIG['output_dir'], 'toxicity_scores.parquet')

        if os.path.exists(toxicity_scores_path):
            logger.info("Loading toxicity scores from disk...")
            try:
                toxicity_scores_dd = dd.read_parquet(toxicity_scores_path)
                print("Sample of Toxicity Scores:")
                print(toxicity_scores_dd.head())
            except Exception as e:
                logger.error(f"Error loading toxicity scores: {str(e)}")
        else:
            logger.info("Computing toxicity scores...")

            # Filter to just comment events to reduce memory usage
            comment_events = self.ddf[self.ddf['type'] == 'IssueCommentEvent'].compute()
            print("Sample of Comment Events:")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_colwidth', 100)
            print(comment_events)

            # Check if the required columns exist
            if 'payload.comment.body' in comment_events.columns:
                print("Flattened key 'payload.comment.body' found. Proceeding with toxicity processing.")
                comments = comment_events['payload.comment.body'].fillna('').tolist()

                print("Sample of Comments:")
                print(comments[:5])  # Print the first 5 comments

                if comments:
                    # Process comments in memory-efficient chunks
                    available_mem = psutil.virtual_memory().available
                    chunk_size = min(5000, max(1000, int(available_mem / (5 * 1024 * 1024))))
                    logger.info(f"Processing comments in chunks of {chunk_size}")

                    all_scores = []
                    for i in range(0, len(comments), chunk_size):
                        chunk = comments[i:i + chunk_size]
                        logger.info(f"Processing comment chunk {i // chunk_size + 1}/{(len(comments) + chunk_size - 1) // chunk_size}")
                        chunk_scores = rater.get_toxicity_ratings(chunk, batch_size=batch_size)
                        all_scores.extend(chunk_scores)

                        # Force garbage collection
                        gc.collect()

                    # Create a DataFrame with all scores
                    toxicity_df = pd.DataFrame({'toxicity_score': all_scores})
                    toxicity_dd = dd.from_pandas(toxicity_df, npartitions=DEFAULT_CONFIG['dask_partitions'])
                    toxicity_dd.to_parquet(toxicity_scores_path)

                    # Add toxicity scores to main DataFrame
                    comment_events = comment_events.assign(toxicity_score=all_scores)
                    comment_events = comment_events.assign(is_toxic=comment_events['toxicity_score'] >= self.toxicity_threshold)

                    # Debug: Print columns after adding is_toxic
                    logger.info(f"Columns in comment_events after adding is_toxic: {comment_events.columns}")

                    # Merge the updated comment events back into the main DataFrame
                    self.ddf = self.ddf.merge(
                        comment_events[['id', 'toxicity_score', 'is_toxic']],
                        on='id',
                        how='left'
                    )

                    # Fill NaN values for rows that were not comment events
                    self.ddf['toxicity_score'] = self.ddf['toxicity_score'].fillna(0.0)
                    self.ddf['is_toxic'] = self.ddf['is_toxic'].fillna(False)

                    # Debug: Print columns in self.ddf after merge
                    logger.info(f"Columns in self.ddf after merge: {self.ddf.columns}")
                else:
                    logger.warning("No comments found to process.")
                    self.ddf = self.ddf.assign(toxicity_score=0.0)
                    self.ddf = self.ddf.assign(is_toxic=False)
            else:
                logger.warning("Column 'payload.comment.body' not found. Skipping toxicity processing.")
                self.ddf = self.ddf.assign(toxicity_score=0.0)
                self.ddf = self.ddf.assign(is_toxic=False)          

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

    def analyze_toxicity_productivity_correlation(self):
        """Analyze how toxic communication affects productivity with checkpointing."""
        checkpoint_path = os.path.join(DEFAULT_CONFIG['output_dir'], 'checkpoints', 'productivity')
        os.makedirs(checkpoint_path, exist_ok=True)

        productivity_path = os.path.join(DEFAULT_CONFIG['output_dir'], 'productivity_analysis.parquet')

        if os.path.exists(productivity_path):
            logger.info("Loading productivity analysis results from disk...")
            try:
                productivity_df = dd.read_parquet(productivity_path)
                print("Sample of Productivity Analysis Data:")
                print(productivity_df.head())
            except Exception as e:
                logger.error(f"Error loading productivity analysis: {str(e)}")
                productivity_df = dd.from_pandas(pd.DataFrame(), npartitions=1)
        else:
            logger.info("Analyzing toxicity-productivity correlation...")

            try:
                # Convert to pandas for analysis
                df = self.ddf.compute()

                # Check if required columns exist
                required_columns = ['type', 'created_at', 'payload.issue.number', 'is_toxic']
                missing_columns = [col for col in required_columns if col not in df.columns]

                if missing_columns:
                    logger.warning(f"Missing columns in DataFrame: {missing_columns}. Skipping toxicity-productivity analysis.")
                else:
                    # Perform analysis
                    # (Add your analysis logic here)

                    # Save results
                    productivity_df = dd.from_pandas(df, npartitions=DEFAULT_CONFIG['dask_partitions'])
                    productivity_df.to_parquet(productivity_path)

            except Exception as e:
                logger.error(f"Error in productivity analysis: {str(e)}")
                productivity_df = dd.from_pandas(pd.DataFrame(), npartitions=1)

    def analyze_toxicity_release_correlation(self):
        """Analyze correlation between toxic communication and software releases with memory efficiency."""
        releases_path = os.path.join(DEFAULT_CONFIG['output_dir'], 'releases_analysis.parquet')

        if os.path.exists(releases_path):
            logger.info("Loading release analysis from disk...")
            try:
                releases_df = dd.read_parquet(releases_path)
                print("Sample of Release Analysis Data:")
                print(releases_df.head())
            except Exception as e:
                logger.error(f"Error loading release analysis: {str(e)}")
        else:
            logger.info("Analyzing toxicity-release correlation...")

            try:
                # Convert to Pandas for analysis - but only select needed columns
                df = self.ddf[['type', 'created_at', 'is_toxic']].compute()

                # Track release events
                if 'type' in df.columns and 'ReleaseEvent' in df['type'].unique():
                    releases = df[df['type'] == 'ReleaseEvent']
                    releases.to_csv(os.path.join(DEFAULT_CONFIG['output_dir'], 'releases.csv'), index=False)
                else:
                    logger.warning("No 'ReleaseEvent' records found. Skipping release analysis.")
                    releases = pd.DataFrame()

                # Correlate toxic communication with release timing
                if 'is_toxic' in df.columns:
                    toxicity_rates = df.groupby('created_at')['is_toxic'].mean()
                    toxicity_rates.to_csv(os.path.join(DEFAULT_CONFIG['output_dir'], 'toxicity_rates.csv'), index=True)
                else:
                    logger.warning("No 'is_toxic' column found. Skipping toxicity rates analysis.")
                    toxicity_rates = pd.DataFrame()

                # Save results
                if not releases.empty or not toxicity_rates.empty:
                    results = []
                    if not releases.empty:
                        results.append(releases)
                    if not isinstance(toxicity_rates, pd.DataFrame):
                        results.append(toxicity_rates.reset_index())

                    combined_df = pd.concat(results, ignore_index=True)
                    releases_df = dd.from_pandas(combined_df, npartitions=DEFAULT_CONFIG['dask_partitions'])
                    releases_df.to_parquet(releases_path)

            except Exception as e:
                logger.error(f"Error in release analysis: {str(e)}")

    def analyze_experience_toxicity_correlation(self):
        """Analyze correlation between contributor experience and toxic communication with caching."""
        experience_path = os.path.join(DEFAULT_CONFIG['output_dir'], 'experience_analysis.parquet')

        if os.path.exists(experience_path):
            logger.info("Loading experience analysis from disk...")
            try:
                experience_df = dd.read_parquet(experience_path)
                print("Sample of Experience Analysis Data:")
                print(experience_df.head())
            except Exception as e:
                logger.error(f"Error loading experience analysis: {str(e)}")
        else:
            logger.info("Analyzing experience-toxicity correlation...")

            try:
                # Debug: Print columns in self.ddf
                logger.info(f"Columns in self.ddf: {self.ddf.columns}")

                # Check if required columns exist
                required_columns = ['actor.login', 'is_toxic']
                missing_columns = [col for col in required_columns if col not in self.ddf.columns]

                if missing_columns:
                    logger.warning(f"Missing columns: {missing_columns}. Skipping experience-toxicity analysis.")
                    return

                # Select only needed columns to reduce memory usage
                df = self.ddf[required_columns].compute()

                # Perform analysis
                # (Add your analysis logic here)

                # Save results
                experience_df = dd.from_pandas(df, npartitions=DEFAULT_CONFIG['dask_partitions'])
                experience_df.to_parquet(experience_path)
            except Exception as e:
                logger.error(f"Error in experience analysis: {str(e)}")
            
    def save_results(self, output_dir):
        """Save all collected data and analysis results to files with memory optimization."""
        logger.info(f"Saving results to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Ensure all column names are strings
            self.ddf.columns = self.ddf.columns.astype(str)

            # Save Dask DataFrame to Parquet
            logger.info("Saving main DataFrame to Parquet...")
            self.ddf.to_parquet(
                os.path.join(output_dir, 'github_data.parquet'),
                write_index=True,
                engine='pyarrow',
                compression='snappy'
            )
            print("Sample of Main DataFrame:")
            print(self.ddf.head())
            logger.info("Save complete.")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")


    def generate_specific_csv_files(self, output_dir):
        """Generate specific CSV files for detailed analysis."""
        logger.info("Generating specific CSV files for analysis...")
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Load the main data if needed
            df = self.ddf.compute()
            
            # 1. Generate commit_freq.csv
            logger.info("Generating commit frequency data...")
            if 'type' in df.columns:
                commit_data = df[df['type'] == 'PushEvent'].copy()
                if not commit_data.empty and 'created_at' in commit_data.columns:
                    # Extract date only from created_at
                    if isinstance(commit_data['created_at'].iloc[0], str):
                        commit_data['date'] = commit_data['created_at'].str.split('T').str[0]
                    else:
                        commit_data['date'] = commit_data['created_at'].dt.date
                    
                    # Count commits by date and repo
                    if 'repo.name' in commit_data.columns:
                        commit_freq = commit_data.groupby(['date', 'repo.name']).size().reset_index(name='commit_count')
                        commit_freq.to_csv(os.path.join(output_dir, 'commit_freq.csv'), index=False)
                        logger.info(f"Saved commit_freq.csv with {len(commit_freq)} records")
                    else:
                        if 'repo' in commit_data.columns:
                            print("i knew it")
                            print(commit_data)
                            if 'repo_name' in commit_data.columns:
                                print('ahh')

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
                # Find issue open events
                issue_opened = df[
                    (df['type'] == 'IssuesEvent') & 
                    (df['payload.action'] == 'opened')
                ].copy()
                
                # Find issue closed events
                issue_closed = df[
                    (df['type'] == 'IssuesEvent') & 
                    (df['payload.action'] == 'closed')
                ].copy()
                
                if not issue_opened.empty and not issue_closed.empty:
                    # Prepare columns for merging
                    issue_opened = issue_opened[['payload.issue.number', 'created_at', 'repo.name']]
                    issue_opened.columns = ['issue_number', 'opened_at', 'repo']
                    
                    issue_closed = issue_closed[['payload.issue.number', 'created_at', 'repo.name']]
                    issue_closed.columns = ['issue_number', 'closed_at', 'repo']
                    
                    # Merge to find matching open/close pairs
                    issue_times = pd.merge(
                        issue_opened, issue_closed, 
                        on=['issue_number', 'repo'], 
                        how='inner'
                    )
                    
                    # Calculate resolution time
                    if not issue_times.empty:
                        issue_times['opened_at'] = pd.to_datetime(issue_times['opened_at'])
                        issue_times['closed_at'] = pd.to_datetime(issue_times['closed_at'])
                        issue_times['resolution_time_hours'] = (
                            issue_times['closed_at'] - issue_times['opened_at']
                        ).dt.total_seconds() / 3600
                        
                        # Only keep valid resolution times
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
                # Filter for toxic comments
                toxic_comments = df[df['is_toxic'] == True].copy()
                
                if not toxic_comments.empty:
                    # Select relevant columns
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
                
                # Add commit counts if available
                if 'type' in df.columns:
                    commit_counts = df[df['type'] == 'PushEvent'].groupby('actor.login').size().reset_index(name='commit_count')
                    commit_counts.columns = ['user', 'commit_count']
                    user_activity = pd.merge(user_activity, commit_counts, on='user', how='left')
                    user_activity['commit_count'] = user_activity['commit_count'].fillna(0).astype(int)
                
                # Add comment counts if available
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
                # Only look at comments
                comments_df = df[df['type'] == 'IssueCommentEvent'].copy()
                
                if not comments_df.empty:
                    user_toxicity = comments_df.groupby('actor.login').agg({
                        'is_toxic': ['count', 'sum']  # Count of comments and sum of toxic flags
                    }).reset_index()
                    
                    # Flatten MultiIndex columns
                    user_toxicity.columns = ['user', 'total_comments', 'toxic_comments']
                    
                    # Calculate toxicity ratio
                    user_toxicity['toxicity_ratio'] = user_toxicity['toxic_comments'] / user_toxicity['total_comments']
                    
                    # Only include users with at least one comment
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
            # Create empty files in case of errors
            for file_name in ['commit_freq.csv', 'issue_resolution_times.csv', 'toxic_comments.csv', 
                            'user_metrics.csv', 'toxicity_by_user.csv']:
                pd.DataFrame().to_csv(os.path.join(output_dir, file_name), index=False)
                logger.error(f"Created empty {file_name} due to error")
    
    def generate_summary_report(self, output_dir):
        """Generate a summary report of the analysis."""
        logger.info("Generating summary report")

        # Check if toxicity_score column exists
        if 'toxicity_score' in self.ddf.columns:
            avg_toxicity = float(self.ddf['toxicity_score'].mean().compute())
        else:
            logger.warning("Column 'toxicity_score' not found. Setting average_toxicity to 0.")
            avg_toxicity = 0.0

        # Make sure to convert Dask/NumPy types to Python native types
        if 'is_toxic' in self.ddf.columns:
            toxic_comments_count = int(self.ddf['is_toxic'].sum().compute())
        else:
            toxic_comments_count = 0

        # Convert total_records to int
        total_records = int(len(self.ddf))

        report = {
            'total_records': total_records,
            'toxic_comments': toxic_comments_count,
            'average_toxicity': avg_toxicity
        }

        with open(os.path.join(output_dir, 'summary_report.json'), 'w') as f:
            json.dump(report, f, indent=2)

        logger.info("Summary report generated")


def main():
    """Main entry point for the analysis script."""
    parser = argparse.ArgumentParser(description='Analyze GitHub Archive data for toxicity patterns')
    parser.add_argument('--start_date', type=str, default=DEFAULT_CONFIG['start_date'], help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, default=DEFAULT_CONFIG['end_date'], help='End date in YYYY-MM-DD format')
    parser.add_argument('--start_hour', type=int, default=DEFAULT_CONFIG['start_hour'], help='Start hour (0-23) for testing')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_CONFIG['output_dir'], help='Directory to save results')
    parser.add_argument('--toxicity_threshold', type=float, default=DEFAULT_CONFIG['toxicity_threshold'], help='Threshold for classifying comments as toxic')
    parser.add_argument('--use_sampling', action='store_true', default=DEFAULT_CONFIG['use_sampling'], help='Use sampling for toxicity analysis')
    parser.add_argument('--sample_size', type=int, default=DEFAULT_CONFIG['sample_size'], help='Sample size for toxicity analysis if sampling is used')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = GitHubArchiveAnalyzer(
        start_date=args.start_date,
        end_date=args.end_date,
        start_hour=args.start_hour,  # Pass start hour
        toxicity_threshold=args.toxicity_threshold
    )

    # Initialize toxicity rater
    rater = ToxicityRater(use_sampling=args.use_sampling, sample_size=args.sample_size)

    # Test the toxicity checker with "hello idiot"
    test_sentence = "hello idiot"
    result = rater.test_toxicity(test_sentence)
    print(f"Toxicity result for '{test_sentence}': {result}")

    # Collect data
    analyzer.collect_data()

    # Process toxicity
    analyzer.process_toxicity(rater)

    # Analyze RQs
    analyzer.analyze_toxicity_productivity_correlation()
    analyzer.analyze_toxicity_release_correlation()
    analyzer.analyze_experience_toxicity_correlation()

    # Save results
    analyzer.save_results(args.output_dir)
    
    # Generate the specific CSV files
    analyzer.generate_specific_csv_files(args.output_dir)

    # Generate summary report
    analyzer.generate_summary_report(args.output_dir)

if __name__ == "__main__":
    main()