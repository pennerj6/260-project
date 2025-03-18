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
from src.toxicity_rater import ToxicityRater
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed

'''
After meetign w TA on suggestions ab how to get repos that have toxicity 
    - for reference,the way we were doing it was geting data from random repos (w a few filters like stars, contributions,etc..) via github api
    - but there was always no toxicity and it took ALOT of computation time (hours of code running just to get like 0.35 toxicity MAX)
    - TA reccommened we find another existing paper and use similar methods they did to get toxic repos (and if they posted code we can use that to help us find the toxic repos themselves)
    - we found an https://github.com/vcu-swim-lab/incivility-dataset/tree/main/dataset which contains more toxic repos (still not very high, but better)

    - i also found an online db that stores (hourly i think) github repo data (i think prof mentioned breifly, along with GHtorrent-> which DNE anymore)
    - what this code file does is go thru https://data.gharchive.org/{date_str}-{hour}.json.gz' and dates/times that we choose and search for toxic comments (0.5 or above), gpt helped w basic syntax, debugging, error handling(try except blocks w logger), making computationally faster, and w memory issues
    - the data from here is not gureenteed to be as toxic as we might want as it doesnt check every hour of everyday, but it is good enough for our purposes
'''

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    'output_dir': './list_of_repos_to_analyze',
    'chunk_size': 5
}

class GitHubArchiveAnalyzer:
    def __init__(self, dates, hours_range, toxicity_threshold=0.5):
        self.dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
        self.hours_range = hours_range
        self.toxicity_threshold = toxicity_threshold
        self.rater = ToxicityRater(use_sampling=True, sample_size=10000)  # Fixed typo in sample_size (was 10a000)

    def generate_urls(self):
        """Generate all URLs for GHArchive given the dates and hours range"""
        urls = []
        
        for date in self.dates:
            for hour in self.hours_range:
                date_str = date.strftime('%Y-%m-%d')
                url = f"https://data.gharchive.org/{date_str}-{hour}.json.gz"
                urls.append(url)
        return urls

    def _flatten_dict(self, d, parent_key='', sep='.'):
        """Flatten nested dictionaries"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def fetch_and_process_data(self, url):
        """Fetch and process data from a GitHub Archive URL, focusing on relevant events with toxicity check."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            logger.info(f"Fetched {url} successfully.")

            chunks = []
            record_count = 0
            filtered_count = 0
            toxic_count = 0
            # Increased sample size for more data
            sample_size = 10000
            
            with gzip.GzipFile(fileobj=BytesIO(response.content)) as f:
                for line in f:
                    try:
                        if record_count >= sample_size:
                            logger.info(f"Reached sample_size limit of {sample_size} records.")
                            break

                        record = json.loads(line)
                        record_count += 1

                        # Filter for relevant event types with comments
                        # Added PullRequestReviewCommentEvent to capture more types of comments
                        if record.get('type') in ['IssueCommentEvent', 'PullRequestReviewCommentEvent'] and \
                           'payload' in record and \
                           'comment' in record['payload'] and \
                           'body' in record['payload']['comment'] and \
                           record['payload']['comment']['body'] and \
                           'repo' in record and \
                           'url' in record['repo']:
                            
                            # Extract the comment body for toxicity check
                            comment_body = record['payload']['comment'].get('body')
                            
                            # Check toxicity directly
                            toxicity_score = self.rater.get_toxicity(comment_body)
                            
                            # Only keep comments that meet the toxicity threshold
                            if toxicity_score['score'] >= self.toxicity_threshold:
                                # Extract fields, added comment_body to save the actual content
                                simplified_record = {
                                    'id': record.get('id'),
                                    'repo_url': record['repo'].get('url'),
                                    'toxicity_score': toxicity_score['score'],
                                    'toxicity_type': toxicity_score.get('type', 'general'),
                                    'comment_body': comment_body[:1000],  # Truncate very long comments
                                    'created_at': record.get('created_at'),
                                    'event_type': record.get('type')
                                }
                                
                                chunks.append(simplified_record)
                                toxic_count += 1
                            
                            filtered_count += 1

                        # Process in batches
                        if len(chunks) >= 100:
                            chunk_df = pd.DataFrame(chunks)
                            yield chunk_df
                            chunks = []
                            gc.collect()

                    except Exception as e:
                        logger.error(f"Error parsing line: {str(e)[:100]}...")
                        continue

                # Process remaining records
                if chunks:
                    chunk_df = pd.DataFrame(chunks)
                    yield chunk_df

            logger.info(f"Processed {record_count} records, filtered {filtered_count}, found {toxic_count} toxic comments from {url}.")

        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            logger.error(traceback.format_exc())
            yield pd.DataFrame()  # Return empty DataFrame on error

    def collect_data(self):
        """Collect and process data from GHArchive, using Dask for parallel processing."""
        # Make sure output directory exists
        os.makedirs(DEFAULT_CONFIG['output_dir'], exist_ok=True)

        date_range_str = ", ".join([date.strftime('%Y-%m-%d') for date in self.dates])
        hours_str = ", ".join(map(str, self.hours_range))
        logger.info(f"Fetching and processing GitHub data for dates [{date_range_str}] and hours [{hours_str}] to find toxic repos")
        
        urls = self.generate_urls()
        
        # Track progress
        start_time = time.time()
        all_toxic_dfs = []
        
        # Process URLs in parallel using ThreadPoolExecutor with fewer workers
        # to avoid hitting rate limits and memory issues
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(self.process_single_url, url): url for url in urls}
            
            for i, future in enumerate(as_completed(future_to_url)):
                url = future_to_url[future]
                if i % 5 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"Processed {i}/{len(urls)} URLs ({i/len(urls)*100:.1f}%) in {elapsed:.1f}s")
                
                try:
                    url_dfs = future.result()
                    if url_dfs:
                        all_toxic_dfs.extend(url_dfs)
                        
                        # Accumulate all data without saving intermediates
                        # Only perform garbage collection to manage memory
                        if len(all_toxic_dfs) > 50:
                            gc.collect()
                            logger.info(f"Accumulated {len(all_toxic_dfs)} dataframes so far")
                except Exception as e:
                    logger.error(f"Error processing {url}: {str(e)}")
                    logger.error(traceback.format_exc())
        
        # Save all results using Dask if we have data
        if all_toxic_dfs:
            try:
                # Combine all DataFrames
                logger.info(f"Combining {len(all_toxic_dfs)} dataframes with Dask...")
                
                # Convert to pandas first to ensure schema consistency
                combined_df = pd.concat(all_toxic_dfs, ignore_index=True)
                
                # Create a Dask DataFrame with appropriate partitioning
                memory_usage = psutil.virtual_memory()
                available_mem_gb = memory_usage.available / (1024**3)
                # Adjust partitions based on available memory and dataframe size
                estimated_size_mb = combined_df.memory_usage(deep=True).sum() / (1024**2)
                partition_size_mb = 100  # Target partition size in MB
                num_partitions = max(4, min(20, int(estimated_size_mb / partition_size_mb)))
                
                logger.info(f"Creating Dask DataFrame with {num_partitions} partitions...")
                dask_df = dd.from_pandas(combined_df, npartitions=num_partitions)
                
                # Basic analysis with Dask
                logger.info("Performing distributed analysis with Dask...")
                
                # Group by repo_url and get count and mean toxicity
                analysis = dask_df.groupby('repo_url').agg({
                    'id': 'count',
                    'toxicity_score': ['mean', 'max']
                }).compute()
                
                # Sort by frequency of toxic comments
                analysis = analysis.sort_values(('id', 'count'), ascending=False)
                
                # Skip saving analysis results
                logger.info("Skipping analysis results file - only generating the main CSV")
                
                # Save the full results efficiently using Dask
                output_file = os.path.join(DEFAULT_CONFIG['output_dir'], 'all_toxic_repos.csv')
                logger.info(f"Writing final results to {output_file}")
                dask_df.to_csv(output_file, single_file=True, index=False)
                
                # Compute and return the final DataFrame
                result = dask_df.compute()
                logger.info(f"Completed processing with {len(result)} total toxic repos")
                return result
                
            except Exception as e:
                logger.error(f"Error saving consolidated results: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Try to recover partial results
                try:
                    if all_toxic_dfs:
                        recovery_df = pd.concat(all_toxic_dfs, ignore_index=True)
                        recovery_file = os.path.join(DEFAULT_CONFIG['output_dir'], 'recovered_toxic_repos.csv')
                        recovery_df.to_csv(recovery_file, index=False)
                        logger.info(f"Saved recovered {len(recovery_df)} toxic repos to {recovery_file}")
                        return recovery_df
                except:
                    logger.error("Failed to recover any results")
                
                return pd.DataFrame()  # Return empty DataFrame on error
        else:
            logger.info("No toxic repos found")
            return pd.DataFrame()  # Return empty DataFrame if no results

    def process_single_url(self, url):
        """Process a single URL and return list of DataFrame chunks with toxic content."""
        toxic_dfs = []
        try:
            for chunk_df in self.fetch_and_process_data(url):
                if not chunk_df.empty:
                    toxic_dfs.append(chunk_df)
        except Exception as e:
            logger.error(f"Error in process_single_url for {url}: {str(e)}")
        
        return toxic_dfs
    
# Example usage
if __name__ == "__main__":
    # Generate dates for all of 2023 and 2024 (up to present)
    def generate_date_range(start_date, end_date):
        date_list = []
        current_date = start_date
        while current_date <= end_date:
            date_list.append(current_date.strftime('%Y-%m-%d'))
            current_date += datetime.timedelta(days=1)
        return date_list
    
    # Generate dates for entire 2023 and most of 2024
    # For a full scan uncomment this - but beware it will be a lot of data!
    """
    all_dates = generate_date_range(
        datetime.datetime(2023, 1, 1),
        datetime.datetime(2024, 10, 31)
    )
    """
    
    # Option 1: Sample evenly throughout both years (1st of each month)
    sample_dates = []
    for year in [2023, 2024]:
        for month in range(1, 13):
            # Skip future months in 2024
            if year == 2024 and month > 10:
                continue
            sample_dates.append(f"{year}-{month:02d}-01")
    
    # Option 2: Key events and dates of interest
    key_dates = [
        # 2023 Quarterly samples
        "2023-01-15", "2023-04-15", "2023-07-15", "2023-10-15",
        
        # 2024 Quarterly samples
        "2024-01-15", "2024-04-15", "2024-07-15", "2024-10-15",
        
        # Major tech events
        "2023-03-14",  # Pi Day events
        "2023-05-10",  # Google I/O
        "2023-06-05",  # Apple WWDC
        "2023-09-12",  # Apple iPhone Event
        "2023-11-28",  # AWS re:Invent
        
        "2024-03-14",  # Pi Day events
        "2024-05-14",  # Google I/O
        "2024-06-10",  # Apple WWDC
        
        # Major incidents/controversies
        "2023-03-10",  # Silicon Valley Bank collapse
        "2023-11-17",  # OpenAI leadership crisis
        "2024-01-30",  # GitHub Copilot controversy
        "2024-07-19",  # CrowdStrike worldwide blue screen
        
        # Weekend vs. weekday comparison (two weeks in 2023)
        "2023-07-10", "2023-07-11", "2023-07-12",  # Mon-Wed
        "2023-07-15", "2023-07-16",                # Weekend
        
        # Weekend vs. weekday comparison (two weeks in 2024)
        "2024-03-04", "2024-03-05", "2024-03-06",  # Mon-Wed
        "2024-03-09", "2024-03-10",                # Weekend
    ]
    
    # Use either the sample_dates or key_dates based on preference
    # For comprehensive scanning, uncomment the all_dates line above
    dates_to_analyze = sample_dates + key_dates
    
    # Define hours to analyze
    # For more complete data, use a wider range of hours
    hours = [0, 6, 12, 18]  # Four times per day to sample different time zones
    
    # Alternatively, for more comprehensive analysis:
    # hours = range(24)  # All hours
    
    # Create and run the analyzer
    analyzer = GitHubArchiveAnalyzer(
        dates=dates_to_analyze,
        hours_range=hours,
        toxicity_threshold=0.6  # Slightly higher threshold to focus on more toxic content
    )
    
    toxic_repos_df = analyzer.collect_data()
    
    # Print summary statistics
    if not toxic_repos_df.empty:
        print(f"Total toxic repos found: {len(toxic_repos_df)}")
        
        # Group by date to see distribution across dates
        toxic_repos_df['date'] = pd.to_datetime(toxic_repos_df['created_at']).dt.date
        date_counts = toxic_repos_df.groupby('date').size()
        print("\nToxic comments by date:")
        for date, count in date_counts.items():
            print(f"{date}: {count} toxic comments")
            
        # Group by repo to see most toxic repositories
        repo_counts = toxic_repos_df.groupby('repo_url').size().sort_values(ascending=False).head(20)
        print("\nTop 20 repositories with most toxic comments:")
        for repo, count in repo_counts.items():
            print(f"{repo}: {count} toxic comments")
    else:
        print("No toxic repos found")