import os
from dotenv import load_dotenv

load_dotenv()

# GitHub API config
GITHUB_ACCESS_TOKEN = os.getenv('GITHUB_ACCESS_TOKEN')

BASE_URL = "https://api.github.com"

# Variables that we can mess around with
toxicity_threshold = 0.05 #0.01  # Threshold for classifying a comment as toxic (0.01 for now for testing)
analysis_window_days = 7  # Days before/after toxic comment to analyze
release_window_days = 14  # Days before release to analyze for toxicity increase

# Predefined issue URLs to analyze (from the Github DB https://github.com/vcu-swim-lab/incivility-dataset/blob/main/dataset/issue_threads.csv)
issue_urls = [
      "https://api.github.com/repos/doctrine/mongodb-odm/issues/554"
    , "https://api.github.com/repos/capistrano/capistrano/issues/440"
    , "https://api.github.com/repos/Leaflet/Leaflet/issues/2195"
    , "https://api.github.com/repos/composer/composer/issues/2545"
    
    , "https://api.github.com/repos/mysqljs/mysql/issues/731"
    , "https://api.github.com/repos/PCSX2/pcsx2/issues/10"
    , "https://api.github.com/repos/openzfs/zfs/issues/2611"
    , "https://api.github.com/repos/thoughtbot/factory_bot/issues/683"
    , "https://api.github.com/repos/scikit-learn/scikit-learn/issues/3808"
    
    # TODO: Add more issue URLs here
    # we can each choose differnt ones to run and write to different CSVs
    # and merge the CSVs into 1 for analysis
    # will write a script to directly get urls from here https://github.com/vcu-swim-lab/incivility-dataset/blob/main/dataset/issue_threads.csv
]

# Headers for GitHub API requests
headers = {
    'Authorization': f'token {GITHUB_ACCESS_TOKEN}',
    'Accept': 'application/vnd.github.v3+json'
}


