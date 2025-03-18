SAMPLE_SIZE = 1000 #100   # smaller is faster


# toxicity threshiold
TOXICITY_THRESHOLD = 0.5  # Comments with score > 0.5 are considered toxic, might change to 0.7

# Analysis window sizes 
DEFAULT_WINDOW_DAYS = 30 # ab a month  
RELEASE_ANALYSIS_WINDOW = 14 # 2 weeks

# save code to these folders
# DATA_DIR = "data"
RESULTS_DIR = "results"
STATS_RESULTS_DIR = "results_stats" #"stats_results" changed name so results folders are alpha together

# GitHub API settings
GITHUB_BASE_URL = "https://api.github.com"
API_REQUEST_TIMEOUT = 30  # api kept timing out so need to have 30 sec sleep when it timeout
RATE_LIMIT_THRESHOLD = 10  # i made 3 diff api tokens so i can rotate them to avoid rate issue, they rotate after ever 10 calls

# Test mode settings -- because of the rate limit issues, chatgpt siggesting added a "switch" for test mode, which limits how much data it fetches
TEST_MODE_ITEMS_LIMIT = 10 #5  # wgen test mode is Fasle in the code, this limit will be ignoreed & computation will be super slow (which is why i always leave it on)
