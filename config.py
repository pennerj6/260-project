# Default configuration parameters
DEFAULT_CONFIG = {
    'start_date': '2024-01-01',
    'end_date': '2024-01-07',
    'hours': range(6, 22),  # Only look at these hours in a day, hopefilly speeds things iup
    'output_dir': 'output',
    'toxicity_threshold': 0.5,  # set toxicity threshfold to 0.5
    'use_sampling': True,
    'sample_size': 1000, # only read smaple size of comments (will chnage to only read these many repos per daytoo)
    # chatgpt reccommended the following limits to avoid a bunch of random issues like w memory/time and it works so far, we can mess around w this to see what high we can go
    'max_workers': 5, 
    'dask_partitions': 10,
    'batch_size': 32,  
    'chunk_size': 3,  
    'memory_limit_factor': 0.7  
}

# Columns required in the raw data
INITIAL_REQUIRED_COLUMNS = [
    'id', 'type', 'created_at', 'actor.login', 'payload.comment.body', 
    'payload.issue.number', 'payload.action', 'repo.name'
]

# Columns added during toxicity processing
TOXICITY_COLUMNS = [
    'toxicity_score', 'is_toxic'
]

# Combine both lists for later use
REQUIRED_COLUMNS = INITIAL_REQUIRED_COLUMNS + TOXICITY_COLUMNS
