from dateutil import parser
from datetime import timedelta
from api_requests import get_all_pages
from config import *

# I dont know if PyDriller is suitable to use bc we have to download each repo locally?
# I could be wrong^

# Get the commits of a repo between a date range
def get_repo_commits(repo_owner, repo_name, start_date, end_date):
    url = f"{BASE_URL}/repos/{repo_owner}/{repo_name}/commits"
    params = {
        'since': start_date.isoformat(),
        'until': end_date.isoformat()
    }
    
    return get_all_pages(url, params)
