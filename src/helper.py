import logging
import datetime
import pandas as pd
import os

logger = logging.getLogger(__name__)

# get issue number from url
def get_issue_number(url):
    try:
        return int(url.split('/')[-1])
    except:
        return 0

# i compleletly forgot i made this file to store small basic functions like getting issue num from url
# if theres time ill move some functions in here from the github_analyzer file (since i define most everything in that file)

import csv

# the code from go_thru_gharchive used GHArchive to search for toxic comments thru a bunch of dates we manually entered (i asked gpt what days it thought would contain toxic code, and it helped w that, like it chose major release days, holdau, ext.)
# the data from there gets stored in a CSV, in that csv we will give the repoowner and the repo name
# this is where we will choose which repos to analyze 
# itsn ot gureenteed to all be toxic (it could kust be 1 toxic commetn for example)
def get_repos():
    repo_urls = []
    
    with open('list_of_repos_to_analyze/all_toxic_repos.csv', mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        
        for row in csv_reader:
            x = row['repo_url'].split('/')[-2:] #['Nyanotrasen', 'Nyanotrasen']
            own,repo = x                        #'Nyanotrasen' & 'Nyanotrasen'
            x = own + '/' + repo                #'Nyanotrasen/Nyanotrasen' this is waht we want 
            repo_urls.append(x)

    # it gonna return this stricture
    # [ [owner,reponame] , [own,rn], [own,rn]]  
    return repo_urls

# repo_urls = get_repos()

# for url in repo_urls:
#     print(url)

def convert_to_dataframes(hash_map):
    # convert data to pandas df
    # i personally dont like to use pandas datafram bc idk the syntax but, professor reccommeneded/mentioned it (and its well know/easyier to debug w so we will use it!)
    dataframes = {}
    # go thru each key/val pair and 
    for key, value in hash_map:
        logger.info(f"Converting {len(value)} {key} items to DataFrame")
        if value:
            dataframes[key] = pd.DataFrame(value)
            
            #str to daatetime
            for col in dataframes[key].columns:
                # might edge case for ifinstatnce not string
                if 'date' in col or col.endswith('_at'):
                    dataframes[key][col] = pd.to_datetime(dataframes[key][col])
    
    logger.info(f"Created dataframes: {list(dataframes.keys())}")
    return dataframes


def save_csv(results_dict, base_filename):
    # need to handle 2025-03-14 14:30:52,107 - ERROR - Error saving results to CSV: 'dict' object has no attribute 'empty'
    try:
        os.makedirs(os.path.dirname(base_filename), exist_ok=True)
        for key, df in results_dict.items():
            if df is not None and hasattr(df, 'empty') and not df.empty:
                filename = f"{base_filename}_{key}.csv"
                df.to_csv(filename, index=False)
                logger.info(f"Saved {len(df)} rows to {filename}")
            elif isinstance(df, dict):
                # Handle dictionary data by converting to DataFrame first
                dict_df = pd.DataFrame([df])
                filename = f"{base_filename}_{key}.csv"
                dict_df.to_csv(filename, index=False)
                logger.info(f"Saved dictionary data to {filename}")
    except Exception as e:
        logger.error(f"Error saving results to CSV: {str(e)}")



def save_csv2(filename, data):   
    folder = 'data'
    
    print(data)
    print(data[0])
    # Create output directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    # Skip empty lists
    if not data:
        return
    
    # Create filename
    output_path = os.path.join(folder, f"total_{filename}.csv")
    
    try:
        # Get field names from the first dictionary
        if data and isinstance(data[0], dict):
            fieldnames = data[0].keys()
            
            # Write data to CSV
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # Handle potential missing keys in some dictionaries
                for item in data:
                    row = {field: item.get(field, '') for field in fieldnames}
                    writer.writerow(row)
    except Exception as e:
        print(f"ISSUE WRITING HERE: {e}")
