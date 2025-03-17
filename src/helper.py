import logging
import datetime
from datetime import datetime
import pandas as pd
import os
import csv
import sys

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
    
    # GH Archive dataset
    with open('list_of_repos_to_analyze/all_toxic_repos.csv', mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        
        for row in csv_reader:
            x = row['repo_url'].split('/')[-2:] #['Nyanotrasen', 'Nyanotrasen']
            own,repo = x                        #'Nyanotrasen' & 'Nyanotrasen'
            x = own + '/' + repo                #'Nyanotrasen/Nyanotrasen' this is waht we want 
            repo_urls.append(x)

    # incivilitty dataset
    with open('list_of_repos_to_analyze/incivility_dataset.csv', mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        
        for row in csv_reader:
            url = row['url'].split('/')
            
            repo_name = url[4] + '/' + url[5]
            repo_urls.append(repo_name)
            
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


def save_csv(filename, data):   
    folder = 'data'
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




# for the analysis in visuals.py
def load_csv(filename):
    
    # data = []
    # with open(filename, 'r', encoding='utf-8') as file:
    #     reader = csv.DictReader(file)
    #     for row in reader:
    #         data.append(row)
    # return data

    # files to big for default csv so were using manually setting the size limit
    # asked gpt to show how i can implement dynamically rather than a set limit
    current_limit = 1000000
    max_attempts = 10
    attempt = 0
    
    while attempt < max_attempts:
        try:
            # Set the current field size limit
            csv.field_size_limit(current_limit)
            
            # Try to read the file
            data = []
            with open(filename, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    data.append(row)
            
            # If we get here, the file was read successfully
            print(f"Successfully loaded CSV with field size limit: {current_limit}")
            return data
            
        except _csv.Error as e:
            if "field larger than field limit" in str(e):
                # Double the limit and try again
                current_limit *= 2
                attempt += 1
                print(f"Increasing field size limit to {current_limit} (attempt {attempt})")
            else:
                # If it's a different error, raise it
                raise
    
    # If we've exceeded max attempts
    raise ValueError(f"Failed to load CSV after {max_attempts} attempts. Last limit tried: {current_limit}")

def parse_date(date_string):
    # str to sate time
    if not date_string:
        return None
    # return datetime.fromisoformat(date_string.replace('Z', '+00:00'))
    # datestring is -> 2024-10-06T16:23:02Z
    # remove the 'Z' if Z is there
    date_string = date_string.replace('Z', '')
    try:
        # return the date time format, gpt helped w error handling and syntax 
        return datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S')
    except ValueError:
        # If there are microseconds
        try:
            return datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S.%f')
        except ValueError:
            print(f"Error parsing date: {date_string}")
            return None

def get_week_key(date):
    # make data into year-week rather than year-week-day-time (time was 00 anywasys)
    # it reutrns string not dataeimte
    # gpt reccommened i iuse the WEEK (like week 1 is days 1-7, week2.... ) like periods of time
    if not date:
        return None
    year = date.isocalendar()[0]
    week = date.isocalendar()[1]
    return f"{year}-{week}"
