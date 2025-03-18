# sahil went to office hours 3/14 to ask how to cite AI
# - references, mention what you used AI for
# - "used it to help debug, refactor, api rate errors timing, token cycling, mem/timeout issues etc.."
# - helped choose key_dates to search (days where toxicity might be high) to find repos w toxicity to anallyze

import os
import logging
import sys
from dotenv import load_dotenv

# from github_analyzer import GitHubToxicityAnalyzer
from config import *
from helper import * # get_repos
from get_data import get_data_main

load_dotenv()
# Set up logging so we can see whats happenin
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)



def main():
    # Get GitHub tokens from env to be cycled (somewhat of a solution to the rate limit issue)
    # idk why but i need to call each variable separtely liek this otherwise theres a parsing error
    # commented out bc we ise them in get_data.py
    # x = os.getenv('GITHUB_ACCESS_TOKEN_1')
    # y = os.getenv('GITHUB_ACCESS_TOKEN_2')
    # z = os.getenv('GITHUB_ACCESS_TOKEN_3')
    # github_tokens = [x,y,z]
    
    repos = [
        # got the below 4 repos from get_repos but set the dates to when the Crowdstrike issue happeed last Summer (the global blue screen issue) (it affected my work, along w airlines and stuff so i figured it'd be toxic the day after)
        "BlueWallet/BlueWallet",
        "CodingPirates/forenings_medlemmer",
        "rust-lang/rust",
        "Fannovel16/ComfyUI-Frame-Interpolation",
        
    ]
    # get repos from GHArchive && i
    repos = repos + get_repos()

    # print(repos)
    print(len(repos))

    # ADD MORE FROM invinisibilty data set OR the GHArchive code i was working in my local
    #  repos = repos + get_incivility_data()

    # print(repos)
    print(f"This many repos: {len(repos)}")

    logger.info("-------STARTING LOOKING INTO REPO DATA NOW-------")    
    # Fetch data and write new CSV files for all repos (makes a lot of CSVs)
    # fetch_data(repos) # had to "wrap" the code in a get_data_main to define global variavles in the file
    get_data_main(repos)
    
    logger.info("----------------ALL DONE----------------")

if __name__ == "__main__":
    main()