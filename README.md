<!-- messaged TA and he said it would help to have a readme file that  has some instructions -->
<!-- If you guys add to this, just write it in google docs and ask gpt to make it into markdown code if you dont know markdown for speed(im assuming they want this file done ASAP) -->

# Understanding & Mitigating Toxic Communication in OSS Communities

Our research project looks into the effects of toxic communication in OSS communities (specifically GitHub), analyzing its correlation with developer productivity, project outcomes, and contributor experience levels.

### Research Questions

1. Does toxic communication in OSS communities negatively affect programmer productivity, measured through commits, issue resolutions, and discussion activity?
2. Is there a correlation between toxic communication and software releases?
3. How does the level of experience of contributors (measured by account age and previous contributions) correlate with their likelihood of engaging in toxic communication?

### QUICK SUMMARY
1. Set up GitHub API tokens in a `.env` file:
```
GITHUB_ACCESS_TOKEN_1=your_token_1
GITHUB_ACCESS_TOKEN_2=''    (leave as '' if no other token)
GITHUB_ACCESS_TOKEN_3=''    (leave as '' if no other token)
```

2. `src/main.py` will load data from the selected repos using GitHub API (the data is already loaded in data/ so you can skip this step , unless you modified the code to get data from different timeperiods)
3. `src/visuals.py` will take the cleaned data from the data/ folder and do statistical analysis to determine correlations and create visuals used in our report in the folder src/visuals


## Repository Structure

- `list_of_repos_to_analyze/` - Contains 2 CSV files with repo information. We will analyze these repos for our analysis. One of the CSV ([incvility data set](https://github.com/vcu-swim-lab/incivility-dataset/blob/main/dataset/issue_threads.csv), we download from this directly as mentioned in report ). How we get the other dataset I tell in the next line.
- `src/go_thru_gharchive.py` - Processes GitHub Archive data for toxic repos. This file populates the othe CSV in list_of_repos_to_analyze/ to contain a list of repos with toxic comments from GH Archive for specific time periods we choose that we think might have toxicity like holidays. The repos listed in here will be used in our analysis. (idea to use GHArchive came from when the professor mentioned GHTorrent. GHTorrent isn't live anymore, but after trying to find data  from there, I found information about GHArchive)

- `src/.env` - Stores 3 GitHub API tokens, if only one token, leave the other 2 as an empty string ''
- `src/config.py` - Contains variables that we may use throughout the code, like toxicity threshold.
- `src/helper.py` - Contains miscellaneous functions like get_repos(from csv dataset) , save_csv, load_csv, etc...

- `src/main.py` - The first code user needs to run if they want to reload the data.**Note, the data will take time to fully run. We recommend not running this if you want to directly use the data we already collected** In main.py, it will call get_repos (in helper.py) which gets a list of repos we will analyze from the list_of_repos_to_analyze folder (with the incivilty dataset and the GHArchive dataset). Next it will call get_data which loads the required data we need.
- `src/get_data.py` - Handles data fetching from GitHub API (with API rotation system in place). This file will get the metrics we desire from each repo and write them into 5 CSV files in data/
- `src/toxicity_rater.py` - Toxicity rater code is in here and is used in get_data as it fetches the data/comments from repos. (Initially used Perspective API, but switched to [unitary/toxic-bert model](https://huggingface.co/unitary/toxic-bert))
- `data/` - Contains 5 CSV files generated in get_data.py, with the data we need to answer our RQ's. total_comments, total_commits, total_contributors, total_issues, total_releases, toxic_repo
- `src/visuals.py` - This code contains our visuals and statistics (spearman, pearson, etc..). First we load the data collected in the data/ directory and analyze the toxicity distribution relative to its percentile. (there were alot of non toxic outweighing the toxic ones, so this normalizes the data). Next we mark comments as toxic, based on the percentile I mentioned earlier. This has better results than a concrete toxicity threshold(which is declared in congfig.py) because it determines toxicity relative to the current dataset. (this has pros/cons, a con being if all comments are positive, the "less" postive ones might be considered toxic. For example: Have a good day(toxic) vs Have a GREAT day!(not toxic) ). Next, we have 3 functions that will run analysis on the data and create visuals with the results to answer each RQ (1,2,3).

- `src/get_toxic_issues.py` - Contains a redundant, unoptimized method to fetch the issues associated with the Incivility Dataset (a result of multiple group members working in parallel). The data is stored as an unfiltered dictionary in a pickle file which is not pushed to the remote to avoid git bloat.
- `src/get_commits.py` - Contains code to collect commits to a repository 180 days before and after the date each issue from the Incivility Dataset was locked and generates the plot located in `src/visuals/commits_vs_locked_date.png`. Data is unfiltered and therefore not pushed.
- `src/get_toxic_comments.py` - Contains redundant code to collect comments from all issues from the Incivility Dataset, retrieves each comment's toxicity rating, and generates the plot located in `src/visuals/toxicity_vs_delay.png`. Data is unfiltered and therefore not pushed.

## Data Sources

As said above, our project uses data from two primary sources:

1. [**Incivility Dataset**](https://github.com/vcu-swim-lab/incivility-dataset/): A publicly available dataset on GitHub that identifies a subset/sample size of uncivil interactions in issue threads within GitHub projects 

2. [**GHArchive Data**](https://www.gharchive.org/): A public archive of GitHub timeline events from selected dates focusing on holidays, big events, and significant incidents within the tech community.







## Team Members

- Sahil Dadhwal
- David Chu
- Manami Nakagawa
- Jordan Penner
- Haochen Dong

## Final Report

Full references can be found in the [final project report](the_final_report/final_report.pdf).
