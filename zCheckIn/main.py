import pandas as pd
from data_collection import parse_gharchive_data, save_data
from toxicity_analysis import analyze_toxicity
from productivity_analysis import calculate_productivity
from visualization import visualize_data
import os
from datetime import datetime, timedelta
import traceback
import gc
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    # Configuration
    dates = [
        "2023-01-15",  # January 2023
        "2023-04-21",  # April 2023
        "2023-07-15",  # July 2023
        "2023-10-01",  # October 2023
        "2023-10-15",  # Mid-October
        "2023-10-31",  # End of October
    ]
    hours = ["09", "12", "15", "18", "21"]  # Active hours

    logging.info("Starting GitHub toxicity analysis pipeline...")

    # Create a timestamped folder for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    # Step 1: Parse GHArchive data with checkpointing
    logging.info(f"Parsing GHArchive data from {len(dates)} dates and {len(hours)} hours per date...")
    all_issue_comments = []
    all_commits = []
    checkpoint_frequency = 3  # Save data every 3 date-hour combinations
    processed_count = 0

    for date in dates:
        for hour in hours:
            logging.info(f"Processing {date}-{hour}...")
            try:
                issue_comments_df, commits_df = parse_gharchive_data(date, hour)

                if not issue_comments_df.empty:
                    all_issue_comments.append(issue_comments_df)
                    logging.info(f"Found {len(issue_comments_df)} issue comments.")
                else:
                    logging.info(f"No issue comments found for {date}-{hour}.")

                if not commits_df.empty:
                    all_commits.append(commits_df)
                    logging.info(f"Found {len(commits_df)} commits.")
                else:
                    logging.info(f"No commits found for {date}-{hour}.")

                # Checkpoint data periodically
                processed_count += 1
                if processed_count % checkpoint_frequency == 0:
                    if all_issue_comments:
                        checkpoint_comments = pd.concat(all_issue_comments, ignore_index=True)
                        checkpoint_comments.to_csv(f"issue_comments_checkpoint_{processed_count}.csv", index=False)
                    if all_commits:
                        checkpoint_commits = pd.concat(all_commits, ignore_index=True)
                        checkpoint_commits.to_csv(f"commits_checkpoint_{processed_count}.csv", index=False)
                    logging.info(f"Checkpoint saved after processing {processed_count} date-hour combinations.")

            except Exception as e:
                logging.error(f"Error processing {date}-{hour}: {str(e)}")
                traceback.print_exc()

    # Combine all collected data
    if all_issue_comments:
        combined_issue_comments = pd.concat(all_issue_comments, ignore_index=True)
        logging.info(f"Total issue comments collected: {len(combined_issue_comments)}")
    else:
        combined_issue_comments = pd.DataFrame()
        logging.info("No issue comments collected across all dates and hours.")

    if all_commits:
        combined_commits = pd.concat(all_commits, ignore_index=True)
        logging.info(f"Total commits collected: {len(combined_commits)}")
    else:
        combined_commits = pd.DataFrame()
        logging.info("No commits collected across all dates and hours.")

    # Save the combined data
    save_data(combined_issue_comments, combined_commits)
    logging.info(f"Data parsing complete. Saved to {output_dir}/issue_comments.csv and {output_dir}/commits.csv.")

    # Free up memory before analysis
    del all_issue_comments
    del all_commits
    gc.collect()

    # Step 2: Analyze toxicity
    logging.info("Analyzing toxicity...")
    try:
        analyze_toxicity()
    except Exception as e:
        logging.error(f"Error during toxicity analysis: {str(e)}")
        traceback.print_exc()

    # Step 3: Analyze productivity
    logging.info("Analyzing productivity...")
    try:
        calculate_productivity()
    except Exception as e:
        logging.error(f"Error during productivity analysis: {str(e)}")
        traceback.print_exc()

    # Step 4: Visualize results
    logging.info("Visualizing results...")
    try:
        visualize_data()
    except Exception as e:
        logging.error(f"Error during visualization: {str(e)}")
        traceback.print_exc()

    logging.info(f"Analysis complete. All results are saved in the '{output_dir}' directory.")

if __name__ == "__main__":
    main()