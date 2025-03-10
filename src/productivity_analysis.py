import pandas as pd
import dask.dataframe as dd
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def calculate_productivity():
    """Calculate productivity metrics from commits."""
    try:
        start_time = time.time()
        logging.info("Starting productivity analysis...")
        
        # Read data with Dask instead of pandas
        commits_dd = dd.read_csv("commits.csv")
        logging.info(f"Loaded commits with Dask. Time: {time.time() - start_time:.2f}s")
        
        # Ensure the 'author' column exists
        if "author" not in commits_dd.columns:
            raise KeyError("The 'author' column is missing from commits.csv.")
        
        # Convert date column
        t0 = time.time()
        commits_dd["date"] = dd.to_datetime(commits_dd["date"])
        logging.info(f"Date conversion complete. Time: {time.time() - t0:.2f}s")
        
        # Extract date components - using Dask's methods
        t0 = time.time()
        commits_dd["month"] = commits_dd["date"].dt.strftime("%Y-%m")
        commits_dd["week"] = commits_dd["date"].dt.strftime("%Y-%U")
        commits_dd["day"] = commits_dd["date"].dt.floor('D')
        logging.info(f"Date components extracted. Time: {time.time() - t0:.2f}s")
        
        # For operations that need full data, compute at appropriate points
        t0 = time.time()
        commits_df = commits_dd.compute()
        logging.info(f"Dask DataFrame computed to pandas. Time: {time.time() - t0:.2f}s")
        
        # Use pandas for the groupby operations since we've already computed to pandas
        t0 = time.time()
        productivity_df = commits_df.groupby("author").size().reset_index(name="commit_count")
        monthly_commits = commits_df.groupby(["author", "month"]).size().reset_index(name="monthly_commits")
        weekly_commits = commits_df.groupby(["author", "week"]).size().reset_index(name="weekly_commits")
        daily_commits = commits_df.groupby(["author", "day"]).size().reset_index(name="daily_commits")
        logging.info(f"Group by operations completed. Time: {time.time() - t0:.2f}s")
        
        # Author stats section - OPTIMIZED VERSION
        t0 = time.time()
        # Calculate active days for each author in a vectorized way
        active_days_df = commits_df.groupby("author")["day"].nunique().reset_index()
        active_days_df.columns = ["author", "active_days"]
        
        # Calculate min and max dates for each author
        date_range_df = commits_df.groupby("author")["date"].agg(["min", "max"]).reset_index()
        
        # Calculate total days and activity ratio
        date_range_df["total_days"] = (date_range_df["max"] - date_range_df["min"]).dt.days + 1
        date_range_df["total_days"] = date_range_df["total_days"].fillna(0).astype(int)
        
        # Merge with active days
        author_stats_df = pd.merge(active_days_df, date_range_df[["author", "total_days"]], on="author", how="left")
        
        # Calculate activity ratio
        author_stats_df["activity_ratio"] = author_stats_df["active_days"] / author_stats_df["total_days"].clip(lower=1)
        
        logging.info(f"Author stats calculated (vectorized). Time: {time.time() - t0:.2f}s")
            
        # Create extended productivity dataframe
        productivity_df = pd.merge(productivity_df, author_stats_df, on="author", how="left")
        
        # Save results
        productivity_df.to_csv("productivity.csv", index=False)
        monthly_commits.to_csv("monthly_commits.csv", index=False)
        weekly_commits.to_csv("weekly_commits.csv", index=False)
        daily_commits.to_csv("daily_commits.csv", index=False)
        logging.info(f"Productivity analysis complete. Results saved to CSV files. Total time: {time.time() - start_time:.2f}s")
        
    except Exception as e:
        logging.error(f"Error in productivity analysis: {str(e)}")
        # Create empty productivity file if needed for the pipeline to continue
        pd.DataFrame(columns=["author", "commit_count"]).to_csv("productivity.csv", index=False)

if __name__ == "__main__":
    calculate_productivity()