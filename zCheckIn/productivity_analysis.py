import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def calculate_productivity():
    """Calculate productivity metrics from commits."""
    try:
        commits_df = pd.read_csv("commits.csv")

        # Ensure the 'author' column exists
        if "author" not in commits_df.columns:
            raise KeyError("The 'author' column is missing from commits.csv.")

        # Group by author and count commits
        productivity_df = commits_df.groupby("author").size().reset_index(name="commit_count")

        # Add more productivity metrics
        commits_df["date"] = pd.to_datetime(commits_df["date"])
        commits_df["month"] = commits_df["date"].dt.to_period("M")
        commits_df["week"] = commits_df["date"].dt.to_period("W")
        commits_df["day"] = commits_df["date"].dt.date

        # Monthly activity
        monthly_commits = commits_df.groupby(["author", "month"]).size().reset_index(name="monthly_commits")

        # Weekly activity
        weekly_commits = commits_df.groupby(["author", "week"]).size().reset_index(name="weekly_commits")

        # Daily activity
        daily_commits = commits_df.groupby(["author", "day"]).size().reset_index(name="daily_commits")

        # Consistency metrics
        author_stats = []
        for author in productivity_df["author"]:
            author_commits = commits_df[commits_df["author"] == author]

            # Calculate days with activity
            active_days = author_commits["day"].nunique()

            # Calculate total days in dataset
            if len(author_commits) > 0:
                total_days = (author_commits["date"].max() - author_commits["date"].min()).days + 1
                activity_ratio = active_days / max(total_days, 1)
            else:
                total_days = 0
                activity_ratio = 0

            author_stats.append({
                "author": author,
                "active_days": active_days,
                "total_days": total_days,
                "activity_ratio": activity_ratio
            })

        # Create extended productivity dataframe
        extended_productivity = pd.DataFrame(author_stats)
        productivity_df = pd.merge(productivity_df, extended_productivity, on="author", how="left")

        # Save results
        productivity_df.to_csv("productivity.csv", index=False)
        monthly_commits.to_csv("monthly_commits.csv", index=False)
        weekly_commits.to_csv("weekly_commits.csv", index=False)
        daily_commits.to_csv("daily_commits.csv", index=False)
        logging.info("Productivity analysis complete. Results saved to CSV files.")

    except Exception as e:
        logging.error(f"Error in productivity analysis: {str(e)}")
        # Create empty productivity file if needed for the pipeline to continue
        pd.DataFrame(columns=["author", "commit_count"]).to_csv("productivity.csv", index=False)

if __name__ == "__main__":
    calculate_productivity()