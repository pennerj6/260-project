import pandas as pd

def calculate_productivity():
    """Calculate productivity metrics from commits."""
    commits_df = pd.read_csv("commits.csv")
    
    # Ensure the 'author' column exists
    if "author" not in commits_df.columns:
        raise KeyError("The 'author' column is missing from commits.csv.")
    
    # Group by author and count commits
    productivity_df = commits_df.groupby("author").size().reset_index(name="commit_count")
    
    # Normalize by time (example: commits per month)
    commits_df["date"] = pd.to_datetime(commits_df["date"])
    commits_df["month"] = commits_df["date"].dt.to_period("M")
    monthly_commits = commits_df.groupby(["author", "month"]).size().reset_index(name="monthly_commits")
    
    # Save results
    productivity_df.to_csv("productivity.csv", index=False)
    monthly_commits.to_csv("monthly_commits.csv", index=False)
    print("Productivity analysis complete. Saved to productivity.csv and monthly_commits.csv.")

if __name__ == "__main__":
    calculate_productivity()