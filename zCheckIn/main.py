from data_collection import parse_gharchive_data
from toxicity_analysis import analyze_toxicity
from productivity_analysis import calculate_productivity
from visualization import visualize_data


def main():
    # Step 1: Parse GHArchive data
    print("Parsing GHArchive data...")
    issue_comments_df, commits_df = parse_gharchive_data("2023-10-01", "15")
    issue_comments_df.to_csv("issue_comments.csv", index=False)
    commits_df.to_csv("commits.csv", index=False)
    print("Data parsing complete. Saved to issue_comments.csv and commits.csv.")

    # Step 2: Analyze toxicity
    print("Analyzing toxicity...")
    analyze_toxicity() # low priority issue: it will truncate long messages when determining toxicity %, idk how to work around that
    print("Toxicity analysis complete. Saved to toxicity_scores.csv.")

    # Step 3: Analyze productivity
    print("Analyzing productivity...")
    calculate_productivity()
    print("Productivity analysis complete. Saved to productivity.csv and monthly_commits.csv.")

    # Step 4: Visualize results
    print("Visualizing results...")
    visualize_data()
    print("Visualization complete. Saved to toxicity_vs_productivity.png.")

if __name__ == "__main__":
    main()