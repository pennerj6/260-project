import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data():
    """Visualize the relationship between toxicity and productivity."""
    toxicity_df = pd.read_csv("toxicity_scores.csv")
    productivity_df = pd.read_csv("productivity.csv")
    
    # Merge datasets
    merged_df = pd.merge(toxicity_df, productivity_df, left_on="comment_author", right_on="author", how="inner")
    
    # Scatter plot: Toxicity vs. Productivity
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="toxicity_score", y="commit_count", data=merged_df)
    plt.title("Toxicity vs. Productivity")
    plt.xlabel("Toxicity Score")
    plt.ylabel("Commit Count")
    plt.savefig("toxicity_vs_productivity.png")
    plt.show()

if __name__ == "__main__":
    visualize_data()