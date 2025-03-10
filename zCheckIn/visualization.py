import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def visualize_data():
    """Visualize the relationship between toxicity and productivity."""
    try:
        # Create an output directory for visualizations
        vis_dir = "visualizations"
        os.makedirs(vis_dir, exist_ok=True)

        # Load data
        toxicity_df = pd.read_csv("toxicity_scores.csv")
        productivity_df = pd.read_csv("productivity.csv")

        if toxicity_df.empty or productivity_df.empty:
            logging.info("No data available for visualization.")
            return

        # Merge datasets (using appropriate columns)
        if "comment_author" in toxicity_df.columns and "author" in productivity_df.columns:
            merged_df = pd.merge(toxicity_df, productivity_df, left_on="comment_author", right_on="author", how="inner")
        else:
            logging.info("Required columns missing for merge. Cannot visualize data.")
            return

        if merged_df.empty:
            logging.info("No matching data between toxicity and productivity datasets.")
            return

        # Filter toxic comments (toxicity >= 0.5)
        toxic_df = merged_df[merged_df["toxicity_score"] >= 0.5]

        # Set up the visualization style
        sns.set(style="whitegrid")

        # 1. Scatter plot: Toxicity vs. Productivity (Toxic Comments Only)
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x="toxicity_score",
            y="commit_count",
            data=toxic_df,
            alpha=0.7,
            hue="has_conflict_keywords" if "has_conflict_keywords" in toxic_df.columns else None,
            size="comment_length" if "comment_length" in toxic_df.columns else None,
            sizes=(20, 200)
        )
        plt.title("Toxicity vs. Productivity (Toxic Comments Only)", fontsize=16)
        plt.xlabel("Toxicity Score", fontsize=14)
        plt.ylabel("Commit Count", fontsize=14)
        plt.savefig(os.path.join(vis_dir, "toxicity_vs_productivity_toxic.png"), dpi=300, bbox_inches="tight")

        # 2. Toxicity distribution (All Comments)
        plt.figure(figsize=(10, 6))
        if "toxicity_level" in merged_df.columns:
            sns.countplot(x="toxicity_level", data=merged_df, palette="viridis")
            plt.title("Distribution of Comment Toxicity Levels", fontsize=16)
        else:
            sns.histplot(merged_df["toxicity_score"], bins=15, kde=True)
            plt.title("Distribution of Toxicity Scores", fontsize=16)
        plt.xlabel("Toxicity", fontsize=14)
        plt.ylabel("Count", fontsize=14)
        plt.savefig(os.path.join(vis_dir, "toxicity_distribution.png"), dpi=300, bbox_inches="tight")

        # 3. Average Toxicity by Repository (Toxic Comments Only)
        if "repo" in toxic_df.columns:
            repo_toxicity = toxic_df.groupby("repo")["toxicity_score"].agg(
                ["mean", "count", "max"]
            ).sort_values(by="count", ascending=False).head(15)

            plt.figure(figsize=(14, 8))
            sns.barplot(x=repo_toxicity.index, y=repo_toxicity["mean"], palette="viridis")
            plt.title("Average Toxicity by Repository (Toxic Comments Only)", fontsize=16)
            plt.xticks(rotation=90)
            plt.xlabel("Repository", fontsize=14)
            plt.ylabel("Average Toxicity Score", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, "repo_toxicity_toxic.png"), dpi=300, bbox_inches="tight")

        # 4. Correlation Heatmap (Toxic Comments Only)
        if len(toxic_df.columns) > 5:
            numeric_cols = toxic_df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_patterns = ["number", "id", "index"]
            corr_cols = [col for col in numeric_cols if not any(pattern in col.lower() for pattern in exclude_patterns)]

            if len(corr_cols) > 1:
                plt.figure(figsize=(10, 8))
                corr_matrix = toxic_df[corr_cols].corr()
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(corr_matrix, mask=mask, annot=True, cmap="coolwarm", center=0)
                plt.title("Correlation Matrix of Metrics (Toxic Comments Only)", fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, "correlation_matrix_toxic.png"), dpi=300, bbox_inches="tight")

        # 5. Toxicity Over Time (Toxic Comments Only)
        if "created_at" in toxic_df.columns:
            try:
                toxic_df["created_at"] = pd.to_datetime(toxic_df["created_at"])
                toxic_df["date"] = toxic_df["created_at"].dt.date

                # Daily average toxicity
                daily_toxicity = toxic_df.groupby("date")["toxicity_score"].mean().reset_index()

                plt.figure(figsize=(14, 6))
                sns.lineplot(x="date", y="toxicity_score", data=daily_toxicity)
                plt.title("Daily Average Toxicity (Toxic Comments Only)", fontsize=16)
                plt.xlabel("Date", fontsize=14)
                plt.ylabel("Average Toxicity Score", fontsize=14)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, "toxicity_over_time_toxic.png"), dpi=300, bbox_inches="tight")
            except Exception as e:
                logging.error(f"Error in time-based analysis: {str(e)}")

        # 6. Comment Length vs. Toxicity (Toxic Comments Only)
        if "comment_length" in toxic_df.columns:
            plt.figure(figsize=(12, 8))
            sns.scatterplot(
                x="comment_length",
                y="toxicity_score",
                data=toxic_df,
                alpha=0.6,
                hue="toxicity_level" if "toxicity_level" in toxic_df.columns else None
            )
            plt.title("Comment Length vs. Toxicity (Toxic Comments Only)", fontsize=16)
            plt.xlabel("Comment Length (characters)", fontsize=14)
            plt.ylabel("Toxicity Score", fontsize=14)
            plt.savefig(os.path.join(vis_dir, "length_vs_toxicity_toxic.png"), dpi=300, bbox_inches="tight")

        # 7. Activity Ratio vs. Toxicity (Toxic Comments Only)
        if "activity_ratio" in toxic_df.columns:
            plt.figure(figsize=(12, 8))
            sns.scatterplot(
                x="activity_ratio",
                y="toxicity_score",
                data=toxic_df,
                alpha=0.6,
                hue="toxicity_level" if "toxicity_level" in toxic_df.columns else None,
                size="commit_count" if "commit_count" in toxic_df.columns else None,
                sizes=(20, 200)
            )
            plt.title("Developer Activity Ratio vs. Toxicity (Toxic Comments Only)", fontsize=16)
            plt.xlabel("Activity Ratio (active days / total days)", fontsize=14)
            plt.ylabel("Toxicity Score", fontsize=14)
            plt.savefig(os.path.join(vis_dir, "activity_vs_toxicity_toxic.png"), dpi=300, bbox_inches="tight")

        logging.info(f"Visualization complete. Images saved to {vis_dir} directory.")

    except Exception as e:
        logging.error(f"Error in visualization: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    visualize_data()