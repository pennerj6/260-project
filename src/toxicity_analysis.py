import pandas as pd
import dask.dataframe as dd
import torch
from transformers import pipeline, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import gc
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ToxicityRater:
    def __init__(self):
        # Load the toxicity detection model
        self.model_name = "unitary/toxic-bert"
        self.device = 0 if torch.cuda.is_available() else -1
        self.toxicity_pipeline = pipeline("text-classification", model=self.model_name, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.max_length = 512  # Maximum sequence length for the model

    def get_toxicity_ratings(self, comments):
        """Process a list of comments and return toxicity scores using parallel batch processing."""
        if not comments:
            return []

        # Pre-process comments to handle long text
        processed_comments = []
        for comment in comments:
            try:
                if not isinstance(comment, str):
                    processed_comments.append("")
                    continue

                # Always truncate comments before processing
                encoding = self.tokenizer(
                    comment,
                    truncation=True,
                    max_length=self.max_length - 2,  # Account for special tokens
                    return_tensors="pt"
                )
                truncated_text = self.tokenizer.decode(encoding['input_ids'][0], skip_special_tokens=True)
                processed_comments.append(truncated_text)
            except Exception as e:
                logging.error(f"Error preprocessing comment: {str(e)[:100]}...")
                processed_comments.append("")  # Add empty string as placeholder

        batch_size = 8  # Reduced from 16 to avoid memory issues
        max_workers = 4  # Adjust based on CPU cores and available memory

        def process_batch(batch_idx):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(processed_comments))
            batch = processed_comments[start_idx:end_idx]

            if not batch:
                return [0] * len(batch)

            try:
                # Process the entire batch at once
                results = self.toxicity_pipeline(batch, truncation=True, max_length=self.max_length)
                batch_scores = [result['score'] if result['label'] == 'toxic' else 0 for result in results]
            except Exception as e:
                logging.error(f"Error processing batch: {str(e)[:100]}...")
                batch_scores = [0] * len(batch)

            return batch_scores

        # Calculate number of batches
        num_batches = (len(processed_comments) + batch_size - 1) // batch_size

        # Process batches in parallel
        all_scores = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {executor.submit(process_batch, i): i for i in range(num_batches)}
            for i in range(num_batches):
                for future in future_to_batch:
                    if future_to_batch[future] == i:
                        all_scores.extend(future.result())
                        break

        return all_scores

def analyze_toxicity():
    """Analyze toxicity of comments in issue comments."""
    try:
        # Use Dask to read the CSV file in chunks
        df = dd.read_csv("issue_comments.csv")

        # Check if the DataFrame has any rows
        if len(df.index) == 0:
            logging.info("No comments to analyze.")
            # Create an empty DataFrame with the required columns
            empty_df = pd.DataFrame(columns=["comment_body", "toxicity_score"])
            empty_df.to_csv("toxicity_scores.csv", index=False)
            return

        # Process comments in chunks using Dask
        df["toxicity_score"] = df["comment_body"].map_partitions(
            lambda x: ToxicityRater().get_toxicity_ratings(x.tolist()),
            meta=("toxicity_score", "float64")
        )

        # Compute the result and save to CSV
        result_df = df.compute()

        # Add toxicity categories for easier analysis
        result_df["toxicity_level"] = pd.cut(
            result_df["toxicity_score"],
            bins=[0, 0.1, 0.3, 0.5, 0.7, 1.0],
            labels=["Very Low", "Low", "Moderate", "High", "Very High"]
        )

        # Print toxicity distribution
        toxicity_distribution = result_df["toxicity_level"].value_counts().sort_index()
        logging.info("Toxicity Distribution:")
        logging.info(toxicity_distribution)

        # Save results
        result_df.to_csv("toxicity_scores.csv", index=False)
        logging.info("Toxicity analysis complete. Saved to toxicity_scores.csv.")

    except Exception as e:
        logging.error(f"Error in toxicity analysis: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    analyze_toxicity()