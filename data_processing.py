import pandas as pd
import os
import dask.dataframe as dd
from .config import DEFAULT_CONFIG
import logging

logger = logging.getLogger(__name__)

def save_to_csv(df, path):
    """Helper to save DataFrame to CSV"""    
    # NOTE: need to make this a zip of the CSV to save mem
    #but idont think our project is going to be THAT big, we will see
    df.to_csv(path, index=False)
    logger.info(f"Saved data to {path}")

def load_from_csv(path):
    """Helper to get data from CSV to DataFrame"""
    # when jordan mentioned "caching", i thought it might be good to "reuse" the CSV's when generating figures rather than wating for the GHArchives to fully download (if it is alreadt loaded of course)
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def process_toxicity_scores(df, rater, batch_size=32):
    """Process toxicity scores for comments"""
    # BIGG ISSUE, payload.comment.body throws erros so i flatted df so it will be inclided (i wasnt sure how else to check for df['payload]['comment']['body'] in df/pandas )
    if 'payload.comment.body' not in df.columns:
        logger.warning("No comments found in the data. (NO payload.comment.body)")
        return df

    comments = df['payload.comment.body'].fillna('').tolist()
    toxicity_scores = rater.get_toxicity_ratings(comments, batch_size=batch_size)

    df['toxicity_score'] = toxicity_scores
    df['is_toxic'] = df['toxicity_score'] >= DEFAULT_CONFIG['toxicity_threshold']
    return df