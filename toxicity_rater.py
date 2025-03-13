import torch
import numpy as np
import logging
import gc
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed
from transformers import pipeline, AutoTokenizer
import time

# Setup logging instead of print statments to show currnent TIME 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ToxicityRater:
    def __init__(self, use_sampling, sample_size):
        # Jordan found Perspective API, btu the issue with that was the api rate limit timing out constantly 
        # Used gpt to give a few differnt suggestions for toxicity calcualtors that DONT have a rate limit
        # This one from HuggingFace seems to do the trick https://huggingface.co/unitary/toxic-bert
        # After a few manual tests/comparisons, it was generating very similar toxicity scores as perspective api
        self.model_name = "unitary/toxic-bert"
        self.device = -1  # GPT said GPU is faster but idk how to activate it (or even if i have access to it w/o a VM) 
        
        logger.info(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        self.model = pipeline(
            "text-classification",
            model=self.model_name,
            device=self.device,
            framework="pt"
        )

        # errors appear if comments are too long, and we have to truncate
        # PAPER: we have to truncate the comment at 512 (letters? or bits? )
        self.max_length = 512
        self.use_sampling = use_sampling
        self.sample_size = sample_size

        logger.info("warming up model, hello world")
        _ = self.model(["hello this is a warm up text"])
        gc.collect()

    @torch.no_grad()
    def get_toxicity_ratings(self, comments, batch_size=32):
        """Given a lsit of commetns, return their toxicity socre """
        if not comments:
            return []

        start_time = time.time()
        logger.info(f"Processing {len(comments)} comments...")

        if self.use_sampling and len(comments) > self.sample_size:
            # if there are more comments than the sample size (in congif), we will choose X comments ones (x is the sample size)
            logger.info(f"Sampling {self.sample_size} comments from {len(comments)} total")
            
            indices = np.random.choice(len(comments), self.sample_size, replace=False)
            sampled_comments = [comments[i] for i in indices]
            results = np.zeros(len(comments))
            
            sample_results = self._process_comments(sampled_comments, batch_size)
            
            for idx, result_idx in enumerate(indices):
                results[result_idx] = sample_results[idx]
            return results.tolist()
        else:
            return self._process_comments(comments, batch_size)

    # was having some issues with memory and proccessing alot of data, so gpt reccommended processing in batches w parallel processing
    def _process_comments(self, comments, batch_size):
        """Process all comments using batching"""
        # Use 75% of available CPU cores for optimal performance (im not really sure why this is considered optimal, why not higher than 75?)
        n_jobs = max(1, int(multiprocessing.cpu_count() * 0.75))
        logger.info(f"Using {n_jobs} CPU cores for processing")

        # Break into smaller batches to avoid memory issues
        num_batches = (len(comments) + batch_size - 1) // batch_size
        batches = [comments[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]

        # Process batches with progress reporting
        all_scores = []
        # Use smaller chunks for parallel processing to avoid memory spikes
        chunk_size = 10  # Process 10 batches at a time, might increase later to 25?
        for i in range(0, len(batches), chunk_size):
            chunk_batches = batches[i:i + chunk_size]
            logger.info(f"Processing batch chunk {i // chunk_size + 1}/{(len(batches) + chunk_size - 1) // chunk_size}")

            chunk_scores = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(self._process_batch)(batch)
                for batch in tqdm(chunk_batches, desc=f"Processing Batches {i}-{i + len(chunk_batches) - 1}")
            )
            all_scores.extend(chunk_scores)

            # Force garbage collection after each chunk
            gc.collect()

        # Flatten the scores
        return [score for batch_scores in all_scores for score in batch_scores]

    def _process_batch(self, batch):
        """Processes a single batch of comments"""
        # before i asked gpt to help speed up the toxicity calculation process, i was geting the score of each individual comment 1 at a time rather than bathces like here
        if not "".join(batch).strip():
            return [0.0] * len(batch)
        try:
            results = self.model(batch, truncation=True, max_length=self.max_length)
            return [result['score'] if result['label'] == 'toxic' else 0 for result in results] # i think i will force all data to be "toxic" (greater than 0.50)
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)[:100]}...")
            return [0.0] * len(batch)

    # def test_toxicity(self, sentence):
    #     """Test the toxicity checker with a given sentence"""
    #     result = self.model([sentence], truncation=True, max_length=self.max_length)
    #     return result
