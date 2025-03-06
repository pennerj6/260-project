from googleapiclient import discovery
from googleapiclient.errors import HttpError

import time

import os
from dotenv import load_dotenv
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer



# Perspective API was very slow, after consulting w chatgpt 
# it reccommened to look into using different toxicity models such asthis one:
# https://huggingface.co/unitary/toxic-bert  # perspective API code is still commented out below
# there also was an issue about comment lengths("tokens") being too long, so we shortened truncated comments to avoid issues

class ToxicityRater:
    def __init__(self):
        # Load the toxicity detection model
        self.model_name = "unitary/toxic-bert"
        self.toxicity_pipeline = pipeline("text-classification", model=self.model_name, device=0 if torch.cuda.is_available() else -1)
        # Load the tokenizer for proper token counting
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.max_length = 512  # Maximum sequence length for the model

    def get_toxicity_ratings(self, comments, batch_size=8):
        """
        Process a list of comments in batches and return toxicity scores.
        Properly truncates comments that exceed the model's maximum token length.
        """
        if not comments:
            return []

        all_scores = []
        
        # Process comments individually to ensure maximum reliability
        for comment in comments:
            try:
                # Encode the comment to get tokens and check length
                tokens = self.tokenizer.encode(comment, add_special_tokens=True, truncation=False)
                
                # If too long, use the tokenizer's built-in truncation 
                # (this ensures proper truncation with model-specific details)
                if len(tokens) > self.max_length:
                    print(f"Truncating comment from {len(tokens)} tokens to {self.max_length} tokens")
                    # Let the tokenizer handle truncation properly with model-specific rules
                    encoding = self.tokenizer(
                        comment, 
                        truncation=True, 
                        max_length=self.max_length,
                        return_tensors="pt"
                    )
                    # Decode back to text
                    truncated_text = self.tokenizer.decode(
                        encoding['input_ids'][0], 
                        skip_special_tokens=True
                    )
                    
                    # Process the truncated comment
                    result = self.toxicity_pipeline(truncated_text)[0]
                else:
                    # Process the original comment if it's not too long
                    result = self.toxicity_pipeline(comment)[0]
                
                # Extract the score
                score = result['score'] if result['label'] == 'toxic' else 0
                all_scores.append(score)
                
            except Exception as e:
                print(f"Error processing comment: {str(e)[:100]}...")
                # If error still occurs, fallback to more aggressive truncation
                try:
                    # Try with even shorter length as a fallback
                    safe_length = min(self.max_length - 50, 450)  # Extra safety margin
                    print(f"Attempting fallback truncation to {safe_length} tokens")
                    encoding = self.tokenizer(
                        comment, 
                        truncation=True, 
                        max_length=safe_length,
                        return_tensors="pt"
                    )
                    very_safe_text = self.tokenizer.decode(
                        encoding['input_ids'][0], 
                        skip_special_tokens=True
                    )
                    result = self.toxicity_pipeline(very_safe_text)[0]
                    score = result['score'] if result['label'] == 'toxic' else 0
                    all_scores.append(score)
                except Exception as e:
                    print(f"Fallback failed: {str(e)[:100]}... Defaulting to non-toxic")
                    all_scores.append(0)  # Default to non-toxic
        
        return all_scores

    def get_toxicity_ratings_batched(self, comments, batch_size=8):
        """
        A batched version that processes multiple comments at once for better performance.
        Only use this if the individual processing is too slow and you're willing to risk some errors.
        """
        if not comments:
            return []

        all_scores = []
        
        # Process in batches
        for i in range(0, len(comments), batch_size):
            batch = comments[i:i + batch_size]
            truncated_batch = []
            
            # Properly truncate each comment in the batch
            for comment in batch:
                # Use the tokenizer's built-in truncation
                encoding = self.tokenizer(
                    comment, 
                    truncation=True, 
                    max_length=self.max_length - 10,  # Extra safety margin
                    return_tensors="pt"
                )
                truncated_text = self.tokenizer.decode(
                    encoding['input_ids'][0], 
                    skip_special_tokens=True
                )
                truncated_batch.append(truncated_text)
            
            try:
                # Process the batch
                results = self.toxicity_pipeline(truncated_batch)
                # Extract scores
                batch_scores = [result['score'] if result['label'] == 'toxic' else 0 for result in results]
                all_scores.extend(batch_scores)
            except Exception as e:
                print(f"Error in batch processing: {e}")
                # Fall back to individual processing
                individual_scores = self.get_toxicity_ratings(batch, batch_size=1)
                all_scores.extend(individual_scores)
                
        return all_scores

if __name__ == "__main__":
    # Test the ToxicityRater
    tr = ToxicityRater()
    comments = [
        "Have a great day fool!",
        "You are so stupid love!",
        "This is a boring project!",
        "I hate you",
        "I love apples",
        # Long comment test
        "This is a very long comment " * 200,  # Will be properly truncated
    ]
    
    # Test individual processing (more reliable)
    print("\nIndividual processing:")
    individual_scores = tr.get_toxicity_ratings(comments)
    for comment, toxicity in zip(comments, individual_scores):
        print(f"Comment: {comment[:50]}... | Toxicity: {toxicity}")
        
    # Optional: Test batch processing (faster but less reliable)
    print("\nBatch processing:")
    batch_scores = tr.get_toxicity_ratings_batched(comments)
    for comment, toxicity in zip(comments, batch_scores):
        print(f"Comment: {comment[:50]}... | Toxicity: {toxicity}")
# USING PERSPECTIVE API: (super slow)
"""
if 1 == 0:
    # Class to compute a toxicity rating on comments, one comment at a time for now.

    # Toxicity is given as a score between 0 and 1.
    # This uses the Perspective API which by default allows one request per second.
    # We can make a request to increase usage if we want to.
    load_dotenv()
    # GitHub API config
    PERSPECTIVE_API_KEY = os.getenv('PERSPECTIVE_API_KEY')

    class ToxicityRater:
        def __init__(self):
            API_KEY = PERSPECTIVE_API_KEY
            self.client = discovery.build(
                "commentanalyzer",
                "v1alpha1",
                developerKey=API_KEY,
                discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                static_discovery=False,
            )

        def get_toxicity_rating(self, comment: str, language="en"):
            # Hnadle Perspective API text length limit error 
            max_bytes = 20480  # Perspective API's limit 
            if len(comment.encode('utf-8')) > max_bytes:
                # Truncate the comment to the first 20,480 bytes
                truncated_text = comment.encode('utf-8')[:max_bytes].decode('utf-8', errors='ignore')
                print(f"Truncated comment from {len(comment.encode('utf-8'))} bytes to {len(truncated_text.encode('utf-8'))} bytes")
                comment = truncated_text
            
            
            analyze_request = {
                'comment': { 'text': comment },
                'requestedAttributes': {'TOXICITY': {}},
                'languages': [language]  # Explicitly specify English 

            }
            try:
                response = self.client.comments().analyze(body=analyze_request).execute()
                time.sleep(1)  # Add a 1-second delay between requests
                return response['attributeScores']['TOXICITY']['summaryScore']['value']
            except HttpError as e:
                print(f"Error analyzing comment: {e}")
                return 0

            # response = self.client.comments().analyze(body=analyze_request).execute()
            # return response['attributeScores']['TOXICITY']['summaryScore']['value']

    if __name__ == "__main__":
        tr = ToxicityRater()
        toxicity = tr.get_toxicity_rating("Have a great day!")
        print(f"toxicity rating: {toxicity}")

"""