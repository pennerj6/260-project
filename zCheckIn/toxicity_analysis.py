import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer

class ToxicityRater:
    def __init__(self):
        # Load the toxicity detection model
        self.model_name = "unitary/toxic-bert"
        self.toxicity_pipeline = pipeline("text-classification", model=self.model_name, device=0 if torch.cuda.is_available() else -1)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.max_length = 512  # Maximum sequence length for the model

    def get_toxicity_ratings(self, comments):
        """Process a list of comments and return toxicity scores."""
        if not comments:
            return []

        all_scores = []
        for comment in comments:
            try:
                tokens = self.tokenizer.encode(comment, add_special_tokens=True, truncation=False)
                if len(tokens) > self.max_length:
                    print(f"Truncating comment from {len(tokens)} tokens to {self.max_length} tokens")
                    # Use the tokenizer's built-in truncation
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
                    comment = truncated_text  # Use the truncated text for analysis
                
                # Process the comment
                result = self.toxicity_pipeline(comment)[0]
                score = result['score'] if result['label'] == 'toxic' else 0
                all_scores.append(score)
            except Exception as e:
                print(f"Error processing comment: {str(e)[:100]}...")
                all_scores.append(0)  # Default to non-toxic if an error occurs
        
        return all_scores

def analyze_toxicity():
    """Analyze toxicity of comments in issue comments."""
    df = pd.read_csv("issue_comments.csv")  # Read from issue_comments.csv instead of issues.csv
    comments = df["comment_body"].tolist()
    
    toxicity_rater = ToxicityRater()
    toxicity_scores = toxicity_rater.get_toxicity_ratings(comments)
    
    # Save results
    df["toxicity_score"] = toxicity_scores
    df.to_csv("toxicity_scores.csv", index=False)
    print("Toxicity analysis complete. Saved to toxicity_scores.csv.")

if __name__ == "__main__":
    analyze_toxicity()