
from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class ToxicityRater:
    def __init__(self):
        # Load the pre-trained toxicity model form hugging face
        logger.info("Loading toxicity model...")
        self.model = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            device=-1
        )
    
    def get_toxicity(self, text):
        result = self.model(text, truncation=True, max_length=512)
        # Return score if toxic, otherwise 0
        return result[0]['score'] if result[0]['label'] == 'toxic' else 0
    

# Simple example usage
# if __name__ == "__main__":
#     rater = ToxicityRater()
#     # Test with a single comment
#     comment = "Hello world ass hate"
#     score = rater.get_toxicity(comment)
#     print(score)
    
    