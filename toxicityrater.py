from googleapiclient import discovery


"""
Class to compute a toxicity rating on comments, one comment at a time for now.

Toxicity is given as a score between 0 and 1.
This uses the Perspective API which by default allows one request per second.
We can make a request to increase usage if we want to.
"""
class ToxicityRater:
    def __init__(self):
        API_KEY = "AIzaSyAYOjEq0AJDZ9hxByoseBriI3S8hNZqZb8"
        self.client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=API_KEY,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )

    def get_toxicity_rating(self, comment: str):
        analyze_request = {
            'comment': { 'text': comment },
            'requestedAttributes': {'TOXICITY': {}}
        }

        response = self.client.comments().analyze(body=analyze_request).execute()
        return response['attributeScores']['TOXICITY']['summaryScore']['value']

if __name__ == "__main__":
    tr = ToxicityRater()
    toxicity = tr.get_toxicity_rating("Have a great day!")
    print(f"toxicity rating: {toxicity}")
