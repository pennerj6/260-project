import os
import requests

from dotenv import load_dotenv
load_dotenv()

# Load the GitHub token from the environment variable
GITHUB_ACCESS_TOKENS = [
    #"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", # MINE
    # "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",
    # "59ff100f27dff59353b1daa18c2bb3e8a7c35e56",
    # "26ebce2ec7e0d50e18519212b6667275156bc852",
    # "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",
    # "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",
    # "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",
    # "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",
    # "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",
    # "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",
    # "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",
    # "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",
    # "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",
    # "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",
    # "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",
    # "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",
    # "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",
    # "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",
    # "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",
    # "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",
    # "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",
    # "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",
    # "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",
    # "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",
    # "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",
    "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",
    "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",
    "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",
    "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"

]

# GitHub API URL for authentication test
url = "https://api.github.com/user"

# Iterate over the tokens and check their validity
for index, token in enumerate(GITHUB_ACCESS_TOKENS):
    # Set up headers for the request
    headers = {
        'Authorization': f'token {token.strip()}'
    }

    # Send the request
    response = requests.get(url, headers=headers)

    # Check the response status
    if response.status_code == 200:
        print(f"Token at index {index} is GOOD.")
    else:
        print(f"Token at index {index} is BAD. Status code: {response.status_code}")
        #print("Response:", response.json())
