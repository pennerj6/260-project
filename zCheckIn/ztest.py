import requests
import json

# Define the date and hour you want to fetch data for
date = "2025-03-08"  # Replace with your desired date
hour = "10"          # Replace with your desired hour (0-23)

# Construct the URL for the GH Archive file
url = f"https://data.gharchive.org/{date}-{hour}.json.gz"

# Send a GET request to download the file
response = requests.get(url, stream=True)

# Check if the request was successful
if response.status_code == 200:
    # Process the gzipped JSON file
    import gzip
    from io import BytesIO

    # Decompress the gzipped content
    with gzip.GzipFile(fileobj=BytesIO(response.content)) as gz_file:
        # Read and parse the JSON data
        for line in gz_file:
            event = json.loads(line)
            print(event)  # Print each GitHub event
else:
    print(f"Failed to fetch data. HTTP Status Code: {response.status_code}")
