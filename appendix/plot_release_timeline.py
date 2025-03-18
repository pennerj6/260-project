import os
import pandas as pd
import ast
import matplotlib.pyplot as plt

os.makedirs("plot", exist_ok=True)

df = pd.read_csv('../data/productivityData.csv', parse_dates=['toxic_date'])

diffs = []
for _, row in df.iterrows():
    toxic_date = row['toxic_date']
    release_dates_str = row['release_dates']
    try:
        release_dates = ast.literal_eval(release_dates_str)
    except Exception as e:
        release_dates = []
    for rd in release_dates:
        try:
            rd_date = pd.to_datetime(rd)
            diff = (rd_date - toxic_date).days
            diffs.append(diff)
        except Exception:
            continue

plt.figure(figsize=(10, 6))
if diffs:
    plt.hist(diffs, bins=range(min(diffs)-1, max(diffs)+2), edgecolor='black')
    plt.xlabel('Days Difference (Release Date - Toxic Date)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Release Date Differences Relative to Toxic Date')
else:
    plt.text(0.5, 0.5, 'No release date data available', horizontalalignment='center', verticalalignment='center')
plt.savefig('plot_release_timeline.png')
plt.show()
