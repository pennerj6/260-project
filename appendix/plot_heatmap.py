import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# os.makedirs("plot", exist_ok=True)

df = pd.read_csv('../data/productivityData.csv', parse_dates=['toxic_date'])
days = [-3, -2, -1, 0, 1, 2, 3]
metrics = ['commits', 'contributors', 'lines_of_code']


data_matrix = np.zeros((len(metrics), len(days)))
for i, metric in enumerate(metrics):
    values = [df[f"day{day}_{metric}"].mean() for day in days]
    data_matrix[i, :] = values

fig, ax = plt.subplots(figsize=(10, 4))
cax = ax.imshow(data_matrix, cmap='viridis', aspect='auto')
ax.set_xticks(np.arange(len(days)))
ax.set_xticklabels([str(day) for day in days])
ax.set_yticks(np.arange(len(metrics)))
ax.set_yticklabels(metrics)
ax.set_xlabel('Days Relative to Toxic Date')
ax.set_title('Heatmap of Average Productivity Metrics')
fig.colorbar(cax, ax=ax)
plt.savefig('plot_heatmap.png')
plt.show()
