import os
import pandas as pd
import matplotlib.pyplot as plt


os.makedirs("plot", exist_ok=True)

df = pd.read_csv('data/productivityData.csv', parse_dates=['toxic_date'])


days = [-3, -2, -1, 0, 1, 2, 3]


avg_commits = [df[f"day{day}_commits"].mean() for day in days]
avg_contributors = [df[f"day{day}_contributors"].mean() for day in days]
avg_loc = [df[f"day{day}_lines_of_code"].mean() for day in days]

plt.figure(figsize=(12, 8))
plt.plot(days, avg_commits, marker='o', label='Average Commits')
plt.plot(days, avg_contributors, marker='o', label='Average Contributors')
plt.plot(days, avg_loc, marker='o', label='Average Lines of Code')
plt.xlabel('Days Relative to Toxic Date')
plt.ylabel('Average Value')
plt.title('Overall Average Productivity Metrics Around Toxic Date')
plt.legend()
plt.grid(True)
plt.savefig('plot/plot_overall_average.png')
plt.show()
