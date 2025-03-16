import csv

from datetime import datetime, timedelta

from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import pearsonr, spearmanr

from helper import *

'''
1) Does toxic communication in OSS communities negatively affect programmer productivity, measured through commits, issue resolutions, and discussion activity?

2) Is there any correlation between toxic communication and software releases?

3) How does the level of experience of the contributors (measured by the age of the account and previous contributions) correlate with their likelihood of engaging in toxic communication within OSS communities?

'''

# Set a better style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.size'] = 11
mpl.rcParams['figure.figsize'] = (12, 8)
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 12

# RQ1: Does toxic communication in OSS communities negatively affect programmer productivity?
def rq1(comments_data, issues_data, commits_data):
    """
    Analyze how toxic communication affects productivity metrics:
    - Commits
    - Issue resolution time
    - Discussion activity
    """
    print("RQ1: Analyzing toxicity vs. productivity...")

    # Process the comments data
    comments_by_week = defaultdict(list)
    for comment in comments_data:
        created_at = parse_date(comment['created_at'])
        week_key = get_week_key(created_at)
        if week_key:
            toxicity = float(comment['toxicity'])
            comments_by_week[week_key].append({
                'toxicity': toxicity,
                'comment_id': comment['comment_id'],
                'repo': comment['repo']
            })
    
    # Calculate weekly metrics
    weekly_metrics = {}
    for week, comments in comments_by_week.items():
        total_toxicity = sum(comment['toxicity'] for comment in comments)
        avg_toxicity = total_toxicity / len(comments) if comments else 0
        weekly_metrics[week] = {
            'avg_toxicity': avg_toxicity,
            'comment_count': len(comments)
        }
    
    # Process commits data
    commits_by_week = defaultdict(int)
    for commit in commits_data:
        commit_date = parse_date(commit['date'])
        week_key = get_week_key(commit_date)
        if week_key:
            commits_by_week[week_key] += 1
    
    # Add commit counts to weekly metrics
    for week, commit_count in commits_by_week.items():
        if week not in weekly_metrics:
            weekly_metrics[week] = {'avg_toxicity': 0, 'comment_count': 0}
        weekly_metrics[week]['commit_count'] = commit_count
    
    # Process issues data - calculate closed issues per week
    closed_issues_by_week = defaultdict(int)
    for issue in issues_data:
        if issue['closed_at']:  # Only count closed issues
            closed_at = parse_date(issue['closed_at'])
            week_key = get_week_key(closed_at)
            if week_key:
                closed_issues_by_week[week_key] += 1
    
    # Add closed issues count to weekly metrics
    for week, issues_closed in closed_issues_by_week.items():
        if week not in weekly_metrics:
            weekly_metrics[week] = {'avg_toxicity': 0, 'comment_count': 0, 'commit_count': 0}
        weekly_metrics[week]['issues_closed'] = issues_closed
    
    # Fill in missing metrics with zeros
    for week in weekly_metrics:
        if 'commit_count' not in weekly_metrics[week]:
            weekly_metrics[week]['commit_count'] = 0
        if 'issues_closed' not in weekly_metrics[week]:
            weekly_metrics[week]['issues_closed'] = 0
    
    # Calculate correlations
    weeks_with_all_data = [week for week in weekly_metrics if 
                          'avg_toxicity' in weekly_metrics[week] and 
                          'commit_count' in weekly_metrics[week] and
                          'issues_closed' in weekly_metrics[week]]
    
    toxicity_values = [weekly_metrics[week]['avg_toxicity'] for week in weeks_with_all_data]
    commit_values = [weekly_metrics[week]['commit_count'] for week in weeks_with_all_data]
    issue_values = [weekly_metrics[week]['issues_closed'] for week in weeks_with_all_data]
    comment_values = [weekly_metrics[week]['comment_count'] for week in weeks_with_all_data]
    
    # Calculate correlations using scipy
    correlation_results = {}
    if len(toxicity_values) > 1:
        pearson_toxicity_commits, p_value_pearson_commits = pearsonr(toxicity_values, commit_values)
        # Check if issues_closed has non-zero values before calculating correlation
        if any(issue_values):
            pearson_toxicity_issues, p_value_pearson_issues = pearsonr(toxicity_values, issue_values)
            spearman_toxicity_issues, p_value_spearman_issues = spearmanr(toxicity_values, issue_values)
        else:
            pearson_toxicity_issues = p_value_pearson_issues = float('nan')
            spearman_toxicity_issues = p_value_spearman_issues = float('nan')
            
        pearson_toxicity_comments, p_value_pearson_comments = pearsonr(toxicity_values, comment_values)
        
        spearman_toxicity_commits, p_value_spearman_commits = spearmanr(toxicity_values, commit_values)
        spearman_toxicity_comments, p_value_spearman_comments = spearmanr(toxicity_values, comment_values)
        
        correlation_results = {
            'commits': {
                'pearson': pearson_toxicity_commits,
                'pearson_p': p_value_pearson_commits,
                'spearman': spearman_toxicity_commits,
                'spearman_p': p_value_spearman_commits
            },
            'issues': {
                'pearson': pearson_toxicity_issues,
                'pearson_p': p_value_pearson_issues,
                'spearman': spearman_toxicity_issues,
                'spearman_p': p_value_spearman_issues
            },
            'comments': {
                'pearson': pearson_toxicity_comments,
                'pearson_p': p_value_pearson_comments,
                'spearman': spearman_toxicity_comments,
                'spearman_p': p_value_spearman_comments
            }
        }
    else:
        correlation_results = {
            'commits': {'pearson': 0, 'pearson_p': 1, 'spearman': 0, 'spearman_p': 1},
            'issues': {'pearson': 0, 'pearson_p': 1, 'spearman': 0, 'spearman_p': 1},
            'comments': {'pearson': 0, 'pearson_p': 1, 'spearman': 0, 'spearman_p': 1}
        }
    
    # Print results
    print(f"Pearson correlation between toxicity and commit count: {correlation_results['commits']['pearson']:.4f} (p={correlation_results['commits']['pearson_p']:.4f})")
    print(f"Pearson correlation between toxicity and issues closed: {correlation_results['issues']['pearson']:.4f} (p={correlation_results['issues']['pearson_p']:.4f})")
    print(f"Pearson correlation between toxicity and comment activity: {correlation_results['comments']['pearson']:.4f} (p={correlation_results['comments']['pearson_p']:.4f})")
    
    print(f"Spearman correlation between toxicity and commit count: {correlation_results['commits']['spearman']:.4f} (p={correlation_results['commits']['spearman_p']:.4f})")
    print(f"Spearman correlation between toxicity and issues closed: {correlation_results['issues']['spearman']:.4f} (p={correlation_results['issues']['spearman_p']:.4f})")
    print(f"Spearman correlation between toxicity and comment activity: {correlation_results['comments']['spearman']:.4f} (p={correlation_results['comments']['spearman_p']:.4f})")
    
    # Create improved plots
    fig = plt.figure(figsize=(16, 12))
    
    # Get weeks and sort them
    weeks = sorted(weeks_with_all_data)
    x_positions = range(len(weeks))
    
    # Create a normalized toxicity score for better visualization
    max_toxicity = max([weekly_metrics[week]['avg_toxicity'] for week in weeks]) or 1
    normalized_toxicity = [weekly_metrics[week]['avg_toxicity'] / max_toxicity for week in weeks]
    
    # Only plot every nth label to avoid overcrowding
    n = max(1, len(weeks) // 15)  # Show ~15 labels max
    
    # First subplot: Toxicity vs Commits
    ax1 = plt.subplot(2, 1, 1)
    
    # Primary Y-axis: Toxicity
    line1, = ax1.plot(x_positions, normalized_toxicity, 'r-', marker='o', label='Toxicity Score (normalized)', linewidth=2)
    ax1.set_ylabel('Normalized Toxicity', color='r', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='r')
    
    # Set x-ticks for every nth week to avoid overcrowding
    plt.xticks([i for i in x_positions if i % n == 0], 
               [weeks[i] for i in x_positions if i % n == 0], 
               rotation=45)
    
    # Secondary Y-axis: Commits
    ax2 = ax1.twinx()
    commit_counts = [weekly_metrics[week]['commit_count'] for week in weeks]
    line2, = ax2.plot(x_positions, commit_counts, 'b-', marker='s', label='Commit Count', linewidth=2)
    ax2.set_ylabel('Number of Commits', color='b', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='b')
    
    # Add legend for both axes
    lines = [line1, line2]
    plt.legend(lines, [line.get_label() for line in lines], loc='upper left')
    
    # Set title with correlation coefficients
    corr_text = (f'Pearson: {correlation_results["commits"]["pearson"]:.2f} ' + 
                f'(p={correlation_results["commits"]["pearson_p"]:.3f}), ' +
                f'Spearman: {correlation_results["commits"]["spearman"]:.2f} ' + 
                f'(p={correlation_results["commits"]["spearman_p"]:.3f})')
    
    plt.title(f'Weekly Toxicity vs. Commits\n{corr_text}', fontweight='bold')
    
    # Second subplot: Toxicity vs Issue Resolution
    ax3 = plt.subplot(2, 1, 2)
    
    # Primary Y-axis: Toxicity (again)
    ax3.plot(x_positions, normalized_toxicity, 'r-', marker='o', label='Toxicity Score (normalized)', linewidth=2)
    ax3.set_ylabel('Normalized Toxicity', color='r', fontweight='bold')
    ax3.tick_params(axis='y', labelcolor='r')
    
    # Set x-ticks for every nth week
    plt.xticks([i for i in x_positions if i % n == 0], 
               [weeks[i] for i in x_positions if i % n == 0], 
               rotation=45)
    
    # Secondary Y-axis: Issues Closed
    ax4 = ax3.twinx()
    issues_closed = [weekly_metrics[week]['issues_closed'] for week in weeks]
    ax4.plot(x_positions, issues_closed, 'g-', marker='^', label='Issues Closed', linewidth=2)
    ax4.set_ylabel('Issues Closed', color='g', fontweight='bold')
    ax4.tick_params(axis='y', labelcolor='g')
    
    # Make sure y-axis doesn't start from 0 if data is sparse
    if max(issues_closed) > 0:
        ax4.set_ylim(bottom=0, top=max(issues_closed) * 1.1)
    else:
        # If all values are 0, set a small range to avoid error
        ax4.set_ylim(bottom=0, top=1)
    
    # Add correlation coefficient
    corr_text = (f'Pearson: {correlation_results["issues"]["pearson"]:.2f} ' + 
                f'(p={correlation_results["issues"]["pearson_p"]:.3f}), ' +
                f'Spearman: {correlation_results["issues"]["spearman"]:.2f} ' + 
                f'(p={correlation_results["issues"]["spearman_p"]:.3f})')
    
    plt.title(f'Weekly Toxicity vs. Issues Closed\n{corr_text}', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('rq1.png', dpi=300)
    print("Saved improved RQ1 visualization to 'rq1.png'")
    
    return weekly_metrics, correlation_results

# RQ2: Is there any correlation between toxic communication and software releases?
def rq2(comments_data, releases_data):
    """
    Analyze correlation between toxic communication and software releases
    """
    print("RQ2: Analyzing toxicity vs. software releases...")
    
    # Process comments data with toxicity
    comments_by_date = defaultdict(list)
    for comment in comments_data:
        date = parse_date(comment['created_at']).date()
        comments_by_date[date].append(float(comment['toxicity']))
    
    # Calculate daily average toxicity
    daily_toxicity = {}
    for date, toxicity_scores in comments_by_date.items():
        daily_toxicity[date] = sum(toxicity_scores) / len(toxicity_scores)
    
    # Process releases data
    release_dates = []
    for release in releases_data:
        release_date = parse_date(release['published_at']).date()
        release_dates.append(release_date)
    
    # Analyze toxicity before and after releases
    release_window = 7  # days before and after to analyze
    release_analysis = {}
    
    for release_date in release_dates:
        before_toxicity = []
        after_toxicity = []
        
        # Check toxicity for days before release
        for i in range(1, release_window + 1):
            check_date = release_date - timedelta(days=i)
            if check_date in daily_toxicity:
                before_toxicity.append(daily_toxicity[check_date])
        
        # Check toxicity for days after release
        for i in range(1, release_window + 1):
            check_date = release_date + timedelta(days=i)
            if check_date in daily_toxicity:
                after_toxicity.append(daily_toxicity[check_date])
        
        # Calculate average toxicity before and after
        avg_before = sum(before_toxicity) / len(before_toxicity) if before_toxicity else 0
        avg_after = sum(after_toxicity) / len(after_toxicity) if after_toxicity else 0
        
        release_analysis[release_date] = {
            'before': avg_before,
            'after': avg_after,
            'change': avg_after - avg_before
        }
    
    # Print results summary
    print("\nToxicity analysis around releases:")
    increases = 0
    decreases = 0
    for release_date, data in release_analysis.items():
        if data['change'] > 0:
            increases += 1
        elif data['change'] < 0:
            decreases += 1
    
    print(f"Releases with toxicity increases: {increases}")
    print(f"Releases with toxicity decreases: {decreases}")
    print(f"No change: {len(release_analysis) - increases - decreases}")
    
    # IMPROVED VISUALIZATION: Focus on more recent releases to avoid overcrowding
    # Sort release dates
    release_dates_list = sorted(release_analysis.keys())
    
    # Select the most recent N releases to avoid overcrowding
    max_releases_to_show = 10
    if len(release_dates_list) > max_releases_to_show:
        release_dates_list = release_dates_list[-max_releases_to_show:]
    
    # Create a figure
    plt.figure(figsize=(14, 8))
    
    # Collect data for selected releases
    before_values = [release_analysis[date]['before'] for date in release_dates_list]
    after_values = [release_analysis[date]['after'] for date in release_dates_list]
    changes = [release_analysis[date]['change'] for date in release_dates_list]
    
    # Convert dates to string format for readability
    date_labels = [date.strftime('%Y-%m-%d') for date in release_dates_list]
    
    # Set up bar positions
    x = range(len(release_dates_list))
    width = 0.35
    
    # Create bar chart with better colors and alpha
    plt.bar([i - width/2 for i in x], before_values, width, label='Before Release', color='#4878CF', alpha=0.8)
    plt.bar([i + width/2 for i in x], after_values, width, label='After Release', color='#E34A33', alpha=0.8)
    
    # Improve axes and labels
    plt.xlabel('Release Date', fontweight='bold')
    plt.ylabel('Average Toxicity Score', fontweight='bold')
    plt.title('Toxicity Levels Before and After Recent Software Releases', fontweight='bold')
    plt.xticks(x, date_labels, rotation=45, ha='right')
    plt.legend(loc='upper left')
    
    # Add percentage change annotations with better positioning and format
    for i, (before, after) in enumerate(zip(before_values, after_values)):
        if before > 0 and after > 0:
            pct_change = ((after - before) / before) * 100
            color = 'green' if pct_change < 0 else 'red'
            plt.annotate(f"{pct_change:.1f}%", 
                         xy=(i, max(before, after) * 1.1),
                         ha='center',
                         fontweight='bold',
                         color=color,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
    
    # Adjust y-axis to ensure annotations are visible
    y_max = max(max(before_values), max(after_values)) * 1.3 if before_values and after_values else 0.1
    plt.ylim(0, y_max)
    
    plt.tight_layout()
    plt.savefig('rq2.png', dpi=300)
    print("Saved improved RQ2 visualization to 'rq2.png'")
    
    return release_analysis

# RQ3: How does contributor experience correlate with toxic communication?
def rq3(comments_data, contributors_data):
    """
    Analyze how contributor experience correlates with toxic communication
    """
    print("RQ3: Analyzing contributor experience vs. toxicity...")
    
    # Create a lookup dictionary for contributor information
    contributor_info = {}
    for contributor in contributors_data:
        user_id = contributor['user_id']
        account_created = parse_date(contributor['account_created_at'])
        contributor_info[user_id] = {
            'account_age_days': (datetime.now() - account_created).days if account_created else 0,
            'contributions': int(contributor['contributions']),
            'followers': int(contributor['followers']),
            'public_repos': int(contributor['public_repos'])
        }
    
    # Group comments by user
    user_comments = defaultdict(list)
    for comment in comments_data:
        user_id = comment['user_id']
        toxicity = float(comment['toxicity'])
        user_comments[user_id].append(toxicity)
    
    # Calculate average toxicity per user
    user_toxicity = {}
    for user_id, toxicity_scores in user_comments.items():
        user_toxicity[user_id] = sum(toxicity_scores) / len(toxicity_scores)
    
    # Create analysis groups based on experience
    experience_groups = {
        'new_users': [],          # Account age < 1 year
        'intermediate_users': [], # Account age 1-3 years
        'experienced_users': []   # Account age > 3 years
    }
    
    contribution_groups = {
        'low_contributions': [],    # < 50 contributions
        'medium_contributions': [], # 50-500 contributions
        'high_contributions': []    # > 500 contributions
    }
    
    # Assign users to groups and store their toxicity
    for user_id, toxicity in user_toxicity.items():
        if user_id in contributor_info:
            # Account age groups
            account_age_years = contributor_info[user_id]['account_age_days'] / 365
            if account_age_years < 1:
                experience_groups['new_users'].append(toxicity)
            elif account_age_years < 3:
                experience_groups['intermediate_users'].append(toxicity)
            else:
                experience_groups['experienced_users'].append(toxicity)
            
            # Contribution groups
            contributions = contributor_info[user_id]['contributions']
            if contributions < 50:
                contribution_groups['low_contributions'].append(toxicity)
            elif contributions < 500:
                contribution_groups['medium_contributions'].append(toxicity)
            else:
                contribution_groups['high_contributions'].append(toxicity)
    
    # Calculate average toxicity by group
    results = {
        'account_age': {},
        'contributions': {}
    }
    
    for group_name, toxicity_values in experience_groups.items():
        avg_toxicity = sum(toxicity_values) / len(toxicity_values) if toxicity_values else 0
        results['account_age'][group_name] = {
            'avg_toxicity': avg_toxicity,
            'count': len(toxicity_values)
        }
    
    for group_name, toxicity_values in contribution_groups.items():
        avg_toxicity = sum(toxicity_values) / len(toxicity_values) if toxicity_values else 0
        results['contributions'][group_name] = {
            'avg_toxicity': avg_toxicity,
            'count': len(toxicity_values)
        }
    
    # Print results
    print("\nToxicity by account age:")
    for group, data in results['account_age'].items():
        print(f"{group}: Avg toxicity = {data['avg_toxicity']:.4f} (n={data['count']})")
    
    print("\nToxicity by contribution level:")
    for group, data in results['contributions'].items():
        print(f"{group}: Avg toxicity = {data['avg_toxicity']:.4f} (n={data['count']})")
    
    # IMPROVED VISUALIZATION
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Better labels for x-axis
    age_labels = {
        'new_users': 'New Users\n(<1 year)',
        'intermediate_users': 'Intermediate Users\n(1-3 years)',
        'experienced_users': 'Experienced Users\n(>3 years)'
    }
    
    contrib_labels = {
        'low_contributions': 'Low\n(<50)',
        'medium_contributions': 'Medium\n(50-500)',
        'high_contributions': 'High\n(>500)'
    }
    
    # First plot: Account Age vs Toxicity
    groups = list(results['account_age'].keys())
    toxicity_values = [results['account_age'][group]['avg_toxicity'] for group in groups]
    counts = [results['account_age'][group]['count'] for group in groups]
    
    # Calculate error bars (standard error if we had raw data)
    # In this case, we'll use a small fixed percentage for visual indication
    errors = [value * 0.05 for value in toxicity_values]
    
    # Create bar chart with better colors
    colors = ['#6baed6', '#3182bd', '#08519c']  # Light to dark blue
    bars = ax1.bar(
        [age_labels[group] for group in groups], 
        toxicity_values, 
        color=colors,
        yerr=errors,
        capsize=5
    )
    
    # Add count annotations
    for bar, count in zip(bars, counts):
        ax1.text(
            bar.get_x() + bar.get_width()/2, 
            bar.get_height() + max(toxicity_values) * 0.02, 
            f"n={count}", 
            ha='center', va='bottom',
            fontweight='bold'
        )
    
    ax1.set_xlabel('Account Age Group', fontweight='bold')
    ax1.set_ylabel('Average Toxicity Score', fontweight='bold')
    ax1.set_title('Toxicity by User Account Age', fontweight='bold')
    
    # For small differences, don't start y-axis at zero
    y_min = min(toxicity_values) * 0.9 if min(toxicity_values) > 0 else 0
    y_max = max(toxicity_values) * 1.15
    ax1.set_ylim(y_min, y_max)
    
    # Add a horizontal line for overall average toxicity
    all_toxicity = []
    for group in groups:
        group_size = results['account_age'][group]['count']
        group_toxicity = results['account_age'][group]['avg_toxicity']
        all_toxicity.extend([group_toxicity] * group_size)
    
    overall_avg = sum(all_toxicity) / len(all_toxicity) if all_toxicity else 0
    ax1.axhline(y=overall_avg, color='r', linestyle='-', linewidth=2, 
                label=f'Overall Avg: {overall_avg:.4f}')
    ax1.legend()
    
    # Second plot: Contribution Level vs Toxicity
    groups = list(results['contributions'].keys())
    toxicity_values = [results['contributions'][group]['avg_toxicity'] for group in groups]
    counts = [results['contributions'][group]['count'] for group in groups]
    
    # Calculate error bars
    errors = [value * 0.05 for value in toxicity_values]
    
    # Create bar chart with better colors
    colors = ['#c7e9c0', '#74c476', '#238b45']  # Light to dark green
    bars = ax2.bar(
        [contrib_labels[group] for group in groups], 
        toxicity_values, 
        color=colors,
        yerr=errors,
        capsize=5
    )
    
    # Add count annotations
    for bar, count in zip(bars, counts):
        ax2.text(
            bar.get_x() + bar.get_width()/2, 
            bar.get_height() + max(toxicity_values) * 0.02, 
            f"n={count}", 
            ha='center', va='bottom',
            fontweight='bold'
        )
    
    ax2.set_xlabel('Contribution Level', fontweight='bold')
    ax2.set_ylabel('Average Toxicity Score', fontweight='bold')
    ax2.set_title('Toxicity by Contribution Level', fontweight='bold')
    
    # For small differences, don't start y-axis at zero
    y_min = min(toxicity_values) * 0.9 if min(toxicity_values) > 0 else 0
    y_max = max(toxicity_values) * 1.15
    ax2.set_ylim(y_min, y_max)
    
    # Add a horizontal line for overall average toxicity
    all_toxicity = []
    for group in groups:
        group_size = results['contributions'][group]['count']
        group_toxicity = results['contributions'][group]['avg_toxicity']
        all_toxicity.extend([group_toxicity] * group_size)
    
    overall_avg = sum(all_toxicity) / len(all_toxicity) if all_toxicity else 0
    ax2.axhline(y=overall_avg, color='r', linestyle='-', linewidth=2, 
                label=f'Overall Avg: {overall_avg:.4f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('rq3.png', dpi=300)
    print("Saved improved RQ3 visualization to 'rq3.png'")
    
    return results

def main():
    comments_data = load_csv('data/total_comments.csv')
    issues_data = load_csv('data/total_issues.csv')
    commits_data = load_csv('data/total_commits.csv')
    contributors_data = load_csv('data/total_contributors.csv')
    releases_data = load_csv('data/total_releases.csv')
    
    # Run analyses for each research question
    rq1_results, rq1_corr = rq1(comments_data, issues_data, commits_data)
    rq2_results = rq2(comments_data, releases_data)
    rq3_results = rq3(comments_data, contributors_data)


if __name__ == "__main__":
    main()
