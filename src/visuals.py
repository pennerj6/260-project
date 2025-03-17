import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from datetime import datetime, timedelta

from scipy.stats import ttest_rel
from scipy.stats import pearsonr, spearmanr

from helper import *
'''
    used LLM to help configure this file give what i prompted, like stuff along the lines of 
    given these column names, give me code to calculate if any of them are correlated with toxicity using spearman correlation ( toxicity corellates to comments,commits, etc.)
    after i noticed almost nothing had a strong correlation ( alot of close to 0 values for toxicity despite the lasrge dataset)
    llm helped implement using a percentile threshold instead of a hard toxicity threshold to determine if toxic
'''

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


# Analysis functions
def analyze_toxicity_distribution(comments_data):
    toxicity_values = [float(comment['toxicity']) for comment in comments_data]
    
    # Gicen the toxicity , vaules, get the percentils
    percentiles = np.percentile(toxicity_values, [50, 75, 90, 95, 99])
    
    # plotinf the data as 2 histograms (gpt helpled with all plotting)
    if 1 == 1:
        plt.figure(figsize=(12, 5))
        
        # Left: All values
        plt.subplot(1, 2, 1)
        plt.hist(toxicity_values, bins=30, color='blue', alpha=0.7)
        plt.title('All Toxicity Values')
        plt.xlabel('Toxicity Score')
        plt.ylabel('Number of Comments')
        
        # Right: Non-zero values only
        plt.subplot(1, 2, 2)
        # non_zero_values = [t for t in toxicity_values if t > 0.001]
        # going to force it to be 50 or > (to see better results since data is skewed towards 0. TA said that is fine)
        non_zero_values = [t for t in toxicity_values if t >= 0.5]
        if non_zero_values:
            plt.hist(non_zero_values, bins=30, color='red', alpha=0.7)
            plt.title('Toxicity Scores >= 0.50')
            plt.xlabel('Toxicity Score')
        else:
            plt.text(0.5, 0.5, 'No values > 0.001', ha='center')
            plt.title('Non-zero Values (None Found)')
        
        plt.tight_layout()
        plt.savefig('toxicity_distribution.png')
    
    return percentiles

def mark_toxic_comments(comments_data, threshold_percentile=90):
    # For our results all of our data was skewed towards 0 toxicity even after normalizing data, after consulting gpt it rec using a threshold percentile (is it toxic based on the other comments rather than a hard concrete threshold number)
    toxicity_values = [float(comment['toxicity']) for comment in comments_data]
    threshold = np.percentile(toxicity_values, threshold_percentile)
    # print(f"Threshold value: {threshold:.4f}")
    
    # Mark comments as toxic or not
    for comment in comments_data:
        comment['is_toxic'] = 1 if float(comment['toxicity']) >= threshold else 0
    
    # Count toxic comments by repository
    toxic_by_repo = defaultdict(int)
    total_by_repo = defaultdict(int)
    
    for comment in comments_data:
        repo = comment['repo']
        total_by_repo[repo] += 1
        toxic_by_repo[repo] += comment['is_toxic']
    
    # Find repositories with highest toxic percentage
    repos_with_enough_comments = []
    
    for repo in total_by_repo:
        if total_by_repo[repo] >= 10:  # At least 10 comments
            toxic_percent = (toxic_by_repo[repo] / total_by_repo[repo]) * 100
            repos_with_enough_comments.append({
                'repo': repo,
                'toxic_count': toxic_by_repo[repo],
                'total_comments': total_by_repo[repo],
                'toxic_percent': toxic_percent
            })
    
    # Sort by percentage
    repos_with_enough_comments.sort(key=lambda x: x['toxic_percent'], reverse=True)
        
    return comments_data, threshold

def rq1(comments_data, issues_data, commits_data):
    
    toxic_by_week = defaultdict(int)
    total_by_week = defaultdict(int)
    
    # Process comments to get toxicity by week
    for comment in comments_data:
        date = parse_date(comment['created_at'])
        week = get_week_key(date)
        if week:
            total_by_week[week] += 1
            toxic_by_week[week] += comment['is_toxic']
    
    # Toxicity percentage by week
    weekly_toxicity = {}
    for week in total_by_week:
        if total_by_week[week] >= 5:  # At least 5 comments
            weekly_toxicity[week] = (toxic_by_week[week] / total_by_week[week]) * 100
    
    # Count commits by week
    commits_by_week = defaultdict(int)
    for commit in commits_data:
        date = parse_date(commit['date'])
        week = get_week_key(date)
        if week:
            commits_by_week[week] += 1
    
    # Count CREATED issues by week (instead of closed)
    issues_created_by_week = defaultdict(int)
    for issue in issues_data:
        # Use created_at instead of closed_at
        if issue.get('created_at'):
            date = parse_date(issue['created_at'])
            week = get_week_key(date)
            if week:
                issues_created_by_week[week] += 1
    
    # Create data for analysis
    analysis_data = []
    for week in weekly_toxicity:
        if week in commits_by_week or week in issues_created_by_week:
            analysis_data.append({
                'week': week,
                'toxicity': weekly_toxicity[week],
                'commits': commits_by_week.get(week, 0),
                'issues_created': issues_created_by_week.get(week, 0),  # New metric
                'comments': total_by_week[week]
            })
    
    # Create a dataframe
    df = pd.DataFrame(analysis_data)
    
    if len(df) < 10:
        return None
    
    # Store correlation results
    correlation_results = {}
    
    # Calculate both Spearman and Pearson correlations and store results
    for metric in ['commits', 'issues_created']:  # Changed from 'issues_closed'
        # Check if there's variation in the metric data
        if df[metric].std() == 0:
            # No variation in the metric, correlation is undefined
            correlation_results[metric] = {
                'spearman': {
                    'correlation': 0,  # Use 0 instead of NaN
                    'p_value': 1.0,    # No significance
                    'significant': False,
                    'no_variation': True  # Flag to indicate no variation
                },
                'pearson': {
                    'correlation': 0,  # Use 0 instead of NaN
                    'p_value': 1.0,    # No significance
                    'significant': False,
                    'no_variation': True  # Flag to indicate no variation
                }
            }
        else:
            # Normal case - calculate correlations
            try:
                # Spearman correlation
                spearman_corr, spearman_p_value = spearmanr(df['toxicity'], df[metric])
                
                # Pearson correlation
                pearson_corr, pearson_p_value = pearsonr(df['toxicity'], df[metric])
                
                correlation_results[metric] = {
                    'spearman': {
                        'correlation': spearman_corr,
                        'p_value': spearman_p_value,
                        'significant': spearman_p_value < 0.05,
                        'no_variation': False
                    },
                    'pearson': {
                        'correlation': pearson_corr,
                        'p_value': pearson_p_value,
                        'significant': pearson_p_value < 0.05,
                        'no_variation': False
                    }
                }
            except:
                # Handle any other errors in correlation calculation
                correlation_results[metric] = {
                    'spearman': {
                        'correlation': 0, 
                        'p_value': 1.0,
                        'significant': False,
                        'calculation_error': True
                    },
                    'pearson': {
                        'correlation': 0,
                        'p_value': 1.0,
                        'significant': False,
                        'calculation_error': True
                    }
                }
                    
    # Create PLOT 
    if 1 == 1:
        plt.figure(figsize=(12, 8))
        
        # Time series plot
        plt.subplot(2, 1, 1)
        weeks_ordered = sorted(df['week'])
        toxicity_values = [df[df['week'] == week]['toxicity'].values[0] for week in weeks_ordered]
        
        plt.plot(range(len(weeks_ordered)), toxicity_values, 'r-o', linewidth=2)
        plt.ylabel('Toxicity %')
        plt.title('Toxicity Percentage Over Time')
        
        # Show some week labels (not all to avoid overcrowding)
        label_positions = range(0, len(weeks_ordered), max(1, len(weeks_ordered) // 8))
        plt.xticks(label_positions, [weeks_ordered[i] for i in label_positions], rotation=45)
        
        # Scatter plots with correlation info in titles
        plt.subplot(2, 2, 3)
        plt.scatter(df['toxicity'], df['commits'])
        plt.xlabel('Toxicity %')
        plt.ylabel('Commits')
        
        # Add correlation information to title for commits
        if correlation_results['commits'].get('spearman', {}).get('no_variation', False):
            plt.title('Toxicity vs. Commits\nNo correlation (no variation in commits)')
        else:
            spearman_corr = correlation_results['commits']['spearman']['correlation']
            spearman_p_val = correlation_results['commits']['spearman']['p_value']
            spearman_sig_symbol = "**" if spearman_p_val < 0.01 else ("*" if spearman_p_val < 0.05 else "")
            
            pearson_corr = correlation_results['commits']['pearson']['correlation']
            pearson_p_val = correlation_results['commits']['pearson']['p_value']
            pearson_sig_symbol = "**" if pearson_p_val < 0.01 else ("*" if pearson_p_val < 0.05 else "")
            
            plt.title(f'Toxicity vs. Commits\nSpearman rho = {spearman_corr:.3f} (p = {spearman_p_val:.3f}) {spearman_sig_symbol}\n'
                      f'Pearson r = {pearson_corr:.3f} (p = {pearson_p_val:.3f}) {pearson_sig_symbol}')

        # Add trend line
        if len(df) > 1 and df['commits'].std() > 0:
            z = np.polyfit(df['toxicity'], df['commits'], 1)
            trend = np.poly1d(z)
            plt.plot(df['toxicity'], trend(df['toxicity']), "r--")
        
        # Issues Created plot (instead of issues closed)
        plt.subplot(2, 2, 4)
        plt.scatter(df['toxicity'], df['issues_created'], color='green')  # Changed to issues_created
        plt.xlabel('Toxicity %')
        plt.ylabel('Issues Created')  # Updated label
        
        # Add correlation information to title for issues created
        if correlation_results['issues_created'].get('spearman', {}).get('no_variation', False):
            plt.title('Toxicity vs. Issues Created\nNo correlation (no variation in issues created)')
        else:
            spearman_corr = correlation_results['issues_created']['spearman']['correlation']
            spearman_p_val = correlation_results['issues_created']['spearman']['p_value']
            spearman_sig_symbol = "**" if spearman_p_val < 0.01 else ("*" if spearman_p_val < 0.05 else "")
            
            pearson_corr = correlation_results['issues_created']['pearson']['correlation']
            pearson_p_val = correlation_results['issues_created']['pearson']['p_value']
            pearson_sig_symbol = "**" if pearson_p_val < 0.01 else ("*" if pearson_p_val < 0.05 else "")
            
            plt.title(f'Toxicity vs. Issues Created\nSpearman rho = {spearman_corr:.3f} (p = {spearman_p_val:.3f}) {spearman_sig_symbol}\n'
                      f'Pearson r = {pearson_corr:.3f} (p = {pearson_p_val:.3f}) {pearson_sig_symbol}')
        
        # Add trend line only if there's variation in issues_created
        if len(df) > 1 and df['issues_created'].std() > 0:
            z = np.polyfit(df['toxicity'], df['issues_created'], 1)
            trend = np.poly1d(z)
            plt.plot(df['toxicity'], trend(df['toxicity']), "r--")
        
        plt.tight_layout()
        plt.savefig('rq1.png')

        # Return results including correlation data
        return {
            'data': df,
            'correlations': correlation_results
        }

def rq2(comments_data, releases_data):    
    comments_by_date = defaultdict(list)
    for comment in comments_data:
        date = parse_date(comment['created_at'])
        if date:
            comments_by_date[date.date()].append(comment['is_toxic'])
    
    daily_toxicity = {}
    for date, toxic_list in comments_by_date.items():
        if len(toxic_list) >= 3:  
            daily_toxicity[date] = (sum(toxic_list) / len(toxic_list)) * 100
    
    # get release dates 
    release_dates = []
    for release in releases_data:
        if release.get('published_at'):
            date = parse_date(release['published_at'])
            if date:
                release_dates.append(date.date())
    
    
    release_dates = sorted(release_dates)
    
    # Check 7 days before/after release
    days_to_check = 7
    
    # collect data for before and after release to anylze ( my guess is toxicity is higher before a release and cools down after.... that is if we had ALOT more data of toxic repos)
    release_comparisons = []
    
    for release_date in release_dates:
        # Before release values
        before_values = []
        for i in range(1, days_to_check + 1):
            check_date = release_date - timedelta(days=i)
            if check_date in daily_toxicity:
                before_values.append(daily_toxicity[check_date])
        
        # After release values
        after_values = []
        for i in range(1, days_to_check + 1):
            check_date = release_date + timedelta(days=i)
            if check_date in daily_toxicity:
                after_values.append(daily_toxicity[check_date])
        
        # Only include releases with both before and after data
        if before_values and after_values:
            before_avg = sum(before_values) / len(before_values)
            after_avg = sum(after_values) / len(after_values)
            
            # Calculate percentage change
            if before_avg > 0:
                pct_change = ((after_avg - before_avg) / before_avg) * 100
            else:
                pct_change = 0 if after_avg == 0 else 100  # Handling division by zero
            
            release_comparisons.append({
                'release_date': release_date,
                'before_avg': before_avg,
                'after_avg': after_avg,
                'pct_change': pct_change
            })
    
    # Overall before vs after statistics
    if release_comparisons:
        avg_before = sum(r['before_avg'] for r in release_comparisons) / len(release_comparisons)
        avg_after = sum(r['after_avg'] for r in release_comparisons) / len(release_comparisons)
        
        increases = sum(1 for r in release_comparisons if r['after_avg'] > r['before_avg'])
        decreases = sum(1 for r in release_comparisons if r['after_avg'] < r['before_avg'])
        no_change = sum(1 for r in release_comparisons if r['after_avg'] == r['before_avg'])
        
        # Perform t-test on before/after pairs
        if len(release_comparisons) > 1:
            before_vals = [r['before_avg'] for r in release_comparisons]
            after_vals = [r['after_avg'] for r in release_comparisons]
            t_stat, p_value = ttest_rel(before_vals, after_vals)
            
            before_after_results = {
                'overall_before_avg': avg_before,
                'overall_after_avg': avg_after,
                'overall_change_pct': (avg_after - avg_before) / avg_before * 100 if avg_before > 0 else 0,
                'increases': increases,
                'decreases': decreases,
                'no_change': no_change,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
    
    # The rest of your correlation analysis code remains the same
    proximity_data = []
    
    for date in sorted(daily_toxicity.keys()):
        days_to_nearest = float('inf')
        for release_date in release_dates:
            days_diff = (date - release_date).days
            if abs(days_diff) < abs(days_to_nearest):
                days_to_nearest = days_diff
        
        if abs(days_to_nearest) <= 30:
            proximity_data.append({
                'date': date,
                'days_to_release': days_to_nearest,
                'toxicity': daily_toxicity[date]
            })
    
    correlation_results = {}
    if len(proximity_data) >= 10:
        prox_df = pd.DataFrame(proximity_data)
        prox_df['distance_from_release'] = prox_df['days_to_release'].abs()
        
        # Calculate both Spearman and Pearson correlations
        spearman_days, spearman_p_days = spearmanr(prox_df['days_to_release'], prox_df['toxicity'])
        pearson_days, pearson_p_days = pearsonr(prox_df['days_to_release'], prox_df['toxicity'])
        
        spearman_distance, spearman_p_distance = spearmanr(prox_df['distance_from_release'], prox_df['toxicity'])
        pearson_distance, pearson_p_distance = pearsonr(prox_df['distance_from_release'], prox_df['toxicity'])
        
        correlation_results = {
            'days_directional': {
                'spearman': {
                    'rho': spearman_days,
                    'p_value': spearman_p_days,
                    'significant': spearman_p_days < 0.05
                },
                'pearson': {
                    'r': pearson_days,
                    'p_value': pearson_p_days,
                    'significant': pearson_p_days < 0.05
                }
            },
            'absolute_distance': {
                'spearman': {
                    'rho': spearman_distance,
                    'p_value': spearman_p_distance,
                    'significant': spearman_p_distance < 0.05
                },
                'pearson': {
                    'r': pearson_distance,
                    'p_value': pearson_p_distance,
                    'significant': pearson_p_distance < 0.05
                }
            }
        }
        
        # Create visualizations
        fig = plt.figure(figsize=(15, 12))
        
        # PLOT 1: Grouped bar chart for each release
        if release_comparisons:
            # Show only most recent N releases if there are too many
            max_releases_to_show = 10
            recent_releases = release_comparisons[-max_releases_to_show:] if len(release_comparisons) > max_releases_to_show else release_comparisons
            
            plt.subplot(2, 1, 1)
            
            # Prepare data for grouped bar chart
            release_dates_str = [r['release_date'].strftime('%Y-%m-%d') for r in recent_releases]
            before_values = [r['before_avg'] for r in recent_releases]
            after_values = [r['after_avg'] for r in recent_releases]
            
            # Position of bars
            x = np.arange(len(release_dates_str))
            width = 0.35
            
            # Create the bars
            # bars1 = plt.bar(x - width/2, before_values, width, label='Before Release', color='royalblue')
            # bars2 = plt.bar(x + width/2, after_values, width, label='After Release', color='coral')
            bars1 = plt.bar(x - width/2, before_values, width, label='Before Release', color='#4878CF')
            bars2 = plt.bar(x + width/2, after_values, width, label='After Release', color='#E34A33')
            # colors = ['#4878CF', '#E34A33']

            # Add percentage change labels
            for i, r in enumerate(recent_releases):
                pct_change = r['pct_change']
                # Format change as string with sign
                change_str = f"{pct_change:.1f}%"
                
                # Determine color of label based on change direction
                color = 'red' if pct_change > 0 else 'green'
                if abs(pct_change) < 0.1:  # Negligible change
                    color = 'black'
                
                # Position label above the higher of the two bars
                y_pos = max(before_values[i], after_values[i]) + 0.002  # Small offset
                
                # Add the label
                plt.annotate(change_str,
                             xy=(x[i], y_pos),
                             xytext=(0, 2),  # Small vertical offset
                             textcoords="offset points",
                             ha='center',
                             va='bottom',
                             color=color,
                             bbox=dict(boxstyle="round,pad=0.2", fc='white', ec=color, alpha=0.7))
            
            # Add labels and title
            plt.xlabel('Release Date')
            plt.ylabel('Average Toxicity Score')
            
            # Set y-axis to start at 0
            plt.ylim(bottom=0)
            
            # Calculate overall significance for title
            sig_text = "*" if before_after_results.get('significant', False) else "(not significant)"
            p_value = before_after_results.get('p_value', 1.0)
            
            plt.title(f'Toxicity Levels Before and After Recent Software Releases\nn={len(recent_releases)}, p={p_value:.3f} {sig_text}')
            
            plt.xticks(x, release_dates_str, rotation=45)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
        # The scatter plots remain the same
        spearman_sig_days = "**" if spearman_p_days < 0.01 else ("*" if spearman_p_days < 0.05 else "")
        pearson_sig_days = "**" if pearson_p_days < 0.01 else ("*" if pearson_p_days < 0.05 else "")
        
        plt.subplot(2, 2, 3)
        plt.scatter(prox_df['days_to_release'], prox_df['toxicity'], alpha=0.7)
        plt.axvline(x=0, color='red', linestyle='--')
        plt.xlabel('Days from Release (- = before, + = after)')
        plt.ylabel('Toxicity %')
        plt.title(f'Toxicity vs. Days from Release \n'
                 f'Spearman rho = {spearman_days:.3f} (p = {spearman_p_days:.3f}) {spearman_sig_days}\n'
                 f'Pearson r = {pearson_days:.3f} (p = {pearson_p_days:.3f}) {pearson_sig_days}')
        
        if len(prox_df) > 1:
            z = np.polyfit(prox_df['days_to_release'], prox_df['toxicity'], 1)
            trend = np.poly1d(z)
            plt.plot(prox_df['days_to_release'], trend(prox_df['days_to_release']), "r--")
        
        spearman_sig_distance = "**" if spearman_p_distance < 0.01 else ("*" if spearman_p_distance < 0.05 else "")
        pearson_sig_distance = "**" if pearson_p_distance < 0.01 else ("*" if pearson_p_distance < 0.05 else "")
        
        plt.subplot(2, 2, 4)
        plt.scatter(prox_df['distance_from_release'], prox_df['toxicity'], alpha=0.7, color='green')
        plt.xlabel('Days Away from Nearest Release')
        plt.ylabel('Toxicity %')
        plt.title(f'Correlation: Distance from Release vs. Toxicity\n'
                 f'Spearman rho = {spearman_distance:.3f} (p = {spearman_p_distance:.3f}) {spearman_sig_distance}\n'
                 f'Pearson r = {pearson_distance:.3f} (p = {pearson_p_distance:.3f}) {pearson_sig_distance}')
        
        if len(prox_df) > 1:
            z = np.polyfit(prox_df['distance_from_release'], prox_df['toxicity'], 1)
            trend = np.poly1d(z)
            plt.plot(prox_df['distance_from_release'], trend(prox_df['distance_from_release']), "r--")
        
        plt.tight_layout()
        plt.savefig('rq2.png')
        
        # Return results including correlation data
        return {
            'release_comparisons': release_comparisons,
            'overall_before_after': before_after_results if release_comparisons else None,
            'correlations': correlation_results,
            'proximity_data': prox_df.to_dict('records') if len(prox_df) > 0 else None
        }
    else:
        return None
  
def rq3(comments_data, contributors_data):
    # to remove insanely large ouliers, gpt helped implement iqr to get more too scaled visuals
    # ex of outlier-> a million followers
    # Create a lookup dictionary for quick access to contributor information
    contributor_info = {}
    for contributor in contributors_data:
        if contributor.get('user_id'):
            user_id = contributor['user_id']
            
            # Parse account creation date for calculating account age
            account_created = None
            if contributor.get('account_created_at'):
                account_created = parse_date(contributor['account_created_at'])
            
            # Safely extract numerical metrics with error handling
            try:
                contributions = int(contributor.get('contributions', 0))
            except:
                contributions = 0
                
            try:
                followers = int(contributor.get('followers', 0))
            except:
                followers = 0
            
            # Store contributor metrics for later analysis
            contributor_info[user_id] = {
                'account_age_days': (datetime.now() - account_created).days if account_created else 0,
                'contributions': contributions,
                'followers': followers
            }
    
    # Organize comments by user to calculate per-user toxicity rates
    user_comments = defaultdict(list)
    for comment in comments_data:
        if comment.get('user_id'):
            user_id = comment['user_id']
            user_comments[user_id].append(comment['is_toxic'])
    
    # Calculate toxicity percentage per user, requiring minimum comment threshold
    user_toxicity = {}
    user_comment_count = {}
    for user_id, toxic_list in user_comments.items():
        if len(toxic_list) >= 3:  # Minimum threshold for reliable percentage
            user_toxicity[user_id] = (sum(toxic_list) / len(toxic_list)) * 100
            user_comment_count[user_id] = len(toxic_list)
    
    # Create comprehensive dataset merging toxicity and contributor metrics
    user_data = []
    for user_id in user_toxicity:
        if user_id in contributor_info:
            info = contributor_info[user_id]
            user_data.append({
                'user_id': user_id,
                'toxicity': user_toxicity[user_id],
                'comments': user_comment_count[user_id],
                'account_age_days': info['account_age_days'],
                'account_age_years': info['account_age_days'] / 365,  # Convert to years for better interpretation
                'contributions': info['contributions'],
                'followers': info['followers']
            })
    
    # Create pandas DataFrame for statistical analysis
    df = pd.DataFrame(user_data)
    
    # Check for minimum data requirements
    if len(df) < 10:
        return None
    
    # Create categorical experience groups for comparison
    df['experience_level'] = pd.cut(
        df['account_age_years'], 
        bins=[0, 1, 3, float('inf')],  # Dividing users into 3 experience tiers
        labels=['New (<1 year)', 'Intermediate (1-3 years)', 'Experienced (3+ years)']
    )
    
    # TARGETED OUTLIER HANDLING
    # =========================
    
    # Define IQR-based outlier filters for contributions and followers
    def filter_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        filter_mask = (df[column] >= (Q1 - 1.5 * IQR)) & (df[column] <= (Q3 + 1.5 * IQR))
        return df[filter_mask], filter_mask
    
    # Create filtered datasets for contributions and followers
    df_contrib_filtered, _ = filter_outliers(df, 'contributions')
    df_followers_filtered, _ = filter_outliers(df, 'followers')
    
    # CORRELATION ANALYSIS
    # =============================================
    
    # Calculate correlations for original and filtered data
    correlation_results = {}
    
    # Define a function to compute both correlation types
    def compute_correlations(x, y):
        spearman_corr, spearman_p_value = spearmanr(x, y)
        pearson_corr, pearson_p_value = pearsonr(x, y)
        
        return {
            'spearman': {
                'rho': spearman_corr,
                'p_value': spearman_p_value,
                'significant': spearman_p_value < 0.05
            },
            'pearson': {
                'r': pearson_corr,
                'p_value': pearson_p_value,
                'significant': pearson_p_value < 0.05
            }
        }
    
    # Original correlations
    correlation_results['original'] = {
        'account_age_days': compute_correlations(df['account_age_days'], df['toxicity']),
        'contributions': compute_correlations(df['contributions'], df['toxicity']),
        'followers': compute_correlations(df['followers'], df['toxicity'])
    }
    
    # Filtered correlations
    correlation_results['filtered'] = {
        'contributions': compute_correlations(df_contrib_filtered['contributions'], df_contrib_filtered['toxicity']),
        'followers': compute_correlations(df_followers_filtered['followers'], df_followers_filtered['toxicity'])
    }
    
    # VISUALIZATIONS
    # =============
    
    # Create multi-panel figure for toxicity vs. contributions
    plt.figure(figsize=(12, 5))
    
    # 1. Original data: contributions vs toxicity
    plt.subplot(1, 2, 1)
    plt.scatter(df['contributions'], df['toxicity'], alpha=0.7, color='green')
    plt.xlabel('Contributions')
    plt.ylabel('Toxicity %')
    
    # Add correlation information to title
    spearman_corr = correlation_results['original']['contributions']['spearman']['rho']
    spearman_p_val = correlation_results['original']['contributions']['spearman']['p_value']
    spearman_sig_symbol = "**" if spearman_p_val < 0.01 else ("*" if spearman_p_val < 0.05 else "")
    
    pearson_corr = correlation_results['original']['contributions']['pearson']['r']
    pearson_p_val = correlation_results['original']['contributions']['pearson']['p_value']
    pearson_sig_symbol = "**" if pearson_p_val < 0.01 else ("*" if pearson_p_val < 0.05 else "")
    
    plt.title(f'Original: Toxicity vs. Contributions\nSpearman rho = {spearman_corr:.3f} (p = {spearman_p_val:.3f}) {spearman_sig_symbol}\n'
              f'Pearson r = {pearson_corr:.3f} (p = {pearson_p_val:.3f}) {pearson_sig_symbol}')
    
    # Add trend line
    if len(df) > 1:
        z = np.polyfit(df['contributions'], df['toxicity'], 1)
        trend = np.poly1d(z)
        plt.plot(df['contributions'], trend(df['contributions']), "r--")
    
    # 2. IQR-filtered: contributions vs toxicity
    plt.subplot(1, 2, 2)
    plt.scatter(df_contrib_filtered['contributions'], df_contrib_filtered['toxicity'], alpha=0.7, color='green')
    plt.xlabel('Contributions')
    plt.ylabel('Toxicity %')
    
    # Add correlation information to title
    spearman_corr = correlation_results['filtered']['contributions']['spearman']['rho']
    spearman_p_val = correlation_results['filtered']['contributions']['spearman']['p_value']
    spearman_sig_symbol = "**" if spearman_p_val < 0.01 else ("*" if spearman_p_val < 0.05 else "")
    
    pearson_corr = correlation_results['filtered']['contributions']['pearson']['r']
    pearson_p_val = correlation_results['filtered']['contributions']['pearson']['p_value']
    pearson_sig_symbol = "**" if pearson_p_val < 0.01 else ("*" if pearson_p_val < 0.05 else "")
    
    plt.title(f'Toxicity vs. Contributions\nSpearman rho = {spearman_corr:.3f} (p = {spearman_p_val:.3f}) {spearman_sig_symbol}\n'
              f'Pearson r = {pearson_corr:.3f} (p = {pearson_p_val:.3f}) {pearson_sig_symbol}\n')
    
    # Add trend line
    if len(df_contrib_filtered) > 1:
        z = np.polyfit(df_contrib_filtered['contributions'], df_contrib_filtered['toxicity'], 1)
        trend = np.poly1d(z)
        plt.plot(df_contrib_filtered['contributions'], trend(df_contrib_filtered['contributions']), "r--")
    
    plt.tight_layout()
    plt.savefig('rq3_contributions_outlier_handling.png')
    
    # Create multi-panel figure for toxicity vs. followers
    plt.figure(figsize=(12, 5))
    
    # 1. Original data: followers vs toxicity
    plt.subplot(1, 2, 1)
    plt.scatter(df['followers'], df['toxicity'], alpha=0.7, color='purple')
    plt.xlabel('Followers')
    plt.ylabel('Toxicity %')
    
    # Add correlation information to title
    spearman_corr = correlation_results['original']['followers']['spearman']['rho']
    spearman_p_val = correlation_results['original']['followers']['spearman']['p_value']
    spearman_sig_symbol = "**" if spearman_p_val < 0.01 else ("*" if spearman_p_val < 0.05 else "")
    
    pearson_corr = correlation_results['original']['followers']['pearson']['r']
    pearson_p_val = correlation_results['original']['followers']['pearson']['p_value']
    pearson_sig_symbol = "**" if pearson_p_val < 0.01 else ("*" if pearson_p_val < 0.05 else "")
    
    plt.title(f'Original: Toxicity vs. Followers\nSpearman rho = {spearman_corr:.3f} (p = {spearman_p_val:.3f}) {spearman_sig_symbol}\n'
              f'Pearson r = {pearson_corr:.3f} (p = {pearson_p_val:.3f}) {pearson_sig_symbol}')
    
    # Add trend line
    if len(df) > 1:
        z = np.polyfit(df['followers'], df['toxicity'], 1)
        trend = np.poly1d(z)
        plt.plot(df['followers'], trend(df['followers']), "r--")
    
    # 2. IQR-filtered: followers vs toxicity
    plt.subplot(1, 2, 2)
    plt.scatter(df_followers_filtered['followers'], df_followers_filtered['toxicity'], alpha=0.7, color='purple')
    plt.xlabel('Followers')
    plt.ylabel('Toxicity %')
    
    # Add correlation information to title
    spearman_corr = correlation_results['filtered']['followers']['spearman']['rho']
    spearman_p_val = correlation_results['filtered']['followers']['spearman']['p_value']
    spearman_sig_symbol = "**" if spearman_p_val < 0.01 else ("*" if spearman_p_val < 0.05 else "")
    
    pearson_corr = correlation_results['filtered']['followers']['pearson']['r']
    pearson_p_val = correlation_results['filtered']['followers']['pearson']['p_value']
    pearson_sig_symbol = "**" if pearson_p_val < 0.01 else ("*" if pearson_p_val < 0.05 else "")
    
    plt.title(f'Toxicity vs. Followers\nSpearman rho = {spearman_corr:.3f} (p = {spearman_p_val:.3f}) {spearman_sig_symbol}\n'
              f'Pearson r = {pearson_corr:.3f} (p = {pearson_p_val:.3f}) {pearson_sig_symbol}\n')
    
    # Add trend line
    if len(df_followers_filtered) > 1:
        z = np.polyfit(df_followers_filtered['followers'], df_followers_filtered['toxicity'], 1)
        trend = np.poly1d(z)
        plt.plot(df_followers_filtered['followers'], trend(df_followers_filtered['followers']), "r--")
    
    plt.tight_layout()
    plt.savefig('rq3_followers_outlier_handling.png')
    
    # Create a figure for the original correlations as in the original function
    plt.figure(figsize=(12, 10))
    
    # Scatter plot: account age vs toxicity
    plt.subplot(2, 2, 1)
    plt.scatter(df['account_age_years'], df['toxicity'], alpha=0.7)
    plt.xlabel('Account Age (Years)')
    plt.ylabel('Toxicity %')
    
    # Add correlation information to title
    spearman_corr = correlation_results['original']['account_age_days']['spearman']['rho']
    spearman_p_val = correlation_results['original']['account_age_days']['spearman']['p_value']
    spearman_sig_symbol = "**" if spearman_p_val < 0.01 else ("*" if spearman_p_val < 0.05 else "")
    
    pearson_corr = correlation_results['original']['account_age_days']['pearson']['r']
    pearson_p_val = correlation_results['original']['account_age_days']['pearson']['p_value']
    pearson_sig_symbol = "**" if pearson_p_val < 0.01 else ("*" if pearson_p_val < 0.05 else "")
    
    plt.title(f'Toxicity vs. Account Age\nSpearman rho = {spearman_corr:.3f} (p = {spearman_p_val:.3f}) {spearman_sig_symbol}\n'
              f'Pearson r = {pearson_corr:.3f} (p = {pearson_p_val:.3f}) {pearson_sig_symbol}')
    
    # Add trend line to visualize correlation direction
    if len(df) > 1:
        z = np.polyfit(df['account_age_years'], df['toxicity'], 1)
        trend = np.poly1d(z)
        plt.plot(df['account_age_years'], trend(df['account_age_years']), "r--")
    
    # Scatter plot: contributions vs toxicity
    plt.subplot(2, 2, 2)
    plt.scatter(df['contributions'], df['toxicity'], alpha=0.7, color='green')
    plt.xlabel('Contributions')
    plt.ylabel('Toxicity %')
    
    # Add correlation information to title
    spearman_corr = correlation_results['original']['contributions']['spearman']['rho']
    spearman_p_val = correlation_results['original']['contributions']['spearman']['p_value']
    spearman_sig_symbol = "**" if spearman_p_val < 0.01 else ("*" if spearman_p_val < 0.05 else "")
    
    pearson_corr = correlation_results['original']['contributions']['pearson']['r']
    pearson_p_val = correlation_results['original']['contributions']['pearson']['p_value']
    pearson_sig_symbol = "**" if pearson_p_val < 0.01 else ("*" if pearson_p_val < 0.05 else "")
    
    plt.title(f'Toxicity vs. Contributions\nSpearman rho = {spearman_corr:.3f} (p = {spearman_p_val:.3f}) {spearman_sig_symbol}\n'
              f'Pearson r = {pearson_corr:.3f} (p = {pearson_p_val:.3f}) {pearson_sig_symbol}')
    
    # Add trend line to visualize correlation direction
    if len(df) > 1:
        z = np.polyfit(df['contributions'], df['toxicity'], 1)
        trend = np.poly1d(z)
        plt.plot(df['contributions'], trend(df['contributions']), "r--")
    
    # Scatter plot: followers vs toxicity
    plt.subplot(2, 2, 3)
    plt.scatter(df['followers'], df['toxicity'], alpha=0.7, color='purple')
    plt.xlabel('Followers')
    plt.ylabel('Toxicity %')
    
    # Add correlation information to title
    spearman_corr = correlation_results['original']['followers']['spearman']['rho']
    spearman_p_val = correlation_results['original']['followers']['spearman']['p_value']
    spearman_sig_symbol = "**" if spearman_p_val < 0.01 else ("*" if spearman_p_val < 0.05 else "")
    
    pearson_corr = correlation_results['original']['followers']['pearson']['r']
    pearson_p_val = correlation_results['original']['followers']['pearson']['p_value']
    pearson_sig_symbol = "**" if pearson_p_val < 0.01 else ("*" if pearson_p_val < 0.05 else "")
    
    plt.title(f'Toxicity vs. Followers \nSpearman rho = {spearman_corr:.3f} (p = {spearman_p_val:.3f}) {spearman_sig_symbol}\n'
              f'Pearson r = {pearson_corr:.3f} (p = {pearson_p_val:.3f}) {pearson_sig_symbol}')
    
    # Add trend line to visualize correlation direction
    if len(df) > 1:
        z = np.polyfit(df['followers'], df['toxicity'], 1)
        trend = np.poly1d(z)
        plt.plot(df['followers'], trend(df['followers']), "r--")
    
    # Box plot to show distribution of toxicity by experience level
    plt.subplot(2, 2, 4)
    sns.boxplot(x='experience_level', y='toxicity', data=df)
    plt.xlabel('Experience Level')
    plt.ylabel('Toxicity %')
    plt.title('Toxicity by Experience Level')
    
    # Analyze toxicity by experience group
    exp_stats = df.groupby('experience_level')['toxicity'].agg(['mean', 'median', 'count']).reset_index()
    
    # Add sample size annotations for context
    for i, exp in enumerate(exp_stats['experience_level']):
        count = exp_stats[exp_stats['experience_level'] == exp]['count'].values[0]
        plt.text(i, df[df['experience_level'] == exp]['toxicity'].max() + 2, 
                f"n={int(count)}", ha='center')
    
    plt.tight_layout()
    plt.savefig('rq3.png')
    
    # Compile and return results
    return {
        'original_correlations': correlation_results['original'],
        'filtered_correlations': correlation_results['filtered'],
        'experience_groups': exp_stats.to_dict('records'),
        'outlier_stats': {
            'total_contributors': len(df),
            'contribution_outliers': len(df) - len(df_contrib_filtered),
            'follower_outliers': len(df) - len(df_followers_filtered)
        }
    }

def main():
    # Get the data AFTER running main.py (
    # ONLY if you changed main.py, otherwise i alr loaded the data from our chosen repos into these datasets
    comments_data = load_csv('data/total_comments.csv')
    issues_data = load_csv('data/total_issues.csv')
    commits_data = load_csv('data/total_commits.csv')
    contributors_data = load_csv('data/total_contributors.csv')
    releases_data = load_csv('data/total_releases.csv')
    
    # Step 1: Analyze toxicity distribution
    percentiles = analyze_toxicity_distribution(comments_data)
    
    # Step 2: Mark toxic comments (using 90th percentile)
    comments_binary, threshold = mark_toxic_comments(comments_data, threshold_percentile=90)
    
    # Step 3: Answer research questions
    print("\nAnalyzing research questions...")
    
    # RQ1: Toxicity vs Productivity
    rq1_results = rq1(comments_binary, issues_data, commits_data)
    
    # RQ2: Toxicity around Releases
    rq2_results = rq2(comments_binary, releases_data)
    
    # RQ3: Contributor Experience vs Toxicity
    rq3_results = rq3(comments_binary, contributors_data)
    
    print("\nAnalysis complete! Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main()