import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import json
import warnings
from scipy import stats

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set the style for plots
plt.style.use('ggplot')
sns.set_palette("Set2")
sns.set_context("talk")

class GitHubToxicityVisualizer:
    """
    Class to visualize toxicity data from GitHub Archive Analysis
    and help answer research questions about toxic communication impact.
    """
    
    def __init__(self, data_dir="output"):
        """Initialize with path to the data directory"""
        self.data_dir = data_dir
        self.data_loaded = False
        self.figures_dir = os.path.join(data_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Main dataframes
        self.commits_df = None
        self.issues_df = None
        self.toxic_comments_df = None
        self.user_metrics_df = None
        self.toxicity_by_user_df = None
        self.summary = None
        
    def load_data(self):
        """Load all CSV data files"""
        try:
            # Load CSV files
            self.commits_df = pd.read_csv(os.path.join(self.data_dir, 'commit_freq.csv'))
            self.issues_df = pd.read_csv(os.path.join(self.data_dir, 'issue_resolution_times.csv'))
            self.toxic_comments_df = pd.read_csv(os.path.join(self.data_dir, 'toxic_comments.csv'))
            self.user_metrics_df = pd.read_csv(os.path.join(self.data_dir, 'user_metrics.csv'))
            self.toxicity_by_user_df = pd.read_csv(os.path.join(self.data_dir, 'toxicity_by_user.csv'))
            
            # Load summary report
            with open(os.path.join(self.data_dir, 'summary_report.json'), 'r') as f:
                self.summary = json.load(f)
                
            # Process date columns
            if 'date' in self.commits_df.columns:
                self.commits_df['date'] = pd.to_datetime(self.commits_df['date'])
            
            if 'created_at' in self.toxic_comments_df.columns:
                self.toxic_comments_df['created_at'] = pd.to_datetime(self.toxic_comments_df['created_at'])
                self.toxic_comments_df['date'] = self.toxic_comments_df['created_at'].dt.date
            
            if 'opened_at' in self.issues_df.columns and 'closed_at' in self.issues_df.columns:
                self.issues_df['opened_at'] = pd.to_datetime(self.issues_df['opened_at'])
                self.issues_df['closed_at'] = pd.to_datetime(self.issues_df['closed_at'])
            
            self.data_loaded = True
            print("Data loaded successfully")
            
            # Print sample data
            self._print_data_summary()
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            # Create empty dataframes if files don't exist
            self._create_empty_dataframes()
    
    def _create_empty_dataframes(self):
        """Create empty dataframes if data files don't exist"""
        self.commits_df = pd.DataFrame(columns=['date', 'repo', 'commit_count'])
        self.issues_df = pd.DataFrame(columns=['issue_number', 'repo', 'opened_at', 'closed_at', 'resolution_time_hours'])
        self.toxic_comments_df = pd.DataFrame(columns=['created_at', 'user', 'comment', 'toxicity_score', 'repo'])
        self.user_metrics_df = pd.DataFrame(columns=['user', 'activity_count', 'commit_count', 'comment_count'])
        self.toxicity_by_user_df = pd.DataFrame(columns=['user', 'total_comments', 'toxic_comments', 'toxicity_ratio'])
        self.summary = {'total_records': 0, 'toxic_comments': 0, 'average_toxicity': 0}
        self.data_loaded = True
    
    def _print_data_summary(self):
        """Print summary information about the loaded data"""
        print("\n===== DATA SUMMARY =====")
        print(f"Total records processed: {self.summary.get('total_records', 'N/A')}")
        print(f"Toxic comments identified: {self.summary.get('toxic_comments', 'N/A')}")
        print(f"Average toxicity score: {self.summary.get('average_toxicity', 'N/A'):.4f}")
        
        if not self.commits_df.empty:
            print(f"\nCommit data: {len(self.commits_df)} records")
            print(self.commits_df.head(3))
        
        if not self.issues_df.empty:
            print(f"\nIssue resolution data: {len(self.issues_df)} records")
            print(self.issues_df.head(3))
        
        if not self.toxic_comments_df.empty:
            print(f"\nToxic comments data: {len(self.toxic_comments_df)} records")
            if 'comment' in self.toxic_comments_df.columns:
                # Show truncated comments
                sample = self.toxic_comments_df.copy()
                if 'comment' in sample.columns:
                    sample['comment'] = sample['comment'].str.slice(0, 30) + "..."
                print(sample.head(3))
            else:
                print(self.toxic_comments_df.head(3))
        
        if not self.user_metrics_df.empty:
            print(f"\nUser metrics data: {len(self.user_metrics_df)} records")
            print(self.user_metrics_df.head(3))
        
        if not self.toxicity_by_user_df.empty:
            print(f"\nToxicity by user data: {len(self.toxicity_by_user_df)} records")
            print(self.toxicity_by_user_df.head(3))
    
    def generate_all_visualizations(self):
        """Generate all visualizations for the research questions"""
        if not self.data_loaded:
            self.load_data()
        
        print("\nGenerating visualizations for all research questions...")
        
        # Generate visualizations for RQ1: Toxicity and Productivity
        self.visualize_toxicity_vs_productivity()
        
        # Generate visualizations for RQ2: Toxicity and Releases
        self.visualize_toxicity_vs_releases()
        
        # Generate visualizations for RQ3: Experience and Toxicity
        self.visualize_experience_vs_toxicity()
        
        # Create overview dashboard
        self.create_dashboard()
        
        print(f"\nAll visualizations saved in: {self.figures_dir}")
    
    def visualize_toxicity_vs_productivity(self):
        """
        RQ1: Does toxic communication in OSS communities negatively affect programmer 
        productivity, measured through commits, issue resolutions, and discussion activity?
        """
        print("\nVisualizing RQ1: Toxicity vs. Productivity")
        
        # Check if we have enough data
        if (self.commits_df.empty or self.issues_df.empty or 
            self.toxic_comments_df.empty):
            print("Not enough data to analyze RQ1")
            return
        
        # 1. Prepare time-based aggregation of toxic comments
        try:
            if 'created_at' in self.toxic_comments_df.columns:
                # Aggregate toxicity by date
                toxic_by_date = self.toxic_comments_df.copy()
                toxic_by_date['date'] = pd.to_datetime(toxic_by_date['created_at']).dt.date
                daily_toxicity = toxic_by_date.groupby('date').size().reset_index(name='toxic_count')
                daily_toxicity['date'] = pd.to_datetime(daily_toxicity['date'])
                
                # If commits_df has date information
                if 'date' in self.commits_df.columns:
                    # Aggregate commits by date
                    daily_commits = self.commits_df.groupby('date')['commit_count'].sum().reset_index()
                    
                    # Merge toxicity and commit data
                    productivity_df = pd.merge(daily_toxicity, daily_commits, on='date', how='outer').fillna(0)
                    
                    # Sort by date
                    productivity_df = productivity_df.sort_values('date')
                    
                    # Plot Toxicity vs Commits over time
                    fig, ax1 = plt.subplots(figsize=(12, 6))
                    
                    # Plot toxic comments
                    ax1.set_xlabel('Date')
                    ax1.set_ylabel('Number of Toxic Comments', color='tab:red')
                    ax1.plot(productivity_df['date'], productivity_df['toxic_count'], color='tab:red', marker='o', linestyle='-')
                    ax1.tick_params(axis='y', labelcolor='tab:red')
                    
                    # Create second y-axis
                    ax2 = ax1.twinx()
                    ax2.set_ylabel('Number of Commits', color='tab:blue')
                    ax2.plot(productivity_df['date'], productivity_df['commit_count'], color='tab:blue', marker='x', linestyle='-')
                    ax2.tick_params(axis='y', labelcolor='tab:blue')
                    
                    fig.tight_layout()
                    plt.title('Toxic Comments vs. Commit Activity Over Time')
                    plt.savefig(os.path.join(self.figures_dir, 'rq1_toxicity_vs_commits_time.png'), dpi=300, bbox_inches='tight')
                    plt.close()
                    print("- Created: rq1_toxicity_vs_commits_time.png")
            
            # 2. Issue resolution time vs. toxicity
            if not self.issues_df.empty and 'resolution_time_hours' in self.issues_df.columns:
                # Calculate issue average resolution time by date
                self.issues_df['opened_date'] = pd.to_datetime(self.issues_df['opened_at']).dt.date
                issues_by_date = self.issues_df.groupby('opened_date')['resolution_time_hours'].mean().reset_index()
                issues_by_date['opened_date'] = pd.to_datetime(issues_by_date['opened_date'])
                
                # Merge with daily toxicity data if available
                if 'toxic_by_date' in locals():
                    resolution_vs_toxicity = pd.merge(
                        issues_by_date, 
                        daily_toxicity,
                        left_on='opened_date',
                        right_on='date',
                        how='outer'
                    ).fillna(0)
                    
                    # Create scatterplot
                    plt.figure(figsize=(10, 6))
                    sns.regplot(x='toxic_count', y='resolution_time_hours', data=resolution_vs_toxicity, 
                                scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
                    plt.title('Relationship Between Toxic Comments and Issue Resolution Time')
                    plt.xlabel('Number of Toxic Comments')
                    plt.ylabel('Average Issue Resolution Time (hours)')
                    plt.savefig(os.path.join(self.figures_dir, 'rq1_toxicity_vs_resolution_time.png'), dpi=300, bbox_inches='tight')
                    plt.close()
                    print("- Created: rq1_toxicity_vs_resolution_time.png")
                    
                    # Correlation analysis
                    corr = resolution_vs_toxicity[['toxic_count', 'resolution_time_hours']].corr().iloc[0,1]
                    print(f"  Correlation between toxic comments and resolution time: {corr:.4f}")
                    
                    # Create boxplot for issue resolution time distribution
                    if len(self.issues_df) > 10:  # Need enough data for meaningful boxplot
                        # Create bins based on toxicity levels in repositories
                        # First, get toxicity level by repo
                        if 'repo' in self.toxic_comments_df.columns and 'repo' in self.issues_df.columns:
                            repo_toxicity = self.toxic_comments_df.groupby('repo').size().reset_index(name='toxic_count')
                            # Categorize repos by toxicity level
                            toxicity_thresholds = repo_toxicity['toxic_count'].quantile([0.33, 0.66]).values
                            
                            def toxicity_category(count):
                                if count == 0:
                                    return "No Toxicity"
                                elif count <= toxicity_thresholds[0]:
                                    return "Low Toxicity"
                                elif count <= toxicity_thresholds[1]:
                                    return "Medium Toxicity"
                                else:
                                    return "High Toxicity"
                            
                            repo_toxicity['toxicity_level'] = repo_toxicity['toxic_count'].apply(toxicity_category)
                            
                            # Merge with issues data
                            issues_with_toxicity = pd.merge(
                                self.issues_df,
                                repo_toxicity[['repo', 'toxicity_level']],
                                on='repo',
                                how='left'
                            )
                            issues_with_toxicity['toxicity_level'] = issues_with_toxicity['toxicity_level'].fillna("No Toxicity")
                            
                            # Create boxplot
                            plt.figure(figsize=(12, 6))
                            sns.boxplot(x='toxicity_level', y='resolution_time_hours', data=issues_with_toxicity, 
                                        order=["No Toxicity", "Low Toxicity", "Medium Toxicity", "High Toxicity"])
                            plt.title('Issue Resolution Time by Repository Toxicity Level')
                            plt.xlabel('Repository Toxicity Level')
                            plt.ylabel('Resolution Time (hours)')
                            plt.savefig(os.path.join(self.figures_dir, 'rq1_resolution_time_by_toxicity_level.png'), dpi=300, bbox_inches='tight')
                            plt.close()
                            print("- Created: rq1_resolution_time_by_toxicity_level.png")
            
            # 3. User Productivity Analysis
            if not self.user_metrics_df.empty and not self.toxicity_by_user_df.empty:
                # Merge user metrics with toxicity data
                user_productivity = pd.merge(
                    self.user_metrics_df,
                    self.toxicity_by_user_df,
                    left_on='user',
                    right_on='user',
                    how='left'
                )
                user_productivity['toxicity_ratio'] = user_productivity['toxicity_ratio'].fillna(0)
                
                # Create scatterplot for commit count vs toxicity ratio
                if 'commit_count' in user_productivity.columns:
                    plt.figure(figsize=(10, 6))
                    sns.regplot(x='toxicity_ratio', y='commit_count', data=user_productivity, 
                                scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
                    plt.title('Developer Toxicity Ratio vs. Commit Count')
                    plt.xlabel('Toxicity Ratio (toxic comments / total comments)')
                    plt.ylabel('Number of Commits')
                    plt.savefig(os.path.join(self.figures_dir, 'rq1_toxicity_ratio_vs_commits.png'), dpi=300, bbox_inches='tight')
                    plt.close()
                    print("- Created: rq1_toxicity_ratio_vs_commits.png")
                    
                    corr = user_productivity[['toxicity_ratio', 'commit_count']].corr().iloc[0,1]
                    print(f"  Correlation between user toxicity ratio and commit count: {corr:.4f}")
                
                # Create scatterplot for overall activity vs toxicity ratio
                if 'activity_count' in user_productivity.columns:
                    plt.figure(figsize=(10, 6))
                    sns.regplot(x='toxicity_ratio', y='activity_count', data=user_productivity, 
                                scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
                    plt.title('Developer Toxicity Ratio vs. Overall Activity')
                    plt.xlabel('Toxicity Ratio (toxic comments / total comments)')
                    plt.ylabel('Overall Activity Count')
                    plt.savefig(os.path.join(self.figures_dir, 'rq1_toxicity_ratio_vs_activity.png'), dpi=300, bbox_inches='tight')
                    plt.close()
                    print("- Created: rq1_toxicity_ratio_vs_activity.png")
                    
        except Exception as e:
            print(f"Error in toxicity vs. productivity visualization: {str(e)}")
    
    def visualize_toxicity_vs_releases(self):
        """
        RQ2: Is there any correlation between toxic communication and software releases?
        """
        print("\nVisualizing RQ2: Toxicity vs. Releases")
        
        # Check if we have release data - our dataset should have ReleaseEvent type
        try:
            # Check if there's a 'type' column in the raw data that might contain 'ReleaseEvent'
            # First, try to load the parquet file
            try:
                raw_data_path = os.path.join(self.data_dir, 'github_data.parquet')
                if os.path.exists(raw_data_path):
                    # Try to read just the necessary columns to save memory
                    import pyarrow.parquet as pq
                    table = pq.read_table(raw_data_path, columns=['type', 'created_at'])
                    raw_df = table.to_pandas()
                    
                    if 'type' in raw_df.columns and 'ReleaseEvent' in raw_df['type'].unique():
                        releases_df = raw_df[raw_df['type'] == 'ReleaseEvent'].copy()
                        releases_df['created_at'] = pd.to_datetime(releases_df['created_at'])
                        releases_df['date'] = releases_df['created_at'].dt.date
                        release_counts = releases_df.groupby('date').size().reset_index(name='release_count')
                        release_counts['date'] = pd.to_datetime(release_counts['date'])
                        
                        # If we have toxic comments data
                        if not self.toxic_comments_df.empty and 'created_at' in self.toxic_comments_df.columns:
                            # Prepare toxic comments by date
                            toxic_by_date = self.toxic_comments_df.copy()
                            toxic_by_date['date'] = pd.to_datetime(toxic_by_date['created_at']).dt.date
                            daily_toxicity = toxic_by_date.groupby('date').size().reset_index(name='toxic_count')
                            daily_toxicity['date'] = pd.to_datetime(daily_toxicity['date'])
                            
                            # Merge release and toxicity data
                            releases_toxicity = pd.merge(
                                release_counts,
                                daily_toxicity,
                                on='date',
                                how='outer'
                            ).fillna(0)
                            releases_toxicity = releases_toxicity.sort_values('date')
                            
                            # 1. Plot releases and toxic comments over time
                            fig, ax1 = plt.subplots(figsize=(12, 6))
                            
                            # Plot toxic comments
                            ax1.set_xlabel('Date')
                            ax1.set_ylabel('Number of Toxic Comments', color='tab:red')
                            ax1.plot(releases_toxicity['date'], releases_toxicity['toxic_count'], color='tab:red', marker='o', linestyle='-')
                            ax1.tick_params(axis='y', labelcolor='tab:red')
                            
                            # Create second y-axis
                            ax2 = ax1.twinx()
                            ax2.set_ylabel('Number of Releases', color='tab:green')
                            ax2.plot(releases_toxicity['date'], releases_toxicity['release_count'], color='tab:green', marker='s', linestyle='-')
                            ax2.tick_params(axis='y', labelcolor='tab:green')
                            
                            fig.tight_layout()
                            plt.title('Toxic Comments vs. Software Releases Over Time')
                            plt.savefig(os.path.join(self.figures_dir, 'rq2_toxicity_vs_releases_time.png'), dpi=300, bbox_inches='tight')
                            plt.close()
                            print("- Created: rq2_toxicity_vs_releases_time.png")
                            
                            # 2. Analyze toxic comments before/after releases
                            # Get dates with releases
                            release_dates = releases_toxicity[releases_toxicity['release_count'] > 0]['date']
                            
                            if not release_dates.empty:
                                # Calculate average toxic comments before and after releases
                                window_days = 3  # Window size in days
                                before_release = []
                                after_release = []
                                
                                for release_date in release_dates:
                                    # Get dates before and after release
                                    before_window = [release_date - pd.Timedelta(days=i) for i in range(1, window_days+1)]
                                    after_window = [release_date + pd.Timedelta(days=i) for i in range(1, window_days+1)]
                                    
                                    # Get toxic counts for these dates
                                    before_counts = releases_toxicity[releases_toxicity['date'].isin(before_window)]['toxic_count'].sum()
                                    after_counts = releases_toxicity[releases_toxicity['date'].isin(after_window)]['toxic_count'].sum()
                                    
                                    before_release.append(before_counts)
                                    after_release.append(after_counts)
                                
                                # Create bar chart
                                plt.figure(figsize=(10, 6))
                                avg_before = np.mean(before_release)
                                avg_after = np.mean(after_release)
                                
                                plt.bar(['Before Release', 'After Release'], [avg_before, avg_after], color=['skyblue', 'lightcoral'])
                                plt.title(f'Average Toxic Comments {window_days} Days Before vs. After Releases')
                                plt.ylabel('Average Number of Toxic Comments')
                                plt.savefig(os.path.join(self.figures_dir, 'rq2_toxicity_before_after_releases.png'), dpi=300, bbox_inches='tight')
                                plt.close()
                                print("- Created: rq2_toxicity_before_after_releases.png")
                                print(f"  Average toxic comments {window_days} days before release: {avg_before:.2f}")
                                print(f"  Average toxic comments {window_days} days after release: {avg_after:.2f}")
                                
                                # Calculate percent change
                                if avg_before > 0:
                                    percent_change = ((avg_after - avg_before) / avg_before) * 100
                                    print(f"  Percent change: {percent_change:.2f}%")
                        else:
                            print("  No toxic comments data available for releases correlation")
                    else:
                        print("  No release events found in the data")
                else:
                    print("  Raw data parquet file not found")
            
            except Exception as e:
                print(f"  Error loading parquet data: {str(e)}")
        
        except Exception as e:
            print(f"Error in toxicity vs. releases visualization: {str(e)}")
    
    def visualize_experience_vs_toxicity(self):
        """
        RQ3: How does the level of experience of the contributors correlate with 
        their likelihood of engaging in toxic communication within OSS communities?
        """
        print("\nVisualizing RQ3: Experience vs. Toxicity")
        
        # Check if we have enough data
        if self.user_metrics_df.empty or self.toxicity_by_user_df.empty:
            print("Not enough data to analyze RQ3")
            return
        
        try:
            # Merge user metrics with toxicity data
            user_experience = pd.merge(
                self.user_metrics_df,
                self.toxicity_by_user_df,
                left_on='user',
                right_on='user',
                how='inner'
            )
            
            if user_experience.empty:
                print("  No overlapping user data found")
                return
            
            # Use activity count as a proxy for experience if account age not available
            if 'activity_count' in user_experience.columns:
                # Create activity bins for experience levels
                experience_bins = [0, 5, 20, 50, 100, 1000, float('inf')]
                experience_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Expert']
                
                user_experience['experience_level'] = pd.cut(
                    user_experience['activity_count'], 
                    bins=experience_bins, 
                    labels=experience_labels
                )
                
                # 1. Create bar chart of toxicity ratio by experience level
                toxicity_by_experience = user_experience.groupby('experience_level')['toxicity_ratio'].mean().reset_index()
                
                plt.figure(figsize=(12, 6))
                sns.barplot(x='experience_level', y='toxicity_ratio', data=toxicity_by_experience, palette='viridis')
                plt.title('Average Toxicity Ratio by Developer Experience Level')
                plt.xlabel('Experience Level (based on activity count)')
                plt.ylabel('Average Toxicity Ratio')
                plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
                plt.savefig(os.path.join(self.figures_dir, 'rq3_toxicity_by_experience.png'), dpi=300, bbox_inches='tight')
                plt.close()
                print("- Created: rq3_toxicity_by_experience.png")
                
                # 2. Create scatterplot of activity count vs. toxicity ratio
                plt.figure(figsize=(10, 6))
                # Use log scale for better visualization if there's a wide range of activity counts
                if user_experience['activity_count'].max() > 100:
                    plt.xscale('log')
                
                sns.scatterplot(x='activity_count', y='toxicity_ratio', data=user_experience, alpha=0.6)
                plt.title('Developer Experience vs. Toxicity Ratio')
                plt.xlabel('Activity Count (log scale)')
                plt.ylabel('Toxicity Ratio')
                plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
                plt.savefig(os.path.join(self.figures_dir, 'rq3_experience_vs_toxicity_scatter.png'), dpi=300, bbox_inches='tight')
                plt.close()
                print("- Created: rq3_experience_vs_toxicity_scatter.png")
                
                # Calculate correlation
                corr = user_experience[['activity_count', 'toxicity_ratio']].corr().iloc[0,1]
                print(f"  Correlation between activity count and toxicity ratio: {corr:.4f}")
                
                # 3. Box plot of toxicity distribution by experience level
                plt.figure(figsize=(12, 6))
                sns.boxplot(x='experience_level', y='toxicity_ratio', data=user_experience)
                plt.title('Distribution of Toxicity Ratio by Experience Level')
                plt.xlabel('Experience Level')
                plt.ylabel('Toxicity Ratio')
                plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
                plt.savefig(os.path.join(self.figures_dir, 'rq3_toxicity_distribution_by_experience.png'), dpi=300, bbox_inches='tight')
                plt.close()
                print("- Created: rq3_toxicity_distribution_by_experience.png")
                
                # 4. Toxic comment count vs. experience level
                if 'toxic_comments' in user_experience.columns:
                    plt.figure(figsize=(12, 6))
                    sns.boxplot(x='experience_level', y='toxic_comments', data=user_experience)
                    plt.title('Number of Toxic Comments by Experience Level')
                    plt.xlabel('Experience Level')
                    plt.ylabel('Number of Toxic Comments')
                    plt.savefig(os.path.join(self.figures_dir, 'rq3_toxic_comment_count_by_experience.png'), dpi=300, bbox_inches='tight')
                    plt.close()
                    print("- Created: rq3_toxic_comment_count_by_experience.png")
            else:
                print("  No activity count data available to use as experience proxy")
        
        except Exception as e:
            print(f"Error in experience vs. toxicity visualization: {str(e)}")
    
    def create_dashboard(self):
        """Create a summary dashboard with key findings from all research questions"""
        try:
            # Create a figure with 2x2 grid
            fig, axs = plt.subplots(2, 2, figsize=(16, 12))
            plt.suptitle('GitHub Toxicity Analysis: Key Findings', fontsize=16, y=0.98)
            
            # For empty/incomplete data, draw placeholder text
            if self.toxic_comments_df.empty or self.commits_df.empty:
                for ax in axs.flat:
                    ax.text(0.5, 0.5, 'Insufficient data for visualization', 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax.transAxes, fontsize=12)
            else:
                # Plot 1: Toxicity Overview (Top-left)
                ax = axs[0, 0]
                if self.summary and 'total_records' in self.summary and 'toxic_comments' in self.summary:
                    labels = ['Non-toxic', 'Toxic']
                    sizes = [self.summary['total_records'] - self.summary['toxic_comments'], 
                            self.summary['toxic_comments']]
                    if sum(sizes) > 0:  # Ensure we have some data
                        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])
                        ax.set_title('Toxic vs. Non-toxic Comments')
                    else:
                        ax.text(0.5, 0.5, 'No data for pie chart', 
                                horizontalalignment='center', verticalalignment='center',
                                transform=ax.transAxes, fontsize=12)
                else:
                    ax.text(0.5, 0.5, 'No summary data available', 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax.transAxes, fontsize=12)

                # Plot 2: Toxicity vs. Commits Over Time (Top-right)
                ax = axs[0, 1]
                if not self.commits_df.empty and not self.toxic_comments_df.empty:
                    # Aggregate toxicity by date
                    toxic_by_date = self.toxic_comments_df.copy()
                    toxic_by_date['date'] = pd.to_datetime(toxic_by_date['created_at']).dt.date
                    daily_toxicity = toxic_by_date.groupby('date').size().reset_index(name='toxic_count')
                    daily_toxicity['date'] = pd.to_datetime(daily_toxicity['date'])

                    # Aggregate commits by date
                    daily_commits = self.commits_df.groupby('date')['commit_count'].sum().reset_index()

                    # Merge toxicity and commit data
                    productivity_df = pd.merge(daily_toxicity, daily_commits, on='date', how='outer').fillna(0)
                    productivity_df = productivity_df.sort_values('date')

                    # Plot Toxicity vs Commits over time
                    ax.plot(productivity_df['date'], productivity_df['toxic_count'], color='tab:red', label='Toxic Comments')
                    ax.set_ylabel('Toxic Comments', color='tab:red')
                    ax.tick_params(axis='y', labelcolor='tab:red')

                    ax2 = ax.twinx()
                    ax2.plot(productivity_df['date'], productivity_df['commit_count'], color='tab:blue', label='Commits')
                    ax2.set_ylabel('Commits', color='tab:blue')
                    ax2.tick_params(axis='y', labelcolor='tab:blue')

                    ax.set_title('Toxic Comments vs. Commit Activity Over Time')
                    ax.legend(loc='upper left')
                    ax2.legend(loc='upper right')
                else:
                    ax.text(0.5, 0.5, 'No data for time series', 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax.transAxes, fontsize=12)

                # Plot 3: Toxicity Ratio by Experience Level (Bottom-left)
                ax = axs[1, 0]
                if not self.user_metrics_df.empty and not self.toxicity_by_user_df.empty:
                    # Merge user metrics with toxicity data
                    user_experience = pd.merge(
                        self.user_metrics_df,
                        self.toxicity_by_user_df,
                        left_on='user',
                        right_on='user',
                        how='inner'
                    )

                    if 'activity_count' in user_experience.columns:
                        # Create experience bins
                        experience_bins = [0, 5, 20, 50, 100, 1000, float('inf')]
                        experience_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Expert']
                        user_experience['experience_level'] = pd.cut(
                            user_experience['activity_count'], 
                            bins=experience_bins, 
                            labels=experience_labels
                        )

                        # Group by experience level
                        toxicity_by_experience = user_experience.groupby('experience_level')['toxicity_ratio'].mean().reset_index()

                        # Plot bar chart
                        sns.barplot(x='experience_level', y='toxicity_ratio', data=toxicity_by_experience, palette='viridis', ax=ax)
                        ax.set_title('Average Toxicity Ratio by Experience Level')
                        ax.set_xlabel('Experience Level')
                        ax.set_ylabel('Average Toxicity Ratio')
                        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
                    else:
                        ax.text(0.5, 0.5, 'No activity count data', 
                                horizontalalignment='center', verticalalignment='center',
                                transform=ax.transAxes, fontsize=12)
                else:
                    ax.text(0.5, 0.5, 'No user metrics data', 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax.transAxes, fontsize=12)

                # Plot 4: Toxicity vs. Releases (Bottom-right)
                ax = axs[1, 1]
                if not self.toxic_comments_df.empty:
                    # Check if release data is available
                    raw_data_path = os.path.join(self.data_dir, 'github_data.parquet')
                    if os.path.exists(raw_data_path):
                        import pyarrow.parquet as pq
                        table = pq.read_table(raw_data_path, columns=['type', 'created_at'])
                        raw_df = table.to_pandas()

                        if 'type' in raw_df.columns and 'ReleaseEvent' in raw_df['type'].unique():
                            releases_df = raw_df[raw_df['type'] == 'ReleaseEvent'].copy()
                            releases_df['created_at'] = pd.to_datetime(releases_df['created_at'])
                            releases_df['date'] = releases_df['created_at'].dt.date
                            release_counts = releases_df.groupby('date').size().reset_index(name='release_count')
                            release_counts['date'] = pd.to_datetime(release_counts['date'])

                            # Prepare toxic comments by date
                            toxic_by_date = self.toxic_comments_df.copy()
                            toxic_by_date['date'] = pd.to_datetime(toxic_by_date['created_at']).dt.date
                            daily_toxicity = toxic_by_date.groupby('date').size().reset_index(name='toxic_count')
                            daily_toxicity['date'] = pd.to_datetime(daily_toxicity['date'])

                            # Merge release and toxicity data
                            releases_toxicity = pd.merge(
                                release_counts,
                                daily_toxicity,
                                on='date',
                                how='outer'
                            ).fillna(0)
                            releases_toxicity = releases_toxicity.sort_values('date')

                            # Plot releases and toxic comments over time
                            ax.plot(releases_toxicity['date'], releases_toxicity['toxic_count'], color='tab:red', label='Toxic Comments')
                            ax.set_ylabel('Toxic Comments', color='tab:red')
                            ax.tick_params(axis='y', labelcolor='tab:red')

                            ax2 = ax.twinx()
                            ax2.plot(releases_toxicity['date'], releases_toxicity['release_count'], color='tab:green', label='Releases')
                            ax2.set_ylabel('Releases', color='tab:green')
                            ax2.tick_params(axis='y', labelcolor='tab:green')

                            ax.set_title('Toxic Comments vs. Software Releases Over Time')
                            ax.legend(loc='upper left')
                            ax2.legend(loc='upper right')
                        else:
                            ax.text(0.5, 0.5, 'No release events found', 
                                    horizontalalignment='center', verticalalignment='center',
                                    transform=ax.transAxes, fontsize=12)
                    else:
                        ax.text(0.5, 0.5, 'No raw data available', 
                                horizontalalignment='center', verticalalignment='center',
                                transform=ax.transAxes, fontsize=12)
                else:
                    ax.text(0.5, 0.5, 'No toxic comments data', 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax.transAxes, fontsize=12)

            # Adjust layout and save dashboard
            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_dir, 'dashboard_summary.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("- Created: dashboard_summary.png")

        except Exception as e:
            print(f"Error creating dashboard: {str(e)}")


if __name__ == "__main__":
    # Initialize visualizer
    visualizer = GitHubToxicityVisualizer(data_dir="output")
    
    # Load data
    visualizer.load_data()
    
    # Generate all visualizations
    visualizer.generate_all_visualizations()