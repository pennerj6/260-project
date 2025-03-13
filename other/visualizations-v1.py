import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import json
import warnings

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

    def visualize_resolution_time_vs_toxicity(self):
        """Enhanced version of issue resolution time vs toxicity analysis with outlier handling"""
        if not self.issues_df.empty and 'resolution_time_hours' in self.issues_df.columns:
            # Calculate issue average resolution time by date
            self.issues_df['opened_date'] = pd.to_datetime(self.issues_df['opened_at']).dt.date
            
            # Handle outliers - cap resolution time at 95th percentile
            resolution_cap = np.percentile(self.issues_df['resolution_time_hours'], 95)
            issues_capped = self.issues_df.copy()
            issues_capped.loc[issues_capped['resolution_time_hours'] > resolution_cap, 'resolution_time_hours'] = resolution_cap
            
            issues_by_date = issues_capped.groupby('opened_date')['resolution_time_hours'].mean().reset_index()
            issues_by_date['opened_date'] = pd.to_datetime(issues_by_date['opened_date'])
            
            # Prepare toxicity data
            toxic_by_date = self.toxic_comments_df.copy()
            toxic_by_date['date'] = pd.to_datetime(toxic_by_date['created_at']).dt.date
            daily_toxicity = toxic_by_date.groupby('date').size().reset_index(name='toxic_count')
            daily_toxicity['date'] = pd.to_datetime(daily_toxicity['date'])
            
            # Merge with daily toxicity data
            resolution_vs_toxicity = pd.merge(
                issues_by_date, 
                daily_toxicity,
                left_on='opened_date',
                right_on='date',
                how='outer'
            ).fillna(0)
            
            # Add rolling average (7-day window)
            resolution_vs_toxicity = resolution_vs_toxicity.sort_values('date')
            resolution_vs_toxicity['resolution_time_rolling'] = resolution_vs_toxicity['resolution_time_hours'].rolling(window=7, min_periods=1).mean()
            resolution_vs_toxicity['toxic_count_rolling'] = resolution_vs_toxicity['toxic_count'].rolling(window=7, min_periods=1).mean()
            
            # Create plot with both scatter and rolling average
            plt.figure(figsize=(12, 8))
            
            # Scatter plot with lower alpha for individual points
            plt.scatter(resolution_vs_toxicity['toxic_count'], resolution_vs_toxicity['resolution_time_hours'], 
                    alpha=0.3, color='blue', label='Daily Data')
            
            # Regression line
            sns.regplot(x='toxic_count', y='resolution_time_hours', data=resolution_vs_toxicity, 
                    scatter=False, line_kws={'color':'red', 'label':'Linear Regression'})
            
            # Rolling average
            non_zero_rolling = resolution_vs_toxicity[(resolution_vs_toxicity['toxic_count_rolling'] > 0) & 
                                                    (resolution_vs_toxicity['resolution_time_rolling'] > 0)]
            if not non_zero_rolling.empty:
                plt.scatter(non_zero_rolling['toxic_count_rolling'], 
                        non_zero_rolling['resolution_time_rolling'],
                        color='green', s=50, label='7-day Rolling Average')
            
            plt.title('Relationship Between Toxic Comments and Issue Resolution Time')
            plt.xlabel('Number of Toxic Comments')
            plt.ylabel('Average Issue Resolution Time (hours)')
            plt.legend()
            
            # Add statistical significance test
            from scipy import stats
            r, p_value = stats.pearsonr(resolution_vs_toxicity['toxic_count'], 
                                    resolution_vs_toxicity['resolution_time_hours'])
            plt.annotate(f'Correlation: {r:.3f}\nP-value: {p_value:.4f}', 
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            plt.savefig(os.path.join(self.figures_dir, 'rq1_toxicity_vs_resolution_time_enhanced.png'), 
                    dpi=300, bbox_inches='tight')
            plt.close()
            print("- Created: rq1_toxicity_vs_resolution_time_enhanced.png")
            print(f"  Correlation coefficient: {r:.4f}, p-value: {p_value:.4f}")
    
    def analyze_toxicity_around_releases(self, release_dates, releases_toxicity):
        """Enhanced analysis of toxicity patterns around releases with configurable windows"""
        # Create multiple window sizes to analyze
        window_sizes = [3, 7, 14]  # Days before/after release
        results = []
        
        for window_days in window_sizes:
            before_release = []
            after_release = []
            during_release = []  # The release day itself
            
            for release_date in release_dates:
                # Get dates before, during, and after release
                before_window = [release_date - pd.Timedelta(days=i) for i in range(1, window_days+1)]
                after_window = [release_date + pd.Timedelta(days=i) for i in range(1, window_days+1)]
                
                # Get toxic counts and normalize by total activity if possible
                # For now just using raw counts
                before_counts = releases_toxicity[releases_toxicity['date'].isin(before_window)]['toxic_count'].sum()
                during_count = releases_toxicity[releases_toxicity['date'] == release_date]['toxic_count'].sum()
                after_counts = releases_toxicity[releases_toxicity['date'].isin(after_window)]['toxic_count'].sum()
                
                # Calculate per-day averages
                before_release.append(before_counts / window_days)
                during_release.append(during_count)
                after_release.append(after_counts / window_days)
            
            # Calculate averages
            avg_before = np.mean(before_release) if before_release else 0
            avg_during = np.mean(during_release) if during_release else 0
            avg_after = np.mean(after_release) if after_release else 0
            
            results.append({
                'window': window_days,
                'before': avg_before,
                'during': avg_during,
                'after': avg_after,
                'change_pct': ((avg_after - avg_before) / avg_before * 100) if avg_before > 0 else 0
            })
        
        # Create visualization
        plt.figure(figsize=(14, 8))
        
        # Set up positions for grouped bars
        bar_width = 0.25
        positions = np.arange(len(window_sizes))
        
        # Plot grouped bars
        plt.bar(positions - bar_width, [r['before'] for r in results], width=bar_width, 
            color='skyblue', label=f'Before Release')
        plt.bar(positions, [r['during'] for r in results], width=bar_width, 
            color='purple', label=f'During Release Day')
        plt.bar(positions + bar_width, [r['after'] for r in results], width=bar_width, 
            color='lightcoral', label=f'After Release')
        
        # Add labels and formatting
        plt.xticks(positions, [f'{w}-day window' for w in window_sizes])
        plt.title('Average Daily Toxic Comments Around Release Events')
        plt.ylabel('Average Daily Toxic Comments')
        plt.xlabel('Analysis Window Size')
        plt.legend()
        
        # Add percentage change annotations
        for i, r in enumerate(results):
            plt.annotate(f"{r['change_pct']:.1f}%", 
                    xy=(positions[i] + bar_width, r['after'] + 0.1),
                    ha='center', va='bottom', rotation=0,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'rq2_toxicity_around_releases_windows.png'), 
                dpi=300, bbox_inches='tight')
        plt.close()
        print("- Created: rq2_toxicity_around_releases_windows.png")
        
        # Table of results
        for r in results:
            print(f"  {r['window']}-day window: Before={r['before']:.2f}, During={r['during']:.2f}, After={r['after']:.2f}, Change={r['change_pct']:.2f}%")
        
        return results
    
    def advanced_experience_analysis(self):
        """Enhanced experience vs toxicity analysis with multiple experience measures"""
        if self.user_metrics_df.empty or self.toxicity_by_user_df.empty:
            print("Not enough data for advanced experience analysis")
            return
        
        # Merge user metrics with toxicity data
        user_data = pd.merge(
            self.user_metrics_df,
            self.toxicity_by_user_df,
            left_on='user',
            right_on='user',
            how='inner'
        )
        
        if user_data.empty:
            print("  No overlapping user data found")
            return
        
        # Create multiple experience metrics if available
        experience_metrics = []
        
        # 1. Activity count
        if 'activity_count' in user_data.columns:
            experience_metrics.append('activity_count')
        
        # 2. Commit count
        if 'commit_count' in user_data.columns:
            experience_metrics.append('commit_count')
        
        # 3. Calculate account age (days active) if we have dates
        if 'first_active' in user_data.columns and 'last_active' in user_data.columns:
            user_data['first_active'] = pd.to_datetime(user_data['first_active'])
            user_data['last_active'] = pd.to_datetime(user_data['last_active'])
            user_data['active_days'] = (user_data['last_active'] - user_data['first_active']).dt.days
            experience_metrics.append('active_days')
        
        # 4. If we have user registration dates
        if 'registration_date' in user_data.columns:
            user_data['registration_date'] = pd.to_datetime(user_data['registration_date'])
            # Current date as reference point
            reference_date = pd.to_datetime('now')  # or use the most recent date in your dataset
            user_data['account_age_days'] = (reference_date - user_data['registration_date']).dt.days
            experience_metrics.append('account_age_days')
        
        # If no experience metrics are available, use activity_count as default
        if not experience_metrics and 'activity_count' not in user_data.columns:
            print("  No experience metrics available, creating a default metric")
            # Count comments and commits if available as a basic activity measure
            for col in ['total_comments', 'commit_count']:
                if col not in user_data.columns:
                    user_data[col] = 0
            user_data['activity_count'] = user_data['total_comments'] + user_data['commit_count']
            experience_metrics = ['activity_count']
        
        # Create visualizations for each experience metric
        for metric in experience_metrics:
            # Determine percentile-based experience levels
            percentiles = [0, 25, 50, 75, 90, 100]
            thresholds = np.percentile(user_data[metric], percentiles)
            
            # Create category labels
            labels = ['Novice', 'Beginner', 'Intermediate', 'Experienced', 'Expert']
            
            def get_experience_level(value):
                for i in range(len(thresholds)-1):
                    if value >= thresholds[i] and value <= thresholds[i+1]:
                        return labels[i]
                return labels[-1]
            
            # Add experience level column
            level_col = f'{metric}_level'
            user_data[level_col] = user_data[metric].apply(get_experience_level)
            
            # 1. Create visualization: Toxicity vs Experience Level (Categorical)
            plt.figure(figsize=(12, 7))
            
            # Calculate mean and confidence intervals
            exp_groups = user_data.groupby(level_col)
            means = exp_groups['toxicity_ratio'].mean()
            
            # Standard error for confidence intervals
            from scipy import stats
            cis = {}
            for level, group in exp_groups:
                if len(group) >= 2:  # Need at least 2 points for confidence interval
                    ci = stats.norm.interval(0.95, 
                                        loc=group['toxicity_ratio'].mean(), 
                                        scale=group['toxicity_ratio'].sem())
                    cis[level] = (max(0, ci[0]), min(1, ci[1]))  # Limit to 0-1 range
                else:
                    cis[level] = (group['toxicity_ratio'].mean(), group['toxicity_ratio'].mean())
            
            # Bar chart with error bars
            plt.bar(means.index, means.values, color='skyblue', alpha=0.7)
            plt.errorbar(means.index, means.values, 
                        yerr=[[means[level]-cis[level][0] for level in means.index], 
                            [cis[level][1]-means[level] for level in means.index]],
                        fmt='none', ecolor='black', capsize=5)
            
            # Add count annotations
            for i, level in enumerate(means.index):
                plt.annotate(f"n={len(exp_groups.get_group(level))}", 
                        xy=(i, means[level]), 
                        xytext=(0, 10), 
                        textcoords='offset points',
                        ha='center')
            
            plt.title(f'Toxicity Ratio by {metric.replace("_", " ").title()} Level')
            plt.xlabel('Experience Level')
            plt.ylabel('Average Toxicity Ratio')
            plt.ylim(0, min(1, means.max() * 1.5))  # Set reasonable y limits
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
            plt.tight_layout()
            
            # Save figure
            filename = f'rq3_toxicity_by_{metric}_level_enhanced.png'
            plt.savefig(os.path.join(self.figures_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"- Created: {filename}")
            
            # 2. Create visualization: Scatter plot with regression
            plt.figure(figsize=(12, 7))
            
            # Calculate log scale for better visualization if appropriate
            if user_data[metric].max() / user_data[metric].min() > 100:
                # Add a small value to avoid log(0)
                plot_metric = np.log10(user_data[metric] + 1)
                log_scale = True
                xlabel = f'{metric.replace("_", " ").title()} (log scale)'
            else:
                plot_metric = user_data[metric]
                log_scale = False
                xlabel = f'{metric.replace("_", " ").title()}'
            
            # Create scatter plot with regression line
            sns.regplot(x=plot_metric, y='toxicity_ratio', data=user_data, 
                    scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
            
            # Add statistical analysis
            from scipy import stats
            r, p_value = stats.pearsonr(plot_metric, user_data['toxicity_ratio'])
            
            # Calculate Spearman rank correlation for non-linear relationships
            rho, p_rho = stats.spearmanr(user_data[metric], user_data['toxicity_ratio'])
            
            # Add annotations
            plt.annotate(
                f'Pearson r: {r:.3f} (p={p_value:.4f})\nSpearman ρ: {rho:.3f} (p={p_rho:.4f})',
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
            
            plt.title(f'Toxicity Ratio vs {metric.replace("_", " ").title()}')
            plt.xlabel(xlabel)
            plt.ylabel('Toxicity Ratio')
            plt.ylim(0, min(1, user_data['toxicity_ratio'].max() * 1.2))
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
            
            # Save figure
            filename = f'rq3_toxicity_vs_{metric}_scatter_enhanced.png'
            plt.savefig(os.path.join(self.figures_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"- Created: {filename}")
            
            # Print correlation results
            print(f"  {metric.replace('_', ' ').title()} correlations with toxicity:")
            print(f"    Pearson r: {r:.4f} (p-value: {p_value:.4f})")
            print(f"    Spearman rho: {rho:.4f} (p-value: {p_rho:.4f})")
# Main execution
if __name__ == "__main__":
    # Initialize visualizer
    visualizer = GitHubToxicityVisualizer(data_dir="output")
    
    # Load data
    visualizer.load_data()
    
    # Generate all visualizations
    visualizer.generate_all_visualizations()