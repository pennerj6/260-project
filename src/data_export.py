import pandas as pd
import json
import csv

# Export to CSV
def export_csv(all_results, all_toxic_comments, issue_details, discussion_metrics):
    # 1. Core commit data CSV (from original code)
    commit_records = []
    
    for result in all_results:
        issue_url = result['issue_url']
        issue_info = issue_details.get(issue_url, {})
        repo_owner = issue_info.get('owner', 'unknown')
        repo_name = issue_info.get('name', 'unknown')
        issue_number = issue_info.get('number', 'unknown')
        issue_title = issue_info.get('title', 'unknown')
        
        toxic_comment = result['toxic_comment']
        
        for commit_data in result['daily_data']:
            record = {
                'repository': f"{repo_owner}/{repo_name}",
                'issue_number': issue_number,
                'issue_title': issue_title,
                'toxic_comment_id': toxic_comment['id'],
                'toxic_comment_user': toxic_comment['user'],
                'toxic_comment_date': toxic_comment['created_at'],
                'toxicity_score': toxic_comment['toxicity_score'],
                'commit_date': commit_data['date'],
                'days_from_toxic_comment': commit_data['days_from_comment'],
                'period': commit_data['period'],
                'commit_author': commit_data['author'],
                'commit_email': commit_data['email'],
                'commit_message': commit_data['message']
            }
            commit_records.append(record)
    
    # Create DataFrame and export
    commit_df = pd.DataFrame(commit_records)
    commit_df.to_csv('powerbi_commit_data.csv', index=False)
    
    # 2. Summary statistics by issue and toxic comment
    summary_records = []
    
    for result in all_results:
        issue_url = result['issue_url']
        issue_info = issue_details.get(issue_url, {})
        repo_owner = issue_info.get('owner', 'unknown')
        repo_name = issue_info.get('name', 'unknown')
        issue_number = issue_info.get('number', 'unknown')
        issue_title = issue_info.get('title', 'unknown')
        
        toxic_comment = result['toxic_comment']
        
        # Get associated discussion metrics if available
        if isinstance(discussion_metrics, dict) and discussion_metrics.get('toxic_comment_id') == toxic_comment['id']:
            discussion_data = discussion_metrics
        else:
            discussion_data = {}
        
        record = {
            'repository': f"{repo_owner}/{repo_name}",
            'issue_number': issue_number,
            'issue_title': issue_title,
            'toxic_comment_id': toxic_comment['id'],
            'toxic_comment_user': toxic_comment['user'],
            'toxic_comment_date': toxic_comment['created_at'],
            'toxicity_score': toxic_comment['toxicity_score'],
            # Commit metrics
            'commits_before': result['before_count'],
            'commits_after': result['after_count'],
            'commit_percent_change': result['percent_change'],
            'commit_absolute_change': result['after_count'] - result['before_count'],
            # Discussion metrics
            'comments_before': discussion_data.get('before', {}).get('comment_count', 0),
            'comments_after': discussion_data.get('after', {}).get('comment_count', 0),
            'comment_percent_change': calculate_percentage_change(
                discussion_data.get('before', {}).get('comment_count', 0),
                discussion_data.get('after', {}).get('comment_count', 0)
            ),
            'participants_before': discussion_data.get('before', {}).get('unique_participants', 0),
            'participants_after': discussion_data.get('after', {}).get('unique_participants', 0),
            'response_time_before': discussion_data.get('before', {}).get('avg_response_time', 0),
            'response_time_after': discussion_data.get('after', {}).get('avg_response_time', 0),
        }
        summary_records.append(record)
    
    # Create DataFrame and export
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv('powerbi_summary_data.csv', index=False)
    
    # 3. Toxic comment details
    toxic_df = pd.DataFrame(all_toxic_comments)
    toxic_df.to_csv('powerbi_toxic_comments.csv', index=False)
    
    print(f"Exported {len(commit_records)} commit records to powerbi_commit_data.csv")
    print(f"Exported {len(summary_records)} summary records to powerbi_summary_data.csv")
    print(f"Exported {len(all_toxic_comments)} toxic comments to powerbi_toxic_comments.csv")

# Percent change btween 2 vals (neg means decreased, 0 means same , pos means inreased)
def calculate_percentage_change(old_value, new_value):
    if old_value == 0:
        return "N/A" if new_value == 0 else "Infinite"
    
    return ((new_value - old_value) / old_value) * 100


# Craete summary of our findings from all analyzed issues
def export_research_summary(issue_details):
    summary_data = {
        'total_issues_analyzed': len(issue_details),
        'issue_resolution': {},
        'release_toxicity': {},
        'contributor_experience': {}
    }
    
    # Aggregate issue resolution metrics
    toxic_resolution_time = []
    non_toxic_resolution_time = []
    toxic_abandonment_rate = 0
    non_toxic_abandonment_rate = 0
    
    # Aggregate release toxicity metrics
    pre_release_toxicity = []
    post_release_toxicity = []
    
    # Aggregate contributor experience metrics
    new_contributor_toxicity = []
    experienced_contributor_toxicity = []
    
    # Process each issue to collect aggregated metrics
    for issue_url, details in issue_details.items():
        # Extract issue resolution metrics
        if 'resolution_metrics' in details:
            toxic_resolution_time.extend(details['resolution_metrics']['toxic']['time_to_close'])
            non_toxic_resolution_time.extend(details['resolution_metrics']['non_toxic']['time_to_close'])
            toxic_abandonment_rate += details['resolution_metrics']['toxic']['abandonment_rate']
            non_toxic_abandonment_rate += details['resolution_metrics']['non_toxic']['abandonment_rate']
        
        # Extract release toxicity metrics
        if 'release_toxicity' in details:
            for release in details['release_toxicity']:
                pre_release_toxicity.append(release['pre_release_window']['toxicity_percentage'])
                post_release_toxicity.append(release['normal_window']['toxicity_percentage'])
        
        # Extract contributor experience metrics
        if 'contributor_analysis' in details:
            new_contributor_toxicity.append(details['contributor_analysis']['new_contributor_toxicity'])
            experienced_contributor_toxicity.append(details['contributor_analysis']['experienced_contributor_toxicity'])
    
    # Calculate averages and other statistical measures
    if toxic_resolution_time:
        summary_data['issue_resolution']['avg_toxic_resolution_time'] = sum(toxic_resolution_time) / len(toxic_resolution_time)
    else:
        summary_data['issue_resolution']['avg_toxic_resolution_time'] = 0
    
    if non_toxic_resolution_time:
        summary_data['issue_resolution']['avg_non_toxic_resolution_time'] = sum(non_toxic_resolution_time) / len(non_toxic_resolution_time)
    else:
        summary_data['issue_resolution']['avg_non_toxic_resolution_time'] = 0
    
    summary_data['issue_resolution']['toxic_abandonment_rate'] = toxic_abandonment_rate / len(issue_details) if issue_details else 0
    summary_data['issue_resolution']['non_toxic_abandonment_rate'] = non_toxic_abandonment_rate / len(issue_details) if issue_details else 0
    
    if pre_release_toxicity:
        summary_data['release_toxicity']['avg_pre_release_toxicity'] = sum(pre_release_toxicity) / len(pre_release_toxicity)
    else:
        summary_data['release_toxicity']['avg_pre_release_toxicity'] = 0
    
    if post_release_toxicity:
        summary_data['release_toxicity']['avg_post_release_toxicity'] = sum(post_release_toxicity) / len(post_release_toxicity)
    else:
        summary_data['release_toxicity']['avg_post_release_toxicity'] = 0
    
    if new_contributor_toxicity:
        summary_data['contributor_experience']['avg_new_contributor_toxicity'] = sum(new_contributor_toxicity) / len(new_contributor_toxicity)
    else:
        summary_data['contributor_experience']['avg_new_contributor_toxicity'] = 0
    
    if experienced_contributor_toxicity:
        summary_data['contributor_experience']['avg_experienced_contributor_toxicity'] = sum(experienced_contributor_toxicity) / len(experienced_contributor_toxicity)
    else:
        summary_data['contributor_experience']['avg_experienced_contributor_toxicity'] = 0
    
    # Write summarized data to CSV
    with open('research_summary.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header row
        writer.writerow(['Research Question', 'Metric', 'Value'])
        
        # Write general information
        writer.writerow(['General', 'Total Issues Analyzed', summary_data['total_issues_analyzed']])
        
        # Write issue resolution metrics
        writer.writerow(['Issue Resolution', 'Avg Toxic Resolution Time (hours)', summary_data['issue_resolution']['avg_toxic_resolution_time']])
        writer.writerow(['Issue Resolution', 'Avg Non-Toxic Resolution Time (hours)', summary_data['issue_resolution']['avg_non_toxic_resolution_time']])
        writer.writerow(['Issue Resolution', 'Toxic Abandonment Rate', summary_data['issue_resolution']['toxic_abandonment_rate']])
        writer.writerow(['Issue Resolution', 'Non-Toxic Abandonment Rate', summary_data['issue_resolution']['non_toxic_abandonment_rate']])
        
        # Write release toxicity metrics
        writer.writerow(['Release Toxicity', 'Avg Pre-Release Toxicity (%)', summary_data['release_toxicity']['avg_pre_release_toxicity']])
        writer.writerow(['Release Toxicity', 'Avg Post-Release Toxicity (%)', summary_data['release_toxicity']['avg_post_release_toxicity']])
        
        # Write contributor experience metrics
        writer.writerow(['Contributor Experience', 'Avg New Contributor Toxicity (%)', summary_data['contributor_experience']['avg_new_contributor_toxicity']])
        writer.writerow(['Contributor Experience', 'Avg Experienced Contributor Toxicity (%)', summary_data['contributor_experience']['avg_experienced_contributor_toxicity']])
    
    # Also create a JSON file with the full summary data for more detailed analysis
    with open('research_summary.json', 'w') as jsonfile:
        json.dump(summary_data, jsonfile, indent=4)
    
    print("Generated research summary files: research_summary.csv and research_summary.json")
