import pandas as pd

# Export Data to CSV
def export_csv(all_results, all_toxic_comments):
    # (As a last resort I will use powerbi from my work to genereate some figures if none have been made)
    commit_records = []
    for result in all_results:
        issue_url = result['issue_url']
        toxic_comment = result['toxic_comment']
        # used gpt to help structure data into proper CSV format    
        for commit_data in result['daily_data']:
            record = {
                'issue_url': issue_url,
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
    
    commit_df.to_csv('commit_data.csv', index=False)
    
    # 2. Toxic comment details
    toxic_df = pd.DataFrame(all_toxic_comments)
    toxic_df.to_csv('toxic_comments.csv', index=False)
    
    print(f"Exported {len(commit_records)} commit records to commit_data.csv")
    print(f"Exported {len(all_toxic_comments)} toxic comments to toxic_comments.csv")