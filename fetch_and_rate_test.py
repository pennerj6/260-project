from github import Github, Auth
from toxicityrater import ToxicityRater

ACCESS_TOKEN="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"



tr = ToxicityRater()
g = Github(auth=Auth.Token(ACCESS_TOKEN))
repo = g.get_repo("adobe/adobe.github.com")
issue_comment = ""
# Just get the first comment from any of the issues for now
for issue in repo.get_issues():
    for comment in issue.get_comments():
        issue_comment = comment.body
        break
    if issue_comment:
        break
print(f"comment: {issue_comment}")
print(f"rating: {tr.get_toxicity_rating(issue_comment)}")
