from config import DEFAULT_CONFIG
from github_analyzer import GitHubArchiveAnalyzer
from toxicity_rater import ToxicityRater

def main():
    # Use DEFAULT_CONFIG values from the confic file
    start_date = DEFAULT_CONFIG['start_date']
    end_date = DEFAULT_CONFIG['end_date']
    output_dir = DEFAULT_CONFIG['output_dir']
    
    sample_size = DEFAULT_CONFIG['sample_size'] # 1000 for now (prev was 10,000)
    toxicity_threshold = DEFAULT_CONFIG['toxicity_threshold'] # 0.50 (for now, probably will stay this #)


    
    # professor mentiontioned "Ghtorrent", in lecture around week 3, which is a BIG github data archive
    # but after researching into it and trying to get access to the data (even via twitter(X) links that the creator posted), the site is inactive and doenst work anymore
    # that rabbit hole lead me to here: https://www.gharchive.org/ this is a large github data archive that DOES work 
    analyzer = GitHubArchiveAnalyzer(
        start_date=start_date,
        end_date=end_date,
        toxicity_threshold=toxicity_threshold
    )

    rater = ToxicityRater(use_sampling=True, sample_size=sample_size)

    # Collect data
    analyzer.collect_data()

    # Process toxicity
    analyzer.process_toxicity(rater)

    if 1==0:
        # IN PROGRESS 
        # Analyze RQs
        analyzer.analyze_toxicity_productivity_correlation()
        analyzer.analyze_toxicity_release_correlation()
        analyzer.analyze_experience_toxicity_correlation()

        # Save results
        analyzer.save_results(output_dir)

        # Generate the specific CSV files
        analyzer.generate_specific_csv_files(output_dir)

        # Generate summary report
        analyzer.generate_summary_report(output_dir)

if __name__ == "__main__":
    main()