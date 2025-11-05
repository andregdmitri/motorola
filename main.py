import argparse
from data_curator.curate import JeopardyCurator
from utils.constants import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curate Jeopardy strata.")
    parser.add_argument("source_file", type=str, nargs="?", default=DEFAULT_DATA_PATH, 
                       help="Path to the source data file")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE, 
                       help="sample size for each subset")
    parser.add_argument("--estimate-total", type=int, default=217000, 
                       help="estimated total for progress bar")
    args = parser.parse_args()
    
    curator = JeopardyCurator(args.source_file, sample_size=args.sample_size)
    curator.process_records(estimate_total=args.estimate_total)