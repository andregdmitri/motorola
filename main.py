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
    parser.add_argument("--stratify", action="store_true", default=False,
                        help="Use stratified sampling for output subsets")
    parser.add_argument("--no-stratify", dest="stratify", action="store_false",
                        help="Disable stratified sampling for output subsets")
    args = parser.parse_args()

    curator = JeopardyCurator(args.source_file, sample_size=args.sample_size)
    curator.process_records(estimate_total=args.estimate_total, stratify=args.stratify)