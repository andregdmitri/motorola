import logging
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import random
from dataclasses import dataclass

from utils.constants import *
from utils.text_analysis import TextAnalyzer
from dataloader.dataloader import JeopardyDataLoader, JeopardyRecord
from data_curator.classifier import JeopardyRecordClassifier

logger = logging.getLogger(__name__)

@dataclass
class CurationResults:
    """Container for results of the curation process."""
    totals: Dict[str, int]
    number_records: List[JeopardyRecord]
    non_english_records: List[JeopardyRecord]
    unusual_records: List[JeopardyRecord]

class JeopardyCurator:
    """Main class for curating and analyzing Jeopardy questions dataset."""
    
    def __init__(self, source_file: str, sample_size: int = SAMPLE_SIZE):
        """
        Initialize curator with data source and configuration.
        
        Args:
            source_file: Path to source JSON file
            sample_size: Size of random samples to save
        """
        random.seed(DEFAULT_SEED)
        
        self.source_file = source_file
        self.sample_size = sample_size
        self.loader = JeopardyDataLoader(source_file)
        
        # Initialize analysis tools
        self.text_analyzer = TextAnalyzer(DEFAULT_SPACY_MODEL)
        self.classifier = JeopardyRecordClassifier(self.text_analyzer)
        
    def process_records(self, estimate_total: int = 217000) -> CurationResults:
        """
        Process all records in the dataset, categorizing them by different criteria.
        
        Args:
            estimate_total: Estimated total records for progress bar
            
        Returns:
            CurationResults containing statistics and categorized records
        """
        # Initialize collectors
        number_records = []
        non_english_records = []
        unusual_records = []

        logger.info(f"Processing records from {self.source_file}")
        for record in tqdm(self.loader.iter_rows(), total=estimate_total, desc="Processing"):
            # Update corpus-wide statistics
            self.classifier.update_corpus_stats(record)

            # Classify record according to different criteria
            try:
                if self.classifier.contains_numbers(record):
                    number_records.append(record)
            except Exception as e:
                logger.debug(f"Error checking numbers: {e}")

            try:
                if self.classifier.contains_non_english(record):
                    non_english_records.append(record)
            except Exception as e:
                logger.debug(f"Error checking language: {e}")

            try:
                if self.classifier.has_unusual_proper_nouns(record):
                    unusual_records.append(record)
            except Exception as e:
                logger.debug(f"Error checking proper nouns: {e}")

        # Calculate statistics
        total_processed = estimate_total  # total records processed
        totals = {
            "number_phrases": len(number_records),
            "non_english_phrases": len(non_english_records),
            "unusual_proper_nouns": len(unusual_records),
        }
        
        # Calculate percentages and estimated totals in full dataset
        stats = {
            "Total Records Processed": total_processed,
            "Statistics per Category": {
                category: {
                    "count": count,
                    "percentage": (count / total_processed) * 100,
                    "estimated_total": int((count / total_processed) * 200000)
                }
                for category, count in totals.items()
            }
        }

        # Log detailed statistics
        logger.info("\n=== Dataset Statistics ===")
        logger.info(f"Total Records Processed: {stats['Total Records Processed']:,}")
        logger.info("\nCategory Breakdown:")
        for category, details in stats["Statistics per Category"].items():
            logger.info(f"\n{category}:")
            logger.info(f"  Found in sample: {details['count']:,}")
            logger.info(f"  Percentage: {details['percentage']:.2f}%")
            logger.info(f"  Estimated total in 200K records: {details['estimated_total']:,}")

        # Save results
        output_dir = Path(DEFAULT_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.loader.save_jsonl(number_records, output_dir / "number_phrases.jsonl", sample_size=self.sample_size)
        self.loader.save_jsonl(non_english_records, output_dir / "non_english_phrases.jsonl", sample_size=self.sample_size)
        self.loader.save_jsonl(unusual_records, output_dir / "unusual_proper_nouns.jsonl", sample_size=self.sample_size)

        return CurationResults(
            totals=totals,
            number_records=number_records,
            non_english_records=non_english_records,
            unusual_records=unusual_records
        )