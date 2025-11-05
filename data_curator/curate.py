import logging
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import random
from dataclasses import dataclass

from utils.constants import *
from utils.text_analysis import TextAnalyzer
from dataloader.dataloader import JeopardyDataLoader, JeopardyRecord
from collections import Counter

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
        
        # Initialize analysis tools --- USING SPACY
        self.text_analyzer = TextAnalyzer(DEFAULT_SPACY_MODEL)
        self.corpus_propn_counter = Counter() # roper-noun frequency counter

    def update_corpus_stats(self, record: JeopardyRecord) -> None:
        """Update corpus-wide proper-noun frequency counts."""
        for tok in self.text_analyzer.extract_proper_nouns(record.get_full_text()):
            self.corpus_propn_counter[tok] += 1

    def contains_numbers(self, record: JeopardyRecord) -> bool:
        """Return True when record question/answer contains numeric tokens."""
        return self.text_analyzer.contains_number(record.get_full_text())

    def contains_non_english(self, record: JeopardyRecord) -> bool:
        """Return True when record question/answer contains non-English tokens."""
        return self.text_analyzer.contains_non_english(record.get_full_text())

    def has_unusual_proper_nouns(
        self,
        record: JeopardyRecord,
        freq_threshold: int = FREQ_THRESHOLD,
        score_threshold: float = SCORE_THRESHOLD,
    ) -> bool:
        """
        Decide whether the record's answer contains an unusual proportion of rare PROPN tokens.
        """
        if not record.answer:
            return False

        try:
            ents = self.text_analyzer.extract_named_entities(record.answer)
            propns = self.text_analyzer.extract_proper_nouns(record.answer)
            tokens = {t for _, t in ents} | propns
            if not tokens:
                return False

            rare_count = sum(
                1 for tok in tokens if self.corpus_propn_counter.get(tok.strip(), 0) <= freq_threshold
            )
            return (rare_count / len(tokens)) >= score_threshold
        except Exception as e:
            logger.warning(f"Error checking unusual proper nouns: {e}")
            return False
        
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
            try:
                # Update corpus-wide statistics once per record
                self.update_corpus_stats(record)

                # Classify record according to different criteria
                if self.contains_numbers(record):
                    number_records.append(record)

                if self.contains_non_english(record):
                    non_english_records.append(record)

                if self.has_unusual_proper_nouns(record):
                    unusual_records.append(record)
            except Exception as e:
                # One-line catch: log and continue to next record
                logger.debug(f"Error processing record: {e}")

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