import logging
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import random
from dataclasses import dataclass
import json
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
        
    def process_records(self, estimate_total: int = 217000, n_process: int = NUM_PROCESS, stratify: bool = False, batch_size: int = BATCH_SIZE) -> CurationResults:
        """
        Process all records in the dataset in batches, categorizing them by different criteria.
        Streams results directly to output files to minimize memory usage.
        Args:
            estimate_total: Estimated total records for progress bar
            batch_size: Number of records to process per batch
        Returns:
            CurationResults containing statistics and categorized records (lists will be empty, only counts are valid)
        """
        # Initialize counters
        total_processed = 0
        number_count = 0
        non_english_count = 0
        unusual_count = 0

        # Prepare output folder and files
        logger.info(f"Processing records from {self.source_file}")
        output_dir = Path(DEFAULT_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        number_path = output_dir / "number_phrases.jsonl"
        non_english_path = output_dir / "non_english_phrases.jsonl"
        unusual_path = output_dir / "unusual_proper_nouns.jsonl"

        # Open output files in append mode
        number_f = open(number_path, 'w', encoding='utf-8')
        non_english_f = open(non_english_path, 'w', encoding='utf-8')
        unusual_f = open(unusual_path, 'w', encoding='utf-8')

        batch = []
        # Iterate over the dataset in batches
        for record in tqdm(self.loader.iter_rows(), total=estimate_total, desc="Processing"):
            batch.append(record)
            if len(batch) >= batch_size:
                n, ne, u = self._process_batch(batch, number_f, non_english_f, unusual_f, n_process)
                number_count += n
                non_english_count += ne
                unusual_count += u
                total_processed += len(batch)
                batch = []

        # Process any remaining records
        if batch:
            n, ne, u = self._process_batch(batch, number_f, non_english_f, unusual_f, n_process)
            number_count += n
            non_english_count += ne
            unusual_count += u
            total_processed += len(batch)

        # closing files
        number_f.close()
        non_english_f.close()
        unusual_f.close()

        totals = {
            "number_phrases": number_count,
            "non_english_phrases": non_english_count,
            "unusual_proper_nouns": unusual_count,
        }

        # computing stats
        stats = {
            "Total Records Processed": total_processed,
            "Statistics per Category": {
                category: {
                    "count": count,
                    "percentage": (count / total_processed) * 100 if total_processed else 0,
                    "estimated_total": int((count / total_processed) * 200000) if total_processed else 0
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

        # 2nd pass: draw sample_size (default 1000) examples from each output
        for name, path in [
            ("number", number_path),
            ("non_english", non_english_path),
            ("unusual", unusual_path),
        ]:
            with open(path, "r", encoding="utf-8") as f:
                items = [json.loads(x) for x in f]

            # split into chunks of self.sample_size
            for i in range(0, len(items), self.sample_size):
                chunk = items[i:i + self.sample_size]
                chunk_idx = i // self.sample_size
                chunk_out = output_dir / f"{name}_chunk_{chunk_idx:04d}.jsonl"
                with chunk_out.open("w", encoding="utf-8") as out:
                    for rec in chunk:
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                logger.info(f"saved chunk {chunk_idx} ({len(chunk)} rows) → {chunk_out}")

        return CurationResults(
            totals=totals,
            number_records=[],
            non_english_records=[],
            unusual_records=[]
        )

    def _process_batch(self, batch, number_f, non_english_f, unusual_f, n_process):
        import json

        # 0) batch update corpus propn stats
        texts = [r.get_full_text() for r in batch]
        batch_propn_results = self.text_analyzer.extract_proper_nouns(texts, n_process=n_process)
        for propn_set in batch_propn_results:
            for tok in propn_set:
                self.corpus_propn_counter[tok] += 1

        # 1) batch NER + PROPN for ANSWERS
        answers = [(r.answer or "") for r in batch]
        batch_answer_ents  = self.text_analyzer.extract_named_entities(answers, n_process=n_process)
        batch_answer_propns = self.text_analyzer.extract_proper_nouns(answers, n_process=n_process)

        n_count = ne_count = u_count = 0
        
        # loop throgh each record
        for idx, record in enumerate(batch):
            try:
                rec_dict = record.__dict__ if hasattr(record, '__dict__') else record
                full = record.get_full_text()

                if self.text_analyzer.contains_number(full):
                    number_f.write(json.dumps(rec_dict, ensure_ascii=False) + "\n")
                    n_count += 1

                if self.text_analyzer.contains_non_english(full):
                    non_english_f.write(json.dumps(rec_dict, ensure_ascii=False) + "\n")
                    ne_count += 1

                # unusual proper noun test
                ents   = batch_answer_ents[idx] if batch_answer_ents else []
                propns = batch_answer_propns[idx] if batch_answer_propns else set()
                tokens = {t for _, t in ents} | propns

                if tokens:
                    rare_count = sum(
                        1 for tok in tokens
                        if self.corpus_propn_counter.get(tok.strip(), 0) <= FREQ_THRESHOLD
                    )
                    if (rare_count / len(tokens)) >= SCORE_THRESHOLD:
                        unusual_f.write(json.dumps(rec_dict, ensure_ascii=False) + "\n")
                        u_count += 1

            except Exception as e:
                logger.debug(f"Error processing record: {e}")

        return n_count, ne_count, u_count
