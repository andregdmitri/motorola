import logging
from collections import Counter
from typing import Dict, Set
from dataloader.dataloader import JeopardyRecord
from utils.text_analysis import TextAnalyzer

logger = logging.getLogger(__name__)

class JeopardyRecordClassifier:
    """Classifies Jeopardy records based on various criteria."""

    def __init__(self, text_analyzer: TextAnalyzer):
        """
        Initialize classifier with text analysis utilities.
        
        Args:
            text_analyzer: TextAnalyzer instance for NLP operations
        """
        self.text_analyzer = text_analyzer
        self.corpus_propn_counter = Counter()
        
    def update_corpus_stats(self, record: JeopardyRecord) -> None:
        """Update corpus-wide statistics with a new record."""
        try:
            # Extract and count proper nouns from full text
            for token in self.text_analyzer.extract_proper_nouns(record.get_full_text()):
                self.corpus_propn_counter[token] += 1
        except Exception as e:
            logger.warning(f"Error updating corpus stats: {e}")

    def contains_numbers(self, record: JeopardyRecord) -> bool:
        """Check if record contains numerical content."""
        return self.text_analyzer.contains_number(record.get_full_text())

    def contains_non_english(self, record: JeopardyRecord) -> bool:
        """Check if record contains non-English content."""
        return self.text_analyzer.contains_non_english(record.get_full_text())

    def has_unusual_proper_nouns(
        self, 
        record: JeopardyRecord, 
        freq_threshold: int = 5, 
        score_threshold: float = 0.5
    ) -> bool:
        """
        Check if record contains unusual proper nouns based on corpus frequencies.
        
        Args:
            record: Record to check
            freq_threshold: Frequency below which a token is considered rare
            score_threshold: Fraction of rare tokens needed to mark as unusual
            
        Returns:
            True if the record contains an unusual proportion of rare proper nouns
        """
        if not record.answer:
            return False

        try:
            # Get all proper nouns/named entities from answer
            ents = self.text_analyzer.extract_named_entities(record.answer)
            propns = self.text_analyzer.extract_proper_nouns(record.answer)
            tokens = {t for _, t in ents} | propns
            
            if not tokens:
                return False

            # Calculate rarity score
            rare_count = sum(
                1 for tok in tokens 
                if self.corpus_propn_counter.get(tok.strip(), 0) <= freq_threshold
            )
            
            return (rare_count / len(tokens)) >= score_threshold
            
        except Exception as e:
            logger.warning(f"Error checking unusual proper nouns: {e}")
            return False