import re
import spacy
from typing import Set, List, Tuple
from langdetect import detect_langs

from utils.constants import BATCH_SIZE, NUM_CORES, NON_ENGLISH_THRESHOLD

# Regexes for text analysis
NUMBER_RE = re.compile(
    r"\b(?:\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?%|"
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million))\b",
    flags=re.IGNORECASE
)

ACCENTED_RE = re.compile(r"[^\x00-\x7F]")  # non-ascii characters

class TextAnalyzer:
    """Utility class for text analysis operations using spaCy."""
    
    def __init__(self, spacy_model: str):
        """Initialize with a spaCy model name."""
        # spacy.require_gpu()
        self.nlp = spacy.load(spacy_model)

    def contains_number(self, text: str) -> bool:
        """Check if text contains number-like tokens."""
        return bool(NUMBER_RE.search(text))

    def contains_non_english(self, text: str, prob_threshold: float = NON_ENGLISH_THRESHOLD) -> bool:
        """
        Check if text contains non-English content using both langdetect and character analysis.
        
        Args:
            text: Text to analyze
            prob_threshold: Confidence threshold for language detection
        """
        if not text.strip():
            return False

        # 1. Try language detection
        try:
            langs = detect_langs(text)
            if langs and hasattr(langs[0], "lang") and langs[0].lang != 'en' and langs[0].prob > prob_threshold:
                return True
        except Exception:
            pass  # langdetect can fail on short strings

        # 2. Check for non-ascii characters
        return bool(ACCENTED_RE.search(text))
    
    def extract_proper_nouns(self, texts: list, n_process: int = NUM_CORES) -> list:
        """Batch extract proper nouns from a list of texts using multiple processes."""
        results = []
        for doc in self.nlp.pipe(texts, n_process=n_process, batch_size=BATCH_SIZE):
            results.append({t.text for t in doc if t.pos_ == "PROPN"})
        return results

    def extract_named_entities(self, texts: list, n_process: int = NUM_CORES) -> list:
        """Batch extract named entities from a list of texts using multiple processes."""
        results = []
        for doc in self.nlp.pipe(texts, n_process=n_process, batch_size=BATCH_SIZE):
            results.append([(ent.label_, ent.text) for ent in doc.ents])
        return results