import re
import spacy
from typing import Set, List, Tuple
from langdetect import detect_langs

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
        spacy.require_gpu()
        self.nlp = spacy.load(spacy_model)

    def contains_number(self, text: str) -> bool:
        """Check if text contains number-like tokens."""
        return bool(NUMBER_RE.search(text))

    def contains_non_english(self, text: str, prob_threshold: float = 0.75) -> bool:
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

    def extract_proper_nouns(self, text: str) -> Set[str]:
        """Extract proper noun tokens from text."""
        doc = self.nlp(text)
        return {t.text for t in doc if t.pos_ == "PROPN"}

    def extract_named_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract named entities as (label, text) pairs."""
        doc = self.nlp(text)
        return [(ent.label_, ent.text) for ent in doc.ents]