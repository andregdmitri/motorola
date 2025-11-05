import ijson
import logging
from pathlib import Path
from typing import Iterator, Dict, Any, Union, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class JeopardyRecord:
    """Container class for a Jeopardy record with utility methods."""
    category: Optional[str]
    air_date: Optional[str]
    question: Optional[str]
    value: Optional[int]
    answer: Optional[str]
    round: Optional[str]
    show_number: Optional[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JeopardyRecord':
        """Create a JeopardyRecord from a dictionary."""
        return cls(**{
            field: data.get(field) for field in cls.__dataclass_fields__
        })
    
    def get_full_text(self) -> str:
        """Get concatenated question and answer text."""
        return f"{self.question or ''} {self.answer or ''}"

class JeopardyDataLoader:
    """
    Streaming loader for JEOPARDY_QUESTIONS1.json

    - does NOT load full JSON into RAM
    - exposes `.iter_rows()` which yields normalized records one by one
    - keeps only required columns
    """
    COLUMNS = [
        "category", 
        "air_date", 
        "question", 
        "value", 
        "answer", 
        "round", 
        "show_number"
    ]
    
    def __init__(self, path: Union[str, Path]):
        """Initializes the loader with the path to the JSON file."""
        self.path = Path(path)
        if not self.path.exists():
            print(f"Warning: File not found at {self.path}")

    def iter_rows(self) -> Iterator[JeopardyRecord]:
        """
        Streams the JSON file and yields one normalized JeopardyRecord at a time.
        """
        try:
            with self.path.open("rb") as f:
                parser = ijson.items(f, "item")
                
                for record in parser:
                    clean = {}
                    for key in self.COLUMNS:
                        value = record.get(key, None)

                        # 1. Normalize "null-like" strings to None
                        if isinstance(value, str):
                            value_stripped = value.strip()
                            if value_stripped.lower() in {"", "n/a", "na", "null", "none"}:
                                value = None
                            else:
                                value = value_stripped

                        # 2. Normalize dollar values to integers
                        if key == "value" and isinstance(value, str):
                            value = self._normalize_value(value)
                        
                        clean[key] = value
                        
                    yield JeopardyRecord.from_dict(clean)
                    
        except FileNotFoundError:
            print(f"Error: Could not open file {self.path}")
            return
        except Exception as e:
            print(f"An error occurred during streaming: {e}")
            return

    @staticmethod
    def _normalize_value(value_str: str) -> Union[int, None]:
        """
        Converts a Jeopardy $ string (e.g., '$1,000') to an int.
        Returns None if parsing fails.
        """
        try:
            # Remove '$', ',' and whitespace, then convert to integer
            return int(value_str.replace("$", "").replace(",", "").strip())
        except (ValueError, TypeError, AttributeError):
            # Fails on "None", bad strings, etc.
            return None

    def save_jsonl(self, items: list, output_path: Union[str, Path], sample_size: int = None) -> None:
        """
        Save records to a JSONL file, optionally taking a random sample.
        
        Args:
            items: List of records to save
            output_path: Path where to save the JSONL file
            sample_size: If provided, save this many randomly sampled items
        """
        import json
        import random
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if sample_size and len(items) > sample_size:
            items = random.sample(items, sample_size)
            
        with output_path.open('w', encoding='utf-8') as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        print(f"Saved {len(items)} records to {output_path}")