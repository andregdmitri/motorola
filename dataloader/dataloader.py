import ijson
import logging
from pathlib import Path
from typing import Iterator, Dict, Any, Union, Optional
from dataclasses import dataclass
import json
import random

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
    Streaming loader and balanced sampler for JEOPARDY_QUESTIONS1.json.
    - Streams JSON without loading all into RAM
    - Normalizes and yields JeopardyRecord objects
    - Provides stratified sampling for balanced output
    """

    COLUMNS = [
        "category", 
        "air_date", "question", 
        "value", 
        "answer", 
        "round", 
        "show_number"
    ]

    def __init__(self, path: Union[str, Path]):
        """Initialize loader with path to JSON file."""
        self.path = Path(path)
        if not self.path.exists():
            print(f"Warning: File not found at {self.path}")

    def iter_rows(self) -> Iterator[JeopardyRecord]:
        """Stream the JSON file and yield normalized JeopardyRecord objects."""
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
        except Exception as e:
            print(f"An error occurred during streaming: {e}")

    def save_jsonl(
        self,
        items: list,
        output_path: Union[str, Path],
        sample_size: int = None,
        stratify: bool = True,
        stratify_fields: list = ["air_date"]
    ) -> None:
        """
        Save records to a JSONL file, optionally using stratified or random sampling.
        Args:
            items: List of records to save
            output_path: Path to save JSONL
            sample_size: Number of records to save
            stratify: If True, use stratified sampling
            stratify_fields: Fields to balance (default: value, air_date, round)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if sample_size and len(items) > sample_size:
            if stratify:
                items = self.stratified_sample(items, sample_size, fields=stratify_fields)
            else:
                items = random.sample(items, sample_size)
        with output_path.open('w', encoding='utf-8') as f:
            for item in items:
                obj = self._to_dict(item)
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"Saved {len(items)} records to {output_path}")

    def stratified_sample(
        self,
        items: list,
        sample_size: int,
        fields: list = ["value", "air_date", "round"]
    ) -> list:
        """
        Return a sample with as uniform a distribution as possible over the specified fields.
        Args:
            items: List of JeopardyRecord or dicts
            sample_size: Number of records to sample
            fields: Fields to balance (default: value, air_date, round)
        Returns:
            List of sampled items
        """
        from collections import defaultdict
        import random
        if not items or sample_size <= 0:
            return []
        # Group by all field combinations, using air_date bins if air_date is a field
        buckets = defaultdict(list)
        for item in items:
            d = self._to_dict(item)
            key = []
            for f in fields:
                if f == "air_date":
                    key.append(self._air_date_bin(d.get("air_date")))
                else:
                    key.append(d.get(f))
            buckets[tuple(key)].append(item)
        # Calculate how many to take from each bucket (as even as possible)
        bucket_keys = list(buckets.keys())
        n_buckets = len(bucket_keys)
        per_bucket = max(1, sample_size // n_buckets)
        result = []
        for key in bucket_keys:
            group = buckets[key]
            if len(result) + per_bucket > sample_size:
                per_bucket = sample_size - len(result)
            if len(group) <= per_bucket:
                result.extend(group)
            else:
                result.extend(random.sample(group, per_bucket))
            if len(result) >= sample_size:
                break
        # If not enough, fill randomly from remaining
        if len(result) < sample_size:
            remaining = [item for group in buckets.values() for item in group if item not in result]
            if remaining:
                result.extend(random.sample(remaining, min(sample_size - len(result), len(remaining))))
        return result[:sample_size]

    @staticmethod
    def _air_date_bin(air_date: str) -> str:
        """Return air_date bin label for stratification."""
        if not air_date or not isinstance(air_date, str) or len(air_date) < 4:
            return "unknown"
        try:
            year = int(air_date[:4])
        except Exception:
            return "unknown"
        if 1984 <= year <= 1995:
            return "1984-1995"
        elif 1996 <= year <= 2002:
            return "1996-2002"
        elif 2003 <= year <= 2012:
            return "2003-2012"
        else:
            return "other"

    @staticmethod
    def _to_dict(item):
        """Convert JeopardyRecord or dataclass to dict."""
        if hasattr(item, 'to_dict'):
            return item.to_dict()
        elif hasattr(item, '__dict__') and not isinstance(item, dict):
            return item.__dict__
        return item

    @staticmethod
    def _normalize_value(value_str: str) -> Union[int, None]:
        """Convert Jeopardy $ string (e.g., '$1,000') to int, or None if parsing fails."""
        try:
            return int(value_str.replace("$", "").replace(",", "").strip())
        except (ValueError, TypeError, AttributeError):
            return None