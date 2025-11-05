import ijson
from pathlib import Path
from typing import Iterator, Dict, Any, Union

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

    def iter_rows(self) -> Iterator[Dict[str, Any]]:
        """
        Streams the JSON file and yields one normalized record dict at a time.
        """
        try:
            with self.path.open("rb") as f:
                # ijson.items(f, "item") assumes a structure like: [ {...}, {...} ]
                # It yields each object in the list as a dict.
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
                        
                    yield clean
                    
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