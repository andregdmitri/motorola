import json
from pathlib import Path
from typing import Iterator, Dict, Any
import ijson  # local import so this module has zero heavy deps on import

COLUMNS = ["category", "air_date", "question", "value", "answer", "round", "show_number"]

class JeopardyDataLoader:
    """
    Streaming loader for JEOPARDY_QUESTIONS1.json

    - does NOT load full JSON into RAM
    - exposes `.iter_rows()` which yields records one by one
    - keeps only required columns
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def iter_rows(self) -> Iterator[Dict[str, Any]]:
        """
        yields one dict per record with only required columns
        """

        with self.path.open("rb") as f:
            parser = ijson.items(f, "item")
            for rec in parser:
                # keep only expected columns
                yield {k: rec.get(k) for k in COLUMNS}