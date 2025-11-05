# Jeopardy Data Curation

Curate focused evaluation sets from the Jeopardy questions corpus for NER and linguistic analysis. This project streams and analyzes a large JSON dump of Jeopardy clues to extract three evaluation strata useful for comparing NER systems and researching rare or tricky linguistic phenomena.

## What this repo produces
- Sampled JSONL datasets (under `data/output/`) containing records selected for three distinct strata:
  1. Phrases containing numbers (digits or written numbers, percentages, decimals)
  2. Phrases containing non-English text or non-ASCII tokens
  3. Phrases containing unusual / low-frequency proper nouns (based on corpus-wide counts and spaCy NER/POS)

These strata are designed to surface edge cases that commonly break off-the-shelf NER models.

## Quick start

1. Install dependencies (recommended: virtualenv or venv):

   pip install -r requirements.txt

2. (Optional) Install spaCy model used by the analyzer:

   python -m spacy download en_core_web_sm

3. Run the main curation script:

   python main.py

By default the curator reads `data/JEOPARDY_QUESTIONS1.json` and writes sampled jsonl outputs to `data/output/`.

## How it works (overview)

- Data loading: `dataloader/dataloader.py` contains a streaming JSON loader that uses ijson to iterate records without loading the entire file into memory. Records are normalized and cleaned as they are streamed.
- Text analysis: `utils/text_analysis.py` wraps spaCy and `langdetect` helpers for tasks such as language detection, number detection (including spelled-out numbers), NER, and POS tagging.
- Curation logic: `data_curator/curate.py` implements the selection rules for each stratum, builds corpus-wide frequency stats for proper nouns, performs sampling, and writes JSONL output files.

## Examples

Minimal usage from Python:

```python
from data_curator.curate import JeopardyCurator

curator = JeopardyCurator(source_file="data/JEOPARDY_QUESTIONS1.json", sample_size=1000)
results = curator.process_records()
# results contains summary counts and paths to generated files
```

Generated files (example):
- `data/output/numbers.jsonl` — sampled records with numeric content
- `data/output/non_english.jsonl` — sampled records flagged as non-English / non-ASCII
- `data/output/unusual_proper_nouns.jsonl` — sampled records containing low-frequency proper nouns

## Project structure

Top-level layout (important files):

```
main.py                     # CLI entrypoint
data/JEOPARDY_QUESTIONS1.json # Source corpus (large JSON)
data/output/                # Generated jsonl outputs
data_curator/curate.py      # Orchestrates selection & sampling
dataloader/dataloader.py    # Streaming loader & normalization helpers
utils/text_analysis.py      # spaCy + langdetect wrappers for detection & extraction
utils/constants.py          # Configuration (defaults, sample sizes, paths)
notebooks/                  # Exploratory analysis and results notebooks
```

## Dependencies

Key runtime dependencies (see `requirements.txt`):
- ijson — streaming JSON parsing
- spacy — NLP pipeline (NER/POS)
- langdetect — language identification
- tqdm — progress bars

## Notes & assumptions

- The pipeline expects a large JSON file where each item is a Jeopardy question record (the provided `JEOPARDY_QUESTIONS1.json`).
- SpaCy model `en_core_web_md` (or another English model) is recommended for accurate NER/POS.
- Sampling sizes and thresholds live in `utils/constants.py` and can be tuned for different evaluation budgets.

## Next steps / suggestions

- Add a small test-suite that validates loader normalization and the three detection heuristics (numbers, language, unusual proper nouns).
- Provide a lightweight CLI option to run each stratum independently for development and faster iteration.

## License

This repository does not include an explicit license file; add one (e.g., MIT) if you intend to publish or share the code.
