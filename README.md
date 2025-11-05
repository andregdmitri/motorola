# Jeopardy Data Curation Project

This project curates specialized datasets from Jeopardy questions for Named Entity Recognition (NER) evaluation. It processes the JEOPARDY_QUESTIONS1.json file to create three distinct strata suitable for comparing NER algorithm performance.

## Strata Types

1. **Phrases Containing Numbers**
   - Detects both numeric digits and spelled-out numbers
   - Includes percentage values and decimal numbers
   - Uses regex pattern matching for reliable detection

2. **Phrases Containing Non-English Words**
   - Uses two-stage detection:
     - Language detection using `langdetect` library
     - Non-ASCII character detection as a fallback
   - Handles both full non-English phrases and mixed language content

3. **Phrases Containing Unusual Proper Nouns**
   - Builds a corpus-wide frequency counter for proper nouns
   - Uses spaCy for Named Entity Recognition and POS tagging
   - Considers entities "unusual" based on frequency thresholds
   - Combines both NER and POS tagging results for better coverage

## Curation Process

1. **Data Loading (JeopardyDataLoader)**
   - Streams JSON using `ijson` to handle large files efficiently
   - Normalizes data fields (e.g., converts dollar values to integers)
   - Handles null values and data cleaning

2. **Text Analysis (TextAnalyzer)**
   - Uses spaCy (`en_core_web_md` model) for NLP tasks
   - Provides utilities for:
     - Number detection
     - Language detection
     - Named entity extraction
     - POS tagging

3. **Classification (JeopardyRecordClassifier)**
   - Maintains corpus statistics
   - Classifies records into different strata
   - Handles edge cases and error conditions

4. **Result Generation (JeopardyCurator)**
   - Creates random samples of 1000 records per stratum
   - Saves results in JSONL format
   - Provides total counts for each category

## Usage

```python
from data_curator.curate import JeopardyCurator
from utils.constants import *

# Initialize curator
curator = JeopardyCurator(
    source_file="data/JEOPARDY_QUESTIONS1.json",
    sample_size=1000
)

# Process records and get results
results = curator.process_records()

# Results are saved to data/output/*.jsonl
```

## Third-Party Libraries

- **ijson**: Stream-processes JSON files without loading entire file into memory
- **spaCy**: Industrial-strength Natural Language Processing
- **langdetect**: Language detection based on character sequences
- **tqdm**: Progress bar for long-running processes
- **pathlib**: Object-oriented filesystem paths
- **typing**: Type hints for better code documentation

## Project Structure
```
├── README.md                    # This file
├── main.py                      # CLI entry point that runs the curator
├── data/
│   ├── JEOPARDY_QUESTIONS1.json  # Source JSON (200K records)
│   └── output/                   # Generated strata (jsonl files)
├── data_curator/
│   ├── __init__.py
│   ├── classifier.py             # Record classification logic (JeopardyRecordClassifier)
│   └── curate.py                 # Main curation orchestrator (JeopardyCurator)
├── dataloader/
│   ├── __init__.py
│   └── dataloader.py             # Streaming loader and save helper (JeopardyDataLoader, JeopardyRecord)
├── utils/
│   ├── __init__.py
│   ├── constants.py              # Config from env vars (DEFAULT_DATA_PATH SAMPLE_SIZE, etc.)
│   └── text_analysis.py          # TextAnalyzer (spaCy + langdetect helpers)
│   └── __pycache__/
├── notebooks/
│   ├── 1_initial_data_exploration.ipynb
│   └── 2_final_results.ipynb
```

## Statistics

The code provides total counts for each stratum before sampling, allowing estimation of the prevalence of each pattern in the full dataset of approximately 200K questions.