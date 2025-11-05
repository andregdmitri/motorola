
import os
from dotenv import load_dotenv

load_dotenv()
DEFAULT_DATA_PATH = os.getenv("DEFAULT_DATA_PATH", "data/JEOPARDY_QUESTIONS1.json")
DEFAULT_SPACY_MODEL = os.getenv("DEFAULT_SPACY_MODEL", "en_core_web_sm") # CPU
# DEFAULT_SPACY_MODEL = os.getenv("DEFAULT_SPACY_MODEL", "en_core_web_trf") # GPU
DEFAULT_OUTPUT_DIR = os.getenv("DEFAULT_OUTPUT_DIR", "data/output/")
DEFAULT_SEED = int(os.getenv("DEFAULT_SEED", "42"))
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "1000"))