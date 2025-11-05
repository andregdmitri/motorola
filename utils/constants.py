
import os
from dotenv import load_dotenv

load_dotenv()
DEFAULT_DATA_PATH = os.getenv("DEFAULT_DATA_PATH", "data/JEOPARDY_QUESTIONS1.json")
DEFAULT_SPACY_MODEL = os.getenv("DEFAULT_SPACY_MODEL", "en_core_web_sm") # CPU
# DEFAULT_SPACY_MODEL = os.getenv("DEFAULT_SPACY_MODEL", "en_core_web_trf") # GPU
DEFAULT_OUTPUT_DIR = os.getenv("DEFAULT_OUTPUT_DIR", "data/output/")
DEFAULT_SEED = int(os.getenv("DEFAULT_SEED", "42"))
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "1000"))
FREQ_THRESHOLD = int(os.getenv("FREQ_THRESHOLD", "5"))  # Frequency threshold for unusual PROPNs
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.5")) # Threshold for unusual PROPNs
NUM_CORES = int(os.getenv("NUM_CORES", "-1"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
NON_ENGLISH_THRESHOLD = float(os.getenv("NON_ENGLISH_THRESHOLD", "0.75"))