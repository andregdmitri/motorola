
import os
from dotenv import load_dotenv

load_dotenv()
DEFAULT_DATA_PATH = os.getenv("DEFAULT_DATA_PATH", "data/JEOPARDY_QUESTIONS1.json")