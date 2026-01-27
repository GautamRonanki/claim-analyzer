import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"

# Input Constraints
MIN_TEXT_LENGTH = 10
MAX_TEXT_LENGTH = 10000
DEFAULT_MAX_CLAIMS = 10

# Uncertainty Labels (enforced)
VALID_UNCERTAINTY_LABELS = ["certain", "likely", "uncertain", "speculative"]

# Retry Configuration
MAX_RETRIES = 1