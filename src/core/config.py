"""
Central configuration module for Trecl.
Responsible for loading environment variables and establishing safe access keys.
Fails fast if required API keys are missing.
"""

import os
from typing import List
from dotenv import load_dotenv
import itertools

# Load variables from the .env file in the project root
load_dotenv()

class RoundRobinKeyManager:
    """Manages a list of API keys and cycles through them in a round-robin fashion."""
    def __init__(self, keys: List[str]):
        if not keys:
            raise ValueError("API key list cannot be empty.")
        self.keys = keys
        self.iterator = itertools.cycle(self.keys)
        
    def get_next_key(self) -> str:
        """Returns the next key in the cycle."""
        return next(self.iterator)

def get_required_env_list(key: str) -> RoundRobinKeyManager:
    """
    Safely retrieves a comma-separated environment variable and initializes a round-robin manager.
    
    Args:
        key (str): The name of the environment variable requested.
        
    Returns:
        RoundRobinKeyManager: An initialized iterator for the keys.
        
    Raises:
        RuntimeError: If the environment variable is not set or is empty.
    """
    val = os.getenv(key)
    if not val:
        raise RuntimeError(f"{key} is not set in environment or is empty")
    
    # Split by comma, strip whitespace, and ignore empty entries
    keys = [k.strip() for k in val.split(',') if k.strip()]
    if not keys:
        raise RuntimeError(f"{key} contains no valid keys")
        
    return RoundRobinKeyManager(keys)

# Required API credentials - Now these are Managers, not strict strings
CEREBRAS_API_KEYS = get_required_env_list("CEREBRAS_API_KEY")
TAVILY_API_KEYS = get_required_env_list("TAVILY_API_KEY")

# Feature Flags
USE_MOCK_SEARCH = os.getenv("USE_MOCK_SEARCH", "true").lower() == "true"
