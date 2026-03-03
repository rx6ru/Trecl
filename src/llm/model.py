"""
LLM Configuration and Initialization for Trecl mult-agent system.
Currently configured to use Cerebras explicitly.
"""

from langchain_cerebras import ChatCerebras
from pydantic import SecretStr

from core.config import CEREBRAS_API_KEYS

def get_llm() -> ChatCerebras:
    """
    Initializes and returns the central Language Model instance.
    Utilizes a round-robin API key manager for load balancing.
    
    Returns:
        ChatCerebras: Authenticated instance of the Cerebras inference model.
    """
    return ChatCerebras(
        model="llama3.1-8b", # Falling back to known supported models if needed - using generic default
        api_key=SecretStr(CEREBRAS_API_KEYS.get_next_key()),
    )

# Note: We keep this as a function to intentionally get a fresh key per initialization
llm = get_llm()
