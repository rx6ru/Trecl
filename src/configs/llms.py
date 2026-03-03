import os

from dotenv import load_dotenv

load_dotenv()

_key = os.getenv("CEREBRAS_API_KEY")

if _key is None:
    raise RuntimeError("CEREBRAS_API_KEY is not set in environment")

CEREBRAS_API_KEY: str = _key
