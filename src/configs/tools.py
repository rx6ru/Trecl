import os

from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if TAVILY_API_KEY is None:
    raise RuntimeError("TAVILY_API_KEY is not set in environment")
