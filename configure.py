import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LLM = os.getenv("LLM_MODEL")
LLM_TEMPERATURE = os.getenv("LLM_TEMPERATURE")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
TOP_K_RESULT = os.getenv("TOP_K_RESULT")










