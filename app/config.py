import os
from dotenv import load_dotenv

load_dotenv()


# --- api keys ---
COHERE_API_KEY  = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")

# --- pinecone ---
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_NAMESPACE  = os.getenv("PINECONE_NAMESPACE", "md-rag")

# --- embedding model ---
COHERE_EMBED_MODEL = "embed-english-v3.0"

# --- llm ---
GEMINI_MODEL = "gemini-2.5-flash-lite"
LLM_TEMPERATURE = 0

# --- docs ---
DOCS_FOLDER = os.getenv("DOCS_FOLDER", "./docs")

# --- splitting ---
# markdown headers used to split on section boundaries first
HEADERS_TO_SPLIT_ON = [
    ("#",   "Header 1"),
    ("##",  "Header 2"),
    ("###", "Header 3"),
]

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150

# custom separators preserve markdown structure during recursive splitting
RECURSIVE_SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", " ", ""]

# --- retrieval ---
TOP_K = 4

# --- conversation ---
# max turns kept in history to avoid bloating the context window
MAX_HISTORY_TURNS = 10