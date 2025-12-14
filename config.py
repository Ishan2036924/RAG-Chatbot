"""
=============================================================================
Configuration Constants
=============================================================================
All settings in one place. Change values here to tune the application.
=============================================================================
"""

import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# API CONFIGURATION
# =============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o-mini"
LLM_MAX_TOKENS = 1500
LLM_TEMPERATURE = 0.7

# =============================================================================
# EMBEDDING CONFIGURATION
# =============================================================================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

# =============================================================================
# TEXT CHUNKING CONFIGURATION
# =============================================================================
CHUNK_SIZE = 800           # Characters per chunk
CHUNK_OVERLAP = 200        # Overlap between chunks

# =============================================================================
# RETRIEVAL CONFIGURATION
# =============================================================================
TOP_K_CHUNKS = 4           # Number of chunks to retrieve

# =============================================================================
# CONVERSATION MEMORY
# =============================================================================
MAX_CONVERSATION_TURNS = 3  # Q&A pairs to remember

# =============================================================================
# TEXT LIMITS
# =============================================================================
SUMMARY_INPUT_LIMIT = 6000
INFOGRAPHIC_INPUT_LIMIT = 5000
PREVIEW_TEXT_LENGTH = 400

# =============================================================================
# UI CONFIGURATION
# =============================================================================
APP_TITLE = "Document Assistant"
APP_ICON = "📄"
APP_DESCRIPTION = "Upload documents, ask questions, and generate insights using RAG"