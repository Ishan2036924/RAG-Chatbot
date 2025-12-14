"""
=============================================================================
Core NLP Module
=============================================================================
All NLP processing functions for the Document Assistant.
=============================================================================
"""

import numpy as np
import re
import fitz
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Import settings from config
from config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_CHUNKS,
    MAX_CONVERSATION_TURNS,
    SUMMARY_INPUT_LIMIT,
    INFOGRAPHIC_INPUT_LIMIT
)

# Import helper functions from utils
from utils import parse_key_value_string

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# =============================================================================
# TEXT EXTRACTION
# =============================================================================

def extract_text_from_pdf(pdf_bytes):
    """Extract text from PDF file bytes."""
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    extracted_text = ""
    for page in document:
        extracted_text += page.get_text()
    document.close()
    return extracted_text


# =============================================================================
# TEXT PREPROCESSING
# =============================================================================

def clean_text(text):
    """Clean and normalize text."""
    text = re.sub(r'\n\s*\n', '\n\n', text)       # Multiple newlines → double
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # Single newline → space
    text = re.sub(r' +', ' ', text)               # Multiple spaces → single
    text = re.sub(r'\x00', '', text)              # Remove null chars
    return text.strip()


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text = text.strip()
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to end at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('. ')
            if last_period > chunk_size // 2:
                end = start + last_period + 1
                chunk = text[start:end]
        
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks


# =============================================================================
# EMBEDDING FUNCTIONS
# =============================================================================

def load_embedding_model():
    """Load sentence transformer model."""
    return SentenceTransformer(EMBEDDING_MODEL)


def generate_embeddings(texts, model):
    """Convert texts to vector embeddings."""
    return model.encode(texts, show_progress_bar=False)


# =============================================================================
# SIMILARITY SEARCH
# =============================================================================

def compute_cosine_similarity(query_embedding, document_embeddings):
    """Compute cosine similarity between query and documents."""
    dot_product = np.dot(document_embeddings, query_embedding)
    query_norm = np.linalg.norm(query_embedding)
    document_norms = np.linalg.norm(document_embeddings, axis=1)
    return dot_product / (query_norm * document_norms + 1e-8)


def retrieve_relevant_chunks(query, chunks, chunk_embeddings, model, top_k=TOP_K_CHUNKS):
    """Retrieve most relevant chunks for a query."""
    query_embedding = model.encode(query)
    similarity_scores = compute_cosine_similarity(query_embedding, chunk_embeddings)
    top_indices = np.argsort(similarity_scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "text": chunks[idx],
            "score": float(similarity_scores[idx])
        })
    return results


# =============================================================================
# LANGUAGE MODEL FUNCTIONS
# =============================================================================

def call_language_model(prompt, system_message):
    """Send prompt to OpenAI API."""
    if not client:
        return "Error: OpenAI API key not configured"
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


def build_conversation_context(chat_history, max_turns=MAX_CONVERSATION_TURNS):
    """Build context from previous conversation."""
    if not chat_history:
        return ""
    
    recent_history = chat_history[-(max_turns * 2):]
    context_parts = []
    
    for msg in recent_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        context_parts.append(f"{role}: {msg['content']}")
    
    return "\n".join(context_parts)


# =============================================================================
# RAG ANSWER GENERATION
# =============================================================================

def generate_rag_answer(question, retrieved_chunks, chat_history):
    """Generate answer using RAG with conversation memory."""
    document_context = "\n\n---\n\n".join([chunk["text"] for chunk in retrieved_chunks])
    conversation_context = build_conversation_context(chat_history)
    
    prompt = f"""You are an intelligent document assistant.

DOCUMENT CONTENT:
{document_context}

{f"PREVIOUS CONVERSATION:{chr(10)}{conversation_context}" if conversation_context else ""}

USER QUESTION: {question}

INSTRUCTIONS:
1. Answer based on the document content above
2. Summarize, explain, and create learning plans as needed
3. Make inferences and connections between ideas
4. If information isn't found, say so and suggest what IS covered
5. Use conversation history for context

Provide a helpful response:"""

    system_message = """You are an expert educational assistant that helps users understand documents.
Always be helpful and provide actionable insights."""

    return call_language_model(prompt, system_message)


# =============================================================================
# DOCUMENT ANALYSIS
# =============================================================================

def generate_document_summary(full_text):
    """Generate comprehensive document summary."""
    text_sample = full_text[:SUMMARY_INPUT_LIMIT]
    
    prompt = f"""Analyze this document and provide:

1. OVERVIEW: 2-3 sentence summary
2. KEY TOPICS: Main topics covered (bullet points)
3. KEY INSIGHTS: Most important takeaways (bullet points)
4. SUGGESTED LEARNING PATH: How to approach this material

DOCUMENT:
{text_sample}

Provide a well-structured summary:"""

    return call_language_model(prompt, "You are an expert at analyzing educational content.")


def extract_infographic_data(full_text):
    """Extract structured data for infographic."""
    text_sample = full_text[:INFOGRAPHIC_INPUT_LIMIT]
    
    prompt = f"""Extract key information in this EXACT format:
TITLE: [Concise title, max 6 words]
SUBTITLE: [One sentence description]
STAT1: [A number or key metric]
STAT1_LABEL: [What it represents]
STAT2: [Another metric]
STAT2_LABEL: [What it represents]
STAT3: [Third metric]
STAT3_LABEL: [What it represents]
POINT1: [Key takeaway 1]
POINT2: [Key takeaway 2]
POINT3: [Key takeaway 3]
POINT4: [Key takeaway 4]

DOCUMENT:
{text_sample}"""

    raw_response = call_language_model(prompt, "Extract information in the exact format requested.")
    return parse_key_value_string(raw_response)


def generate_image_prompt(summary):
    """Generate AI image prompt from summary."""
    prompt = f"""Create a detailed AI image generator prompt based on:

{summary}

Include: visual style, color scheme, key elements, composition, mood.

IMAGE PROMPT:"""

    return call_language_model(prompt, "You are a creative designer specializing in AI visuals.")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_api_key():
    """Check if OpenAI API key is configured."""
    return OPENAI_API_KEY is not None and len(OPENAI_API_KEY) > 0