
# 🧠 Document-Aware Conversational Assistant  
### A RAG-Based NLP System  


## 📘 Abstract  
This project implements a **Document-Aware Conversational Assistant** using **Retrieval-Augmented Generation (RAG)**. Users can upload PDF documents and ask questions naturally in English. The system retrieves contextually relevant sections using **semantic embeddings** and generates grounded responses using a **large language model (LLM)**.  

It demonstrates core NLP techniques such as **text preprocessing**, **sentence embeddings**, **cosine similarity-based retrieval**, and **context-grounded response generation**.  

***

## ❓ Problem Statement  
Finding specific information in long documents is challenging.  
Traditional keyword search often fails because of **vocabulary mismatch** — for example, searching *“vacation policy”* may not return results from a section labeled *“annual leave”*.  

This project solves that with **semantic search**, which understands meaning instead of relying on exact keyword matches.  

***

## 💡 Solution: RAG Architecture  
**RAG = Retrieval-Augmented Generation**

Instead of letting the LLM “guess” from memory (causing hallucination), this system:  
1. Retrieves relevant chunks from uploaded documents.  
2. Augments the user prompt with retrieved context.  
3. Generates answers grounded in actual sources.  

This improves accuracy, reduces hallucination, and enables **source transparency**.  

***

## 🗂️ Project Structure  

```bash
document-assistant/
├── config.py           # Configuration constants
├── utils.py            # Helper functions
├── nlp_core.py         # Core NLP logic
├── app.py              # Streamlit UI
├── requirements.txt    # Dependencies
├── .env                # API keys (excluded from git)
└── README.md           # Documentation
```

### File Overview
| File | Purpose | Key Contents |
|------|----------|--------------|
| `config.py` | Configuration settings | API keys, model names, chunk size (800), overlap (200), top-k (4), temperature (0.7) |
| `utils.py` | Helper functions | `truncate_text()`, `format_percentage()`, `parse_key_value_string()`, `safe_get()` |
| `nlp_core.py` | NLP logic | Text extraction, cleaning, chunking, embedding, similarity search, RAG pipeline |
| `app.py` | Streamlit UI | Chat interface with tabs (chat, infographic, image prompt) |
| `requirements.txt` | Python dependencies | Streamlit, Sentence Transformers, PyMuPDF, NumPy, OpenAI |
| `.env` | Secrets | `OPENAI_API_KEY=sk-xxxxx` (excluded from Git) |

***

## 🏗️ System Architecture  

### **Indexing Phase (Document Upload)**  
PDF → Extract (PyMuPDF) → Clean → Chunk (800c/200o) → Embed (384-dim) → Store in RAM  

### **Query Phase (User Question)**  
Question → Embed → Cosine Similarity → Top-4 Chunks → Build Prompt → GPT-4o-mini → Answer + Sources  

***

## ⚙️ Key Implementation Details  

### **6.1 Text Processing**  
- Extraction: PyMuPDF extracts text from PDFs.  
- Cleaning: Removes noise such as extra spaces and special characters.  
- Chunking: 800-character chunks with 200-character overlap, aligned on sentence boundaries.  

### **6.2 Embeddings & Search**  
- Model: `all-MiniLM-L6-v2`  
- Vector Size: 384 dimensions per chunk.  
- Search: Cosine similarity for semantic closeness.  
- Retrieval: Top 4 most relevant chunks returned.  

### **6.3 Answer Generation**  
- LLM: OpenAI `gpt-4o-mini`.  
- Prompt: Retrieved chunks + last 3 conversation turns + current question.  
- Constraint: Answers are restricted to document context.  

### **6.4 Memory Solution**  
- Memory storage: `st.session_state.chat_history`.  
- Injects last 3 conversation turns into each prompt for continuity.  

### **6.5 Infographic Generation**  
- Extracts structured data from LLM responses → Parses into dictionary → Injects into HTML template.  
- HTML allows fast, local visualization without external APIs.  

***

## 🧩 Technologies Used  

| Component | Technology | Why Chosen |
|------------|-------------|-----------|
| UI | Streamlit | Easy Python-native UI |
| PDF Extraction | PyMuPDF | Fast and stable |
| Embeddings | Sentence-Transformers | Compact & high-quality |
| LLM | OpenAI GPT-4o-mini | Cost-efficient and reliable |
| Vector Storage | NumPy | Simple for demo-scale needs |
| Similarity | Cosine Similarity | Captures semantic meaning |

**Alternatives Not Used:**  
- **React/Flask:** Added complexity for prototype.  
- **Word2Vec:** Word-level only, lacks sentence context.  
- **Local LLM:** Requires GPU.  
- **FAISS/Pinecone:** Overkill for small demos.  

***

## 🌟 Features  

- 📄 PDF Upload & Text Extraction  
- 🧠 Natural-Language Search & Q&A  
- 🔍 Source Attribution (view retrieved chunks)  
- 💬 Conversation Memory for follow-ups  
- 🗃️ Multi-document Support  
- 📊 HTML Infographic Generation  
- 🎨 AI Image Prompt Generator  

***

## 🔧 Key Parameters  

| Parameter | Value | Reason |
|------------|--------|--------|
| Chunk Size | 800 chars | Keeps paragraph context |
| Overlap | 200 chars | Avoids boundary loss |
| Top-k | 4 | Covers context while avoiding noise |
| Embedding Dims | 384 | From MiniLM model |
| Memory Turns | 3 | Retains limited chat continuity |
| Temperature | 0.7 | Balances factual and creative tone |

***

## ⚠️ Limitations  

| Limitation | Reason | Future Fix |
|-------------|---------|------------|
| Data lost on refresh | Stored in memory only | Add persistent database |
| Needs internet | OpenAI API dependency | Use local LLM |
| Text-only | No image/table handling | Multi-modal RAG |
| Small scale | NumPy brute-force search | Upgrade to FAISS/Pinecone |
| Noisy extraction | Complex PDF layouts | Enhanced parsing pipeline |

***

## 🚀 Future Scope  
- Integrate **FAISS/Pinecone** for scalable vector storage.  
- Add **local LLMs** (Llama/Mistral) for offline use.  
- Support **multi-modal RAG** (text + tables + images).  
- Introduce **hybrid retrieval** (semantic + keyword).  
- Implement **user authentication & document libraries**.  

***

## 🧭 How to Run  

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create a .env file
echo "OPENAI_API_KEY=sk-your-api-key-here" > .env

# 3. Launch the app
streamlit run app.py
```

***

## 🏁 Conclusion  
The **Document-Aware Conversational Assistant** demonstrates how **RAG** can blend **semantic search** with **language generation** to build intelligent, grounded NLP systems.  
It mitigates vocabulary mismatch, reduces hallucination, and forms the foundation for scalable tools like **ChatGPT (file upload)** and **Google NotebookLM**.  

***

## 📚 References  
1. Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.  
2. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.*  
3. [Sentence-Transformers Documentation](https://www.sbert.net/)  
4. [Streamlit Docs](https://docs.streamlit.io/)  
5. [OpenAI API Docs](https://platform.openai.com/docs/)  

***

## 📦 Appendix  

### **requirements.txt**
```
streamlit>=1.28.0
sentence-transformers>=2.2.0
pymupdf>=1.23.0
numpy>=1.24.0
openai>=1.0.0
python-dotenv>=1.0.0
```

### **.env (sample)**
```
OPENAI_API_KEY=sk-your-api-key-here
```

### **.gitignore**
```
.env
__pycache__/
*.pyc
.streamlit/
```

**Total Files:** 6 (`config.py`, `utils.py`, `nlp_core.py`, `app.py`, `requirements.txt`, `.env`)  
**Total LOC:** ~600  
**Core NLP Techniques:** Text preprocessing -  Sentence Embeddings -  Cosine Similarity -  RAG Pipeline -  Prompt Engineering  
