"""
=============================================================================
Document Assistant - Frontend
=============================================================================
Streamlit UI for the RAG-based document assistant.
Run with: streamlit run app.py
=============================================================================
"""

import streamlit as st

# Import from config
from config import (
    APP_TITLE,
    APP_ICON,
    APP_DESCRIPTION,
    TOP_K_CHUNKS,
    PREVIEW_TEXT_LENGTH
)

# Import from utils
from utils import truncate_text, format_percentage, safe_get

# Import from nlp_core
from nlp_core import (
    extract_text_from_pdf,
    clean_text,
    chunk_text,
    load_embedding_model,
    generate_embeddings,
    retrieve_relevant_chunks,
    generate_rag_answer,
    generate_document_summary,
    extract_infographic_data,
    generate_image_prompt,
    check_api_key
)


# =============================================================================
# INFOGRAPHIC HTML GENERATOR
# =============================================================================

def create_infographic_html(data):
    """Generate HTML infographic from data dictionary."""
    html = f"""
<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            max-width: 750px; margin: 0 auto; 
            background: linear-gradient(145deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); 
            padding: 35px; border-radius: 16px; color: #ffffff;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);">
    
    <div style="text-align: center; margin-bottom: 28px; padding-bottom: 20px; 
                border-bottom: 1px solid rgba(255,255,255,0.1);">
        <h1 style="font-size: 22px; margin: 0; color: #60a5fa; font-weight: 600;">
            {safe_get(data, 'TITLE', 'Document Summary')}
        </h1>
        <p style="color: #94a3b8; margin-top: 8px; font-size: 13px;">
            {safe_get(data, 'SUBTITLE', 'Key insights from your document')}
        </p>
    </div>
    
    <div style="display: flex; justify-content: center; gap: 16px; margin: 24px 0; flex-wrap: wrap;">
        <div style="text-align: center; padding: 18px 24px; background: rgba(96,165,250,0.1); 
                    border-radius: 12px; border: 1px solid rgba(96,165,250,0.2); min-width: 130px;">
            <div style="font-size: 26px; font-weight: 700; color: #60a5fa;">
                {safe_get(data, 'STAT1', 'N/A')}
            </div>
            <div style="font-size: 10px; color: #94a3b8; margin-top: 4px; text-transform: uppercase;">
                {safe_get(data, 'STAT1_LABEL', 'Metric')}
            </div>
        </div>
        <div style="text-align: center; padding: 18px 24px; background: rgba(244,114,182,0.1); 
                    border-radius: 12px; border: 1px solid rgba(244,114,182,0.2); min-width: 130px;">
            <div style="font-size: 26px; font-weight: 700; color: #f472b6;">
                {safe_get(data, 'STAT2', 'N/A')}
            </div>
            <div style="font-size: 10px; color: #94a3b8; margin-top: 4px; text-transform: uppercase;">
                {safe_get(data, 'STAT2_LABEL', 'Metric')}
            </div>
        </div>
        <div style="text-align: center; padding: 18px 24px; background: rgba(52,211,153,0.1); 
                    border-radius: 12px; border: 1px solid rgba(52,211,153,0.2); min-width: 130px;">
            <div style="font-size: 26px; font-weight: 700; color: #34d399;">
                {safe_get(data, 'STAT3', 'N/A')}
            </div>
            <div style="font-size: 10px; color: #94a3b8; margin-top: 4px; text-transform: uppercase;">
                {safe_get(data, 'STAT3_LABEL', 'Metric')}
            </div>
        </div>
    </div>
    
    <div style="margin-top: 24px;">
        <h3 style="color: #e2e8f0; margin-bottom: 14px; font-size: 14px; font-weight: 600; 
                   text-transform: uppercase; letter-spacing: 1px;">Key Insights</h3>
        <div style="display: flex; flex-direction: column; gap: 10px;">
            <div style="padding: 14px 16px; background: rgba(96,165,250,0.08); 
                        border-left: 3px solid #60a5fa; border-radius: 0 8px 8px 0; font-size: 13px; color: #e2e8f0;">
                {safe_get(data, 'POINT1', 'Key insight 1')}
            </div>
            <div style="padding: 14px 16px; background: rgba(244,114,182,0.08); 
                        border-left: 3px solid #f472b6; border-radius: 0 8px 8px 0; font-size: 13px; color: #e2e8f0;">
                {safe_get(data, 'POINT2', 'Key insight 2')}
            </div>
            <div style="padding: 14px 16px; background: rgba(52,211,153,0.08); 
                        border-left: 3px solid #34d399; border-radius: 0 8px 8px 0; font-size: 13px; color: #e2e8f0;">
                {safe_get(data, 'POINT3', 'Key insight 3')}
            </div>
            <div style="padding: 14px 16px; background: rgba(251,191,36,0.08); 
                        border-left: 3px solid #fbbf24; border-radius: 0 8px 8px 0; font-size: 13px; color: #e2e8f0;">
                {safe_get(data, 'POINT4', 'Key insight 4')}
            </div>
        </div>
    </div>
    
    <div style="text-align: center; margin-top: 24px; padding-top: 16px; 
                border-top: 1px solid rgba(255,255,255,0.08);">
        <p style="color: #475569; font-size: 10px; margin: 0;">
            RAG Document Assistant | NLP Project
        </p>
    </div>
</div>
"""
    return html


# =============================================================================
# CUSTOM CSS
# =============================================================================

def load_custom_css():
    """Load custom CSS styles."""
    st.markdown("""
    <style>
        .stApp { background-color: #0a0a0f; }
        .main-title { font-size: 2rem; font-weight: 600; color: #f1f5f9; margin-bottom: 4px; }
        .sub-title { font-size: 0.95rem; color: #64748b; margin-bottom: 24px; }
        [data-testid="stSidebar"] { background-color: #0f0f1a; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: transparent; }
        .stTabs [data-baseweb="tab"] { background-color: #1a1a2e; border-radius: 8px; color: #94a3b8; padding: 10px 20px; }
        .stTabs [aria-selected="true"] { background-color: #2563eb; color: white; }
        [data-testid="stChatMessage"] { background-color: #12121a; border-radius: 12px; padding: 16px; margin: 8px 0; }
        .stButton > button { background-color: #2563eb; color: white; border: none; border-radius: 8px; padding: 10px 24px; font-weight: 500; }
        .stButton > button:hover { background-color: #1d4ed8; }
        .stTextInput > div > div > input, .stTextArea > div > div > textarea { background-color: #1a1a2e; border: 1px solid #2a2a3e; color: #f1f5f9; border-radius: 8px; }
        .streamlit-expanderHeader { background-color: #1a1a2e; border-radius: 8px; }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stAlert { background-color: #1a1a2e; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
        st.session_state.embeddings = None
        st.session_state.chat_history = []
        st.session_state.full_text = ""
        st.session_state.documents = []


def clear_all_data():
    """Clear all session state."""
    st.session_state.chunks = []
    st.session_state.embeddings = None
    st.session_state.chat_history = []
    st.session_state.full_text = ""
    st.session_state.documents = []
    for key in ["infographic_html", "doc_summary", "image_prompt"]:
        if key in st.session_state:
            del st.session_state[key]


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar(embedding_model):
    """Render sidebar with upload controls."""
    with st.sidebar:
        st.markdown("### Add Documents")
        
        input_method = st.radio(
            "Input method:",
            ["Upload PDF", "Paste Text"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if input_method == "Upload PDF":
            uploaded_files = st.file_uploader(
                "Choose PDF files",
                type=["pdf"],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
            
            if uploaded_files and st.button("Process Documents", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    for file in uploaded_files:
                        raw_text = extract_text_from_pdf(file.read())
                        cleaned_text = clean_text(raw_text)
                        new_chunks = chunk_text(cleaned_text)
                        
                        st.session_state.full_text += "\n\n" + cleaned_text
                        st.session_state.chunks.extend(new_chunks)
                        st.session_state.documents.append(file.name)
                    
                    st.session_state.embeddings = generate_embeddings(
                        st.session_state.chunks, embedding_model
                    )
                st.success(f"Processed {len(uploaded_files)} file(s)")
        
        else:
            text_input = st.text_area("Enter text:", height=150, placeholder="Paste content here...", label_visibility="collapsed")
            doc_name = st.text_input("Document name:", value="pasted_document", label_visibility="collapsed")
            
            if text_input and st.button("Process Text", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    cleaned_text = clean_text(text_input)
                    new_chunks = chunk_text(cleaned_text)
                    
                    st.session_state.full_text += "\n\n" + cleaned_text
                    st.session_state.chunks.extend(new_chunks)
                    st.session_state.documents.append(doc_name)
                    
                    st.session_state.embeddings = generate_embeddings(
                        st.session_state.chunks, embedding_model
                    )
                st.success("Text processed")
        
        # Document list
        if st.session_state.documents:
            st.markdown("---")
            st.markdown("### Loaded Documents")
            for i, doc in enumerate(st.session_state.documents, 1):
                st.markdown(f"**{i}.** {doc}")
            st.caption(f"{len(st.session_state.chunks)} chunks indexed")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Clear All", use_container_width=True):
                    clear_all_data()
                    st.rerun()
            with col2:
                if st.button("Clear Chat", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()


# =============================================================================
# MAIN CONTENT TABS
# =============================================================================

def render_chat_tab(embedding_model):
    """Render chat interface."""
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                
                if "sources" in message and message["sources"]:
                    with st.expander("View Source Chunks"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"**Chunk {i}** (Relevance: {format_percentage(source['score'])})")
                            st.caption(truncate_text(source["text"]))
                            if i < len(message["sources"]):
                                st.markdown("---")
    
    user_question = st.chat_input("Ask anything about your documents...")
    
    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        with st.spinner("Thinking..."):
            retrieved = retrieve_relevant_chunks(
                user_question,
                st.session_state.chunks,
                st.session_state.embeddings,
                embedding_model,
                TOP_K_CHUNKS
            )
            
            answer = generate_rag_answer(
                user_question, 
                retrieved, 
                st.session_state.chat_history[:-1]
            )
            
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": retrieved
            })
        st.rerun()


def render_infographic_tab():
    """Render infographic tab."""
    st.markdown("### Visual Summary")
    st.caption("Generate an infographic-style summary")
    
    if st.button("Generate Infographic", type="primary"):
        with st.spinner("Creating infographic..."):
            parsed_data = extract_infographic_data(st.session_state.full_text)
            st.session_state.infographic_html = create_infographic_html(parsed_data)
    
    if "infographic_html" in st.session_state:
        st.components.v1.html(st.session_state.infographic_html, height=550, scrolling=False)
        st.markdown("---")
        st.download_button("Download HTML", st.session_state.infographic_html, 
                          file_name="infographic.html", mime="text/html")


def render_image_prompt_tab():
    """Render image prompt tab."""
    st.markdown("### Image Prompt Generator")
    st.caption("Create prompts for AI image generators")
    
    if st.button("Generate Image Prompt", type="primary"):
        with st.spinner("Analyzing document..."):
            summary = generate_document_summary(st.session_state.full_text)
            st.session_state.doc_summary = summary
            st.session_state.image_prompt = generate_image_prompt(summary)
    
    if "doc_summary" in st.session_state:
        st.markdown("#### Document Summary")
        st.info(st.session_state.doc_summary)
    
    if "image_prompt" in st.session_state:
        st.markdown("#### Generated Prompt")
        st.success(st.session_state.image_prompt)
        st.code(st.session_state.image_prompt, language=None)


def render_empty_state():
    """Render empty state."""
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px; color: #64748b;">
        <h3 style="color: #94a3b8; margin-bottom: 16px;">No documents loaded</h3>
        <p>Upload a PDF or paste text using the sidebar to get started</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("Example Questions You Can Ask"):
        st.markdown("""
        - "Give me a detailed summary"
        - "Create a learning plan for this topic"
        - "Explain [concept] in simple terms"
        - "What are the key takeaways?"
        """)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main application."""
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide", initial_sidebar_state="expanded")
    
    load_custom_css()
    init_session_state()
    
    st.markdown(f'<p class="main-title">{APP_ICON} {APP_TITLE}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-title">{APP_DESCRIPTION}</p>', unsafe_allow_html=True)
    
    if not check_api_key():
        st.error("OpenAI API key not found. Add OPENAI_API_KEY to your .env file.")
        st.code("OPENAI_API_KEY=sk-your-key-here", language="text")
        st.stop()
    
    @st.cache_resource
    def get_embedding_model():
        return load_embedding_model()
    
    embedding_model = get_embedding_model()
    render_sidebar(embedding_model)
    
    if st.session_state.chunks:
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Infographic", "🎨 Image Prompt"])
        with tab1:
            render_chat_tab(embedding_model)
        with tab2:
            render_infographic_tab()
        with tab3:
            render_image_prompt_tab()
    else:
        render_empty_state()


if __name__ == "__main__":
    main()