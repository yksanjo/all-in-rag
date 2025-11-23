"""
Streamlit UI for Enterprise RAG system.
"""

import sys
from pathlib import Path
import streamlit as st
import requests

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config

# Page config
st.set_page_config(
    page_title="Enterprise Offline RAG",
    page_icon="🔒",
    layout="wide"
)

# API URL
API_URL = f"http://{config.api.host}:{config.api.port}"


def check_api_health():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    """Main Streamlit app."""
    st.title("🔒 Enterprise Offline RAG System")
    st.markdown("**Fully offline, private RAG system for enterprise use**")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # API health check
        if check_api_health():
            st.success("✅ API Connected")
        else:
            st.error("❌ API Not Available")
            st.info(f"Make sure the API is running at {API_URL}")
            return
        
        # Get stats
        try:
            stats_response = requests.get(f"{API_URL}/stats", timeout=5)
            if stats_response.status_code == 200:
                stats = stats_response.json()
                st.subheader("System Stats")
                st.json(stats)
        except:
            pass
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Query", "Upload Documents", "About"])
    
    with tab1:
        st.header("Query the RAG System")
        
        query = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="Ask a question about your documents..."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Number of documents to retrieve", 1, 20, 5)
            use_reranking = st.checkbox("Use reranking", value=False)
        
        with col2:
            max_tokens = st.number_input("Max tokens", 100, 4096, 2048)
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        
        if st.button("Query", type="primary"):
            if not query:
                st.warning("Please enter a question.")
                return
            
            with st.spinner("Processing query..."):
                try:
                    response = requests.post(
                        f"{API_URL}/query",
                        json={
                            "query": query,
                            "top_k": top_k,
                            "use_reranking": use_reranking,
                            "max_tokens": max_tokens,
                            "temperature": temperature
                        },
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.subheader("Answer")
                        st.write(result["answer"])
                        
                        st.subheader("Sources")
                        if result["sources"]:
                            for i, source in enumerate(result["sources"], 1):
                                with st.expander(f"Source {i} (Score: {source.get('score', 0):.3f})"):
                                    st.json(source.get("metadata", {}))
                        else:
                            st.info("No sources found.")
                    else:
                        st.error(f"Error: {response.text}")
                
                except Exception as e:
                    st.error(f"Error querying API: {e}")
    
    with tab2:
        st.header("Upload Documents")
        st.markdown("Upload documents to index them in the RAG system.")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt", "md", "jpg", "jpeg", "png"]
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} file(s)")
            
            if st.button("Upload and Index", type="primary"):
                with st.spinner("Uploading and indexing documents..."):
                    try:
                        files = [("files", (f.name, f.read(), f.type)) for f in uploaded_files]
                        
                        response = requests.post(
                            f"{API_URL}/upload",
                            files=files,
                            timeout=300
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"✅ {result['message']}")
                            st.info(f"Indexed {result['num_documents']} document chunks")
                        else:
                            st.error(f"Error: {response.text}")
                    
                    except Exception as e:
                        st.error(f"Error uploading files: {e}")
    
    with tab3:
        st.header("About")
        st.markdown("""
        ## Enterprise Offline RAG System
        
        A fully offline, enterprise-grade Retrieval-Augmented Generation framework.
        
        ### Features
        - ✅ **100% Offline** - No external API calls
        - ✅ **Private** - All data stays within your infrastructure
        - ✅ **Local LLMs** - Supports Llama, Qwen, Mistral (GGUF format)
        - ✅ **Local Embeddings** - BGE-M3, GTE-Large, OpenCLIP
        - ✅ **Local Vector DB** - FAISS or Qdrant
        - ✅ **Text + Image RAG** - Multimodal document retrieval
        - ✅ **Modular Architecture** - Easy to customize and extend
        
        ### Configuration
        - LLM Model: `{llm_model}`
        - Embedding Model: `{embedding_model}`
        - Vector Store: `{vector_store}`
        """.format(
            llm_model=config.llm.model_type,
            embedding_model=config.embedding.model_type,
            vector_store=config.vectorstore.store_type
        ))


if __name__ == "__main__":
    main()

