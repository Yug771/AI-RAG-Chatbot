"""
RAG-based AI Agent Web App - DYNAMIC STEP-BY-STEP VERSION
Built with Streamlit and LlamaIndex v0.12+
"""

import streamlit as st
import os
import logging
from typing import Dict, Any, List
import time
from datetime import datetime

# Import our custom modules
from rag_pipeline import RAGPipeline
from llm_wrapper import llm_wrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('rag_app.log', mode='a')  # File output
    ]
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAG AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f2937;
        margin: 1rem 0;
    }
    .model-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .model-info h3 {
        color: white !important;
        margin-bottom: 0.5rem;
    }
    .model-info p {
        color: #f0f0f0 !important;
        margin: 0.3rem 0;
    }
    .step-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #e5e7eb;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    .step-completed {
        background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
        border-left-color: #22c55e;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.15);
    }
    .step-active {
        background: linear-gradient(135deg, #fefce8 0%, #fffbeb 100%);
        border-left-color: #f59e0b;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.15);
    }
    .comparison-container {
        background-color: #f3f4f6;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #e5e7eb;
        margin: 1rem 0;
    }
    .model-selection-box {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #d1d5db;
        margin: 0.5rem 0;
    }
    .model-selection-box:hover {
        border-color: #3b82f6;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.1);
    }
    .source-box {
        background: linear-gradient(135deg, #f3f4f6 0%, #f9fafb 100%);
        color: #374151;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #d1d5db;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .source-box strong {
        color: #1f2937 !important;
    }
    .source-box small {
        color: #6b7280 !important;
    }
    .error-box {
        background-color: #fef2f2;
        color: #dc2626;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #fecaca;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #f0fdf4;
        color: #166534;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bbf7d0;
        margin: 1rem 0;
    }
    .config-box {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #f59e0b;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(245, 158, 11, 0.2);
    }
    .config-box strong {
        color: white !important;
    }
    .status-badge {
        display: inline-block;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 0.9rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(16, 185, 129, 0.25);
        transform: translateY(0);
        transition: all 0.3s ease;
    }
    .status-badge:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(16, 185, 129, 0.35);
    }
    .step-number {
        background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
        color: white;
        width: 2.2rem;
        height: 2.2rem;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 0.75rem;
        box-shadow: 0 2px 4px rgba(107, 114, 128, 0.2);
        transition: all 0.3s ease;
    }
    .step-number.completed {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        box-shadow: 0 4px 8px rgba(16, 185, 129, 0.25);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 4px 8px rgba(16, 185, 129, 0.25); }
        50% { box-shadow: 0 6px 12px rgba(16, 185, 129, 0.4); }
        100% { box-shadow: 0 4px 8px rgba(16, 185, 129, 0.25); }
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(102, 126, 234, 0.2);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid #e5e7eb;
        transition: all 0.3s ease;
    }
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    logger.info("Initializing session state")
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'rag_configured' not in st.session_state:
        st.session_state.rag_configured = False
    if 'index_created' not in st.session_state:
        st.session_state.index_created = False
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = None
    if 'chunk_size' not in st.session_state:
        st.session_state.chunk_size = 1024
    if 'chunk_overlap' not in st.session_state:
        st.session_state.chunk_overlap = 200
    if 'similarity_top_k' not in st.session_state:
        st.session_state.similarity_top_k = 5
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None
    if 'comparison_models' not in st.session_state:
        st.session_state.comparison_models = []
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

def check_api_keys():
    """Check if required API keys are configured"""
    logger.info("Checking API keys configuration")
    
    # Import API keys from modules
    from llm_wrapper import OPENAI_API_KEY, ANTHROPIC_API_KEY
    
    if not OPENAI_API_KEY and not ANTHROPIC_API_KEY:
        logger.error("No API keys configured")
        st.error("⚠️ **No API keys configured!**")
        st.info("""
        **For Local Development:**
        - Create a `.env` file in the project root
        - Add: `OPENAI_API_KEY=your_openai_key_here`
        - Add: `ANTHROPIC_API_KEY=your_anthropic_key_here`
        
        **For Streamlit Cloud:**
        - Go to your app settings
        - Add secrets in the Secrets section:
        ```
        OPENAI_API_KEY = "your_openai_key_here"
        ANTHROPIC_API_KEY = "your_anthropic_key_here"
        ```
        """)
        st.stop()
    
    if not OPENAI_API_KEY:
        logger.warning("OpenAI API key not configured - OpenAI models and embeddings will not be available")
        st.error("⚠️ **OpenAI API key required for embeddings!**")
        st.info("Please configure your OpenAI API key in `.env` file or Streamlit Cloud secrets.")
        st.stop()
    else:
        logger.info("OpenAI API key configured")
    
    if not ANTHROPIC_API_KEY:
        logger.warning("Anthropic API key not configured - Anthropic models will not be available")
    else:
        logger.info("Anthropic API key configured")

def render_step_1_upload():
    """Step 1: Document Upload"""
    step_number_class = "completed" if st.session_state.documents_loaded else ""
    step_number = "✓" if st.session_state.documents_loaded else "1"
    
    st.markdown(f'''
    <div class="section-header">
        <span class="step-number {step_number_class}">{step_number}</span>
        📄 Upload Documents
    </div>
    ''', unsafe_allow_html=True)
    
    step_class = "step-completed" if st.session_state.documents_loaded else "step-active"
    
    with st.container():
        st.markdown(f'<div class="step-container {step_class}">', unsafe_allow_html=True)
        
        if st.session_state.documents_loaded:
            st.markdown('<span class="status-badge">✅ COMPLETED</span>', unsafe_allow_html=True)
            st.write(f"**Files uploaded:** {len(st.session_state.uploaded_files)} files")
            for file in st.session_state.uploaded_files:
                file_size_mb = len(file.getvalue()) / (1024 * 1024)
                st.write(f"• {file.name} ({file_size_mb:.1f} MB)")
            
            if st.button("🔄 Upload Different Documents"):
                st.session_state.documents_loaded = False
                st.session_state.rag_configured = False
                st.session_state.index_created = False
                st.session_state.uploaded_files = None
                st.session_state.rag_pipeline = None
                st.rerun()
        else:
            st.write("**Select PDF documents to create your RAG knowledge base:**")
            uploaded_files = st.file_uploader(
                "Choose PDF files",
                type=['pdf'],
                accept_multiple_files=True,
                help="Upload one or more PDF files to create your knowledge base",
                label_visibility="collapsed"
            )
            
            if uploaded_files:
                total_size = sum(len(file.getvalue()) / (1024 * 1024) for file in uploaded_files)
                st.write(f"**{len(uploaded_files)} file(s) selected (Total: {total_size:.1f} MB):**")
                for file in uploaded_files:
                    file_size_mb = len(file.getvalue()) / (1024 * 1024)
                    st.write(f"• {file.name} ({file_size_mb:.1f} MB)")
                
                if st.button("📂 Load Documents", type="primary"):
                    with st.spinner("Loading documents..."):
                        st.session_state.uploaded_files = uploaded_files
                        st.session_state.documents_loaded = True
                        logger.info(f"Documents loaded: {[f.name for f in uploaded_files]}")
                        st.rerun()
            else:
                st.info("Please select PDF documents to get started.")
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_step_2_configure():
    """Step 2: Configure RAG Settings"""
    if not st.session_state.documents_loaded:
        return
    
    step_number_class = "completed" if st.session_state.rag_configured else ""
    step_number = "✓" if st.session_state.rag_configured else "2"
    
    st.markdown(f'''
    <div class="section-header">
        <span class="step-number {step_number_class}">{step_number}</span>
        ⚙️ Configure RAG Settings
    </div>
    ''', unsafe_allow_html=True)
    
    step_class = "step-completed" if st.session_state.rag_configured else "step-active"
    
    with st.container():
        st.markdown(f'<div class="step-container {step_class}">', unsafe_allow_html=True)
        
        if st.session_state.rag_configured:
            st.markdown('<span class="status-badge">✅ CONFIGURED</span>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="config-box">
                <strong>Current Configuration:</strong><br>
                • Chunk Size: {st.session_state.chunk_size} characters<br>
                • Chunk Overlap: {st.session_state.chunk_overlap} characters<br>
                • Top-K Results: {st.session_state.similarity_top_k} chunks<br>
                • Embedding Model: OpenAI text-embedding-3-small (1536 dimensions)
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔄 Reconfigure Settings"):
                st.session_state.rag_configured = False
                st.session_state.index_created = False
                st.session_state.rag_pipeline = None
                st.rerun()
        else:
            st.write("**Configure how your documents will be processed for optimal RAG performance:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                chunk_size = st.slider(
                    "**Chunk Size**",
                    min_value=512,
                    max_value=2048,
                    value=st.session_state.chunk_size,
                    step=128,
                    help="Size of each text chunk (larger = more context, slower processing)"
                )
            
            with col2:
                chunk_overlap = st.slider(
                    "**Chunk Overlap**",
                    min_value=50,
                    max_value=500,
                    value=st.session_state.chunk_overlap,
                    step=50,
                    help="Overlap between chunks (higher = better context continuity)"
                )
            
            with col3:
                similarity_top_k = st.slider(
                    "**Top-K Results**",
                    min_value=3,
                    max_value=10,
                    value=st.session_state.similarity_top_k,
                    help="Number of similar chunks to retrieve for each query"
                )
            
            st.info("🔧 **Embedding Model:** OpenAI text-embedding-3-small (1536 dimensions)")
            
            # Estimate processing time and tokens
            if st.session_state.uploaded_files:
                total_chars = sum(len(file.getvalue()) for file in st.session_state.uploaded_files)
                estimated_chunks = total_chars // chunk_size
                estimated_tokens = total_chars // 4
                
                st.write(f"**Estimated Processing:**")
                st.write(f"• Chunks: ~{estimated_chunks:,}")
                st.write(f"• Tokens: ~{estimated_tokens:,}")
                st.write(f"• Processing time: ~{max(1, estimated_chunks // 100)} minutes")
            
            if st.button("✅ Confirm Settings & Create Pipeline", type="primary"):
                st.session_state.chunk_size = chunk_size
                st.session_state.chunk_overlap = chunk_overlap
                st.session_state.similarity_top_k = similarity_top_k
                st.session_state.rag_configured = True
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_step_3_generate_embeddings():
    """Step 3: Generate Embeddings"""
    if not st.session_state.rag_configured:
        return
    
    step_number_class = "completed" if st.session_state.index_created else ""
    step_number = "✓" if st.session_state.index_created else "3"
    
    st.markdown(f'''
    <div class="section-header">
        <span class="step-number {step_number_class}">{step_number}</span>
        🔧 Generate Vector Embeddings
    </div>
    ''', unsafe_allow_html=True)
    
    step_class = "step-completed" if st.session_state.index_created else "step-active"
    
    with st.container():
        st.markdown(f'<div class="step-container {step_class}">', unsafe_allow_html=True)
        
        if st.session_state.index_created:
            st.markdown('<span class="status-badge">✅ EMBEDDINGS READY</span>', unsafe_allow_html=True)
            
            # Show document stats
            if st.session_state.rag_pipeline:
                stats = st.session_state.rag_pipeline.get_document_stats()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Documents", stats['total_docs'])
                with col2:
                    st.metric("Characters", f"{stats['total_characters']:,}")
                with col3:
                    st.metric("Est. Tokens", f"{stats['estimated_tokens']:,}")
            
            if st.button("🔄 Regenerate Embeddings"):
                st.session_state.index_created = False
                st.session_state.rag_pipeline = None
                st.rerun()
        else:
            st.write("**Generate vector embeddings for your documents using OpenAI's embedding model.**")
            st.write("This process will convert your text into numerical vectors for semantic search.")
            
            if st.button("🚀 Generate Embeddings", type="primary"):
                generate_embeddings()
        
        st.markdown('</div>', unsafe_allow_html=True)

def generate_embeddings():
    """Generate embeddings with configured settings"""
    logger.info("Starting embedding generation")
    
    try:
        # Create RAG pipeline with configured settings
        rag_pipeline = RAGPipeline(
            chunk_size=st.session_state.chunk_size,
            chunk_overlap=st.session_state.chunk_overlap,
            similarity_top_k=st.session_state.similarity_top_k,
            embedding_model="openai"
        )
        
        # Load documents
        with st.spinner("Loading and processing documents..."):
            if not rag_pipeline.load_documents_from_upload(st.session_state.uploaded_files):
                st.error("❌ Failed to load documents")
                return False
        
        st.success("✅ Documents loaded successfully!")
        
        # Create vector index
        with st.spinner("Generating vector embeddings... This may take a few minutes."):
            if not rag_pipeline.create_vector_index():
                st.error("❌ Failed to create vector index")
                return False
        
        # Store in session state
        st.session_state.rag_pipeline = rag_pipeline
        st.session_state.index_created = True
        
        logger.info("Embedding generation completed successfully")
        st.success("✅ Vector embeddings generated successfully!")
        st.rerun()
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
        st.error(f"❌ Error generating embeddings: {str(e)}")
        return False

def render_step_4_query():
    """Step 4: Query Interface"""
    if not st.session_state.index_created:
        return
    
    st.markdown(f'''
    <div class="section-header">
        <span class="step-number completed">4</span>
        💬 Ask Questions
    </div>
    ''', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="step-container step-active">', unsafe_allow_html=True)
        
        st.markdown('<span class="status-badge">🎯 READY TO QUERY</span>', unsafe_allow_html=True)
        st.write("**Your RAG system is ready! Ask questions about your uploaded documents.**")
        
        # Model selection and comparison mode
        col1, col2 = st.columns([2, 1])
        
        with col1:
            comparison_mode = st.checkbox("🔍 **Comparison Mode**", help="Compare responses from 2 selected models")
        
        with col2:
            if st.button("🗑️ Clear History"):
                st.session_state.query_history = []
                st.rerun()
        
        if comparison_mode:
            render_comparison_interface()
        else:
            render_single_model_interface()
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_single_model_interface():
    """Single model interface"""
    st.subheader("🤖 Single Model Mode")
    
    available_models = llm_wrapper.get_available_models()
    
    # Model selection
    model_options = list(available_models.keys())
    model_labels = [f"{available_models[key]} ({llm_wrapper.get_model_info(key).get('provider', 'Unknown')})" 
                   for key in model_options]
    
    selected_idx = st.selectbox(
        "Select AI Model:",
        range(len(model_options)),
        format_func=lambda x: model_labels[x],
        index=0 if not st.session_state.selected_model else model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
    )
    
    selected_model = model_options[selected_idx]
    st.session_state.selected_model = selected_model
    
    # Display model info
    model_info = llm_wrapper.get_model_info(selected_model)
    st.markdown(f"""
    <div class="model-info">
        <h3>🤖 {available_models[selected_model]}</h3>
        <p><strong>Provider:</strong> {model_info.get('provider', 'Unknown')}</p>
        <p><strong>Context Length:</strong> {model_info.get('context_length', 'Unknown'):,} tokens</p>
        <p><strong>Description:</strong> {model_info.get('description', 'No description available')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Query interface
    question = st.text_area(
        "Enter your question:",
        placeholder="Ask anything about your uploaded documents...",
        height=100
    )
    
    if st.button("🔍 Ask Question", type="primary", disabled=not question.strip()):
        llm_model = llm_wrapper.get_model(selected_model)
        if llm_model:
            render_single_result(st.session_state.rag_pipeline, question, llm_model, selected_model)

def render_comparison_interface():
    """Comparison mode interface"""
    st.subheader("🔍 Comparison Mode")
    
    available_models = llm_wrapper.get_available_models()
    model_options = list(available_models.keys())
    
    if len(model_options) < 2:
        st.warning("Need at least 2 models for comparison mode.")
        return
    
    st.write("Select exactly 2 models to compare:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="model-selection-box">', unsafe_allow_html=True)
        st.write("**Model 1:**")
        model1_idx = st.selectbox(
            "First model:",
            range(len(model_options)),
            format_func=lambda x: f"{available_models[model_options[x]]} ({llm_wrapper.get_model_info(model_options[x]).get('provider', 'Unknown')})",
            key="model1"
        )
        model1 = model_options[model1_idx]
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="model-selection-box">', unsafe_allow_html=True)
        st.write("**Model 2:**")
        model2_options = [m for m in model_options if m != model1]
        if model2_options:
            model2_idx = st.selectbox(
                "Second model:",
                range(len(model2_options)),
                format_func=lambda x: f"{available_models[model2_options[x]]} ({llm_wrapper.get_model_info(model2_options[x]).get('provider', 'Unknown')})",
                key="model2"
            )
            model2 = model2_options[model2_idx]
        else:
            model2 = None
        st.markdown('</div>', unsafe_allow_html=True)
    
    if model2:
        st.session_state.comparison_models = [model1, model2]
        
        # Query interface
        question = st.text_area(
            "Enter your question:",
            placeholder="Ask anything about your uploaded documents...",
            height=100,
            key="comparison_question"
        )
        
        if st.button("🔍 Compare Responses", type="primary", disabled=not question.strip()):
            render_comparison_results(st.session_state.rag_pipeline, question, [model1, model2])

def render_single_result(rag_pipeline, question, llm_model, model_name):
    """Render single model result"""
    logger.info(f"Generating response using model: {model_name}")
    start_time = time.time()
    
    try:
        result = rag_pipeline.query(question, llm_model, st.session_state.similarity_top_k)
        end_time = time.time()
        response_time = end_time - start_time
        
        logger.info(f"Response generated in {response_time:.2f} seconds")
        
        if "error" in result:
            logger.error(f"Query error: {result['error']}")
            st.markdown(f'<div class="error-box">❌ {result["error"]}</div>', unsafe_allow_html=True)
            return
        
        # Add to query history
        st.session_state.query_history.append({
            "timestamp": datetime.now(),
            "question": question,
            "model": model_name,
            "response": result["response"],
            "response_time": response_time
        })
        
        # Display result
        st.markdown("### 💡 Response")
        st.write(result["response"])
        
        # Response metadata
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"⏱️ Response time: {response_time:.2f} seconds")
        with col2:
            st.caption(f"🤖 Model: {llm_wrapper.get_available_models()[model_name]}")
        
        # Source information
        if result.get("source_nodes"):
            logger.info(f"Found {len(result['source_nodes'])} source nodes")
            with st.expander(f"📚 Sources ({len(result['source_nodes'])} chunks)"):
                for i, source in enumerate(result["source_nodes"], 1):
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>Source {i}:</strong><br>
                        {source['content']}<br>
                        <small><strong>File:</strong> {source['metadata'].get('filename', 'Unknown')}</small>
                    </div>
                    """, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Error in render_single_result: {str(e)}", exc_info=True)
        st.error(f"Error generating response: {str(e)}")

def render_comparison_results(rag_pipeline, question, selected_models):
    """Render comparison results from selected models"""
    logger.info(f"Starting comparison mode with models: {selected_models}")
    
    st.markdown("### 🔍 Model Comparison Results")
    
    results = {}
    
    # Get responses from both models
    col1, col2 = st.columns(2)
    
    for i, model_id in enumerate(selected_models):
        model_name = llm_wrapper.get_available_models()[model_id]
        logger.info(f"Getting response from {model_name}")
        
        llm_model = llm_wrapper.get_model(model_id)
        if llm_model:
            with (col1 if i == 0 else col2):
                with st.spinner(f"Getting response from {model_name}..."):
                    start_time = time.time()
                    try:
                        result = rag_pipeline.query(question, llm_model, st.session_state.similarity_top_k)
                        end_time = time.time()
                        
                        if "error" not in result:
                            results[model_id] = {
                                "model_name": model_name,
                                "response": result["response"],
                                "response_time": end_time - start_time,
                                "model_info": llm_wrapper.get_model_info(model_id)
                            }
                            logger.info(f"Response received from {model_name}")
                        else:
                            logger.error(f"Error from {model_name}: {result['error']}")
                            st.error(f"Error from {model_name}: {result['error']}")
                    except Exception as e:
                        logger.error(f"Exception with {model_name}: {str(e)}")
                        st.error(f"Error with {model_name}: {str(e)}")
    
    # Display comparison in two columns
    if len(results) == 2:
        logger.info("Displaying comparison results")
        
        col1, col2 = st.columns(2)
        
        for i, (model_id, data) in enumerate(results.items()):
            with (col1 if i == 0 else col2):
                st.markdown(f"""
                <div class="model-info">
                    <h3>🤖 {data['model_name']}</h3>
                    <p><strong>Provider:</strong> {data['model_info']['provider']}</p>
                    <p><strong>Context Length:</strong> {data['model_info'].get('context_length', 'Unknown'):,} tokens</p>
                    <p><strong>Response Time:</strong> {data['response_time']:.2f} seconds</p>
                    <p><strong>Description:</strong> {data['model_info'].get('description', 'No description available')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 💡 Response")
                st.write(data["response"])
                st.markdown("---")

def render_query_history():
    """Render query history"""
    if st.session_state.query_history:
        logger.debug(f"Rendering query history with {len(st.session_state.query_history)} entries")
        st.markdown('<div class="section-header">📝 Query History</div>', unsafe_allow_html=True)
        
        with st.expander(f"💬 Previous Queries ({len(st.session_state.query_history)})", expanded=False):
            for i, query in enumerate(reversed(st.session_state.query_history[-10:])):  # Show last 10
                st.markdown(f"""
                <div class="source-box">
                    <strong>🕒 {query['timestamp'].strftime('%H:%M:%S')}</strong> - <span style="color: #667eea;">🤖 {query['model']}</span><br>
                    <strong>❓ Question:</strong> {query['question'][:100]}{'...' if len(query['question']) > 100 else ''}<br>
                    <strong>💡 Answer:</strong> {query['response'][:200]}{'...' if len(query['response']) > 200 else ''}<br>
                    <small>⏱️ Response time: {query.get('response_time', 0):.2f}s</small>
                </div>
                """, unsafe_allow_html=True)

def main():
    """Main application function"""
    logger.info("=== RAG AI Agent Starting ===")
    
    # Initialize
    initialize_session_state()
    check_api_keys()
    
    # Header
    st.markdown('<div class="main-header">🚀 Smart RAG Assistant - Ace Technology</div>', unsafe_allow_html=True)
    
    # Progress indicator in an info box
    progress_steps = ["📄 Upload", "⚙️ Configure", "🔧 Embeddings", "💬 Query"]
    current_step = 0
    if st.session_state.documents_loaded:
        current_step = 1
    if st.session_state.rag_configured:
        current_step = 2
    if st.session_state.index_created:
        current_step = 3
    
    # Create progress status
    progress_status = []
    for i, step in enumerate(progress_steps):
        if i <= current_step:
            progress_status.append(f"**{step}** ✅")
        else:
            progress_status.append(f"{step}")
    
    st.info(f"**🚀 Smart RAG Pipeline Progress:** {' → '.join(progress_status)}")
    st.markdown("---")
    
    # Main workflow
    render_step_1_upload()
    render_step_2_configure()
    render_step_3_generate_embeddings()
    render_step_4_query()
    
    # Query history (full width)
    render_query_history()
    
    # Footer
    st.markdown("---")
    st.markdown("**🚀 Smart RAG Assistant - Ace Technology | Built with ❤️ using Streamlit + LlamaIndex v0.12+ | Embeddings: OpenAI text-embedding-3-small**")
    
    logger.debug("Main function completed")

if __name__ == "__main__":
    main() 