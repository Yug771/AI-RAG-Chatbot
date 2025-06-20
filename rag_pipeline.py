"""
RAG Pipeline using LlamaIndex v0.12+ 
Modern implementation using Settings object and latest APIs
"""

import os
import tempfile
import shutil
import logging
from typing import List, Optional, Any, Dict
from pathlib import Path

import streamlit as st

# Load environment variables for local development
try:
    from dotenv import load_dotenv
    # Try to load .env with UTF-8 encoding first, fallback to other encodings
    try:
        load_dotenv(encoding='utf-8')
    except UnicodeDecodeError:
        try:
            load_dotenv(encoding='latin-1')
        except:
            # If all encoding attempts fail, continue without .env
            print("Warning: Could not load .env file due to encoding issues. Using system environment variables only.")
            pass
except ImportError:
    pass  # dotenv not available, skip

# API Key Configuration for Streamlit Cloud and Local Development
def get_api_key(key_name):
    """Get API key from Streamlit secrets or environment variables"""
    try:
        # Try Streamlit secrets first (for cloud deployment)
        return st.secrets.get(key_name)
    except:
        # Fallback to environment variables (for local development)
        return os.getenv(key_name)

OPENAI_API_KEY = get_api_key("OPENAI_API_KEY")

# Setup logger
logger = logging.getLogger(__name__)

# LlamaIndex Core
from llama_index.core import (
    VectorStoreIndex, 
    Document, 
    StorageContext,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.readers.file import PDFReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

import faiss
import numpy as np

class RAGPipeline:
    """Modern RAG Pipeline using LlamaIndex v0.12+"""
    
    def __init__(self, 
                 chunk_size: int = 1024,
                 chunk_overlap: int = 200,
                 similarity_top_k: int = 5,
                 embedding_model: str = "openai"):
        
        logger.info(f"Initializing RAG Pipeline - chunk_size: {chunk_size}, overlap: {chunk_overlap}, top_k: {similarity_top_k}")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap 
        self.similarity_top_k = similarity_top_k
        self.embedding_model_name = "openai"  # Force OpenAI
        
        # Initialize components
        self.documents: List[Document] = []
        self.index: Optional[VectorStoreIndex] = None
        self.vector_store: Optional[FaissVectorStore] = None
        self.storage_context: Optional[StorageContext] = None
        
        # Setup embedding model
        self._setup_embedding_model()
        
        # Setup node parser
        self._setup_node_parser()
        
        logger.info("RAG Pipeline initialized successfully")
        
    def _setup_embedding_model(self):
        """Setup embedding model using Settings - OpenAI only"""
        logger.info("Setting up OpenAI embedding model: text-embedding-3-small")
        
        # Force OpenAI embeddings only
        embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=OPENAI_API_KEY
        )
        
        # Set global embedding model using Settings
        Settings.embed_model = embed_model
        logger.info("OpenAI embedding model configured successfully")
        
    def _setup_node_parser(self):
        """Setup node parser using Settings"""
        node_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=" "
        )
        
        # Set global node parser using Settings
        Settings.node_parser = node_parser
        
    def load_documents_from_upload(self, uploaded_files) -> bool:
        """Load documents from Streamlit uploaded files"""
        logger.info(f"Loading {len(uploaded_files)} documents from upload")
        
        try:
            self.documents = []
            
            for uploaded_file in uploaded_files:
                logger.info(f"Processing file: {uploaded_file.name}")
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # Use LlamaIndex PDFReader
                    pdf_reader = PDFReader()
                    docs = pdf_reader.load_data(file=Path(tmp_file_path))
                    
                    # Add metadata
                    for doc in docs:
                        doc.metadata["filename"] = uploaded_file.name
                        doc.metadata["file_size"] = len(uploaded_file.getvalue())
                    
                    self.documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} documents from {uploaded_file.name}")
                    
                finally:
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
            
            logger.info(f"Total documents loaded: {len(self.documents)}")
            return len(self.documents) > 0
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}", exc_info=True)
            st.error(f"Error loading documents: {str(e)}")
            return False
    
    def create_vector_index(self) -> bool:
        """Create vector index from loaded documents"""
        logger.info(f"Creating vector index from {len(self.documents)} documents")
        
        try:
            if not self.documents:
                logger.error("No documents loaded for indexing")
                st.error("No documents loaded. Please upload documents first.")
                return False
            
            # Create FAISS vector store with OpenAI text-embedding-3-small dimensions
            embedding_dim = 1536  # Fixed for OpenAI text-embedding-3-small
            logger.info(f"Creating FAISS index with {embedding_dim} dimensions")
            faiss_index = faiss.IndexFlatL2(embedding_dim)
            
            self.vector_store = FaissVectorStore(faiss_index=faiss_index)
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            # Create index using Settings (no need to pass embed_model explicitly)
            logger.info("Starting vector index creation with LlamaIndex")
            with st.spinner("Creating vector index... This may take a few minutes."):
                self.index = VectorStoreIndex.from_documents(
                    documents=self.documents,
                    storage_context=self.storage_context,
                    show_progress=True
                )
            
            logger.info("Vector index created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating vector index: {str(e)}", exc_info=True)
            st.error(f"Error creating vector index: {str(e)}")
            return False
    
    def get_query_engine(self, llm_model, 
                        similarity_top_k: Optional[int] = None) -> Optional[BaseQueryEngine]:
        """Create query engine with specified LLM"""
        try:
            if not self.index:
                st.error("No index available. Please create index first.")
                return None
            
            # Set LLM in Settings
            Settings.llm = llm_model
            
            # Create query engine
            query_engine = self.index.as_query_engine(
                similarity_top_k=similarity_top_k or self.similarity_top_k,
                response_mode="compact"
            )
            
            return query_engine
            
        except Exception as e:
            st.error(f"Error creating query engine: {str(e)}")
            return None
    
    def query(self, 
              question: str, 
              llm_model,
              similarity_top_k: Optional[int] = None) -> Dict[str, Any]:
        """Query the RAG system"""
        logger.info(f"Executing query: {question[:100]}...")
        
        try:
            query_engine = self.get_query_engine(llm_model, similarity_top_k)
            if not query_engine:
                logger.error("Failed to create query engine")
                return {"error": "Failed to create query engine"}
            
            # Execute query
            logger.info("Generating response with LLM")
            with st.spinner("Generating response..."):
                response = query_engine.query(question)
            
            # Extract source information
            source_info = []
            if hasattr(response, 'source_nodes'):
                logger.info(f"Found {len(response.source_nodes)} source nodes")
                for node in response.source_nodes:
                    source_info.append({
                        "content": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                        "score": node.score if hasattr(node, 'score') else None,
                        "metadata": node.metadata
                    })
            
            logger.info("Query completed successfully")
            return {
                "response": str(response),
                "source_nodes": source_info,
                "question": question
            }
            
        except Exception as e:
            logger.error(f"Error during query: {str(e)}", exc_info=True)
            return {"error": f"Error during query: {str(e)}"}
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded documents"""
        if not self.documents:
            return {"total_docs": 0}
        
        total_chars = sum(len(doc.text) for doc in self.documents)
        filenames = list(set(doc.metadata.get("filename", "Unknown") 
                           for doc in self.documents))
        
        return {
            "total_docs": len(self.documents),
            "total_characters": total_chars,
            "filenames": filenames,
            "estimated_tokens": total_chars // 4,  # Rough estimate
        }
    
    def reset(self):
        """Reset the pipeline"""
        self.documents = []
        self.index = None
        self.vector_store = None
        self.storage_context = None
    
    def save_index(self, persist_dir: str = "./storage"):
        """Save index to disk"""
        try:
            if self.index and self.storage_context:
                self.storage_context.persist(persist_dir=persist_dir)
                return True
            return False
        except Exception as e:
            st.error(f"Error saving index: {str(e)}")
            return False
    
    def load_index(self, persist_dir: str = "./storage") -> bool:
        """Load index from disk"""
        try:
            if os.path.exists(persist_dir):
                # Setup vector store with OpenAI dimensions
                embedding_dim = 1536  # Fixed for OpenAI text-embedding-3-small
                faiss_index = faiss.IndexFlatL2(embedding_dim)
                self.vector_store = FaissVectorStore(faiss_index=faiss_index)
                
                # Load storage context
                self.storage_context = StorageContext.from_defaults(
                    vector_store=self.vector_store,
                    persist_dir=persist_dir
                )
                
                # Load index
                self.index = VectorStoreIndex.from_storage(
                    storage_context=self.storage_context
                )
                
                return True
            return False
        except Exception as e:
            st.error(f"Error loading index: {str(e)}")
            return False 