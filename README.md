# 🤖 RAG-based AI Agent Web App

A modern, user-friendly web application that combines **Retrieval-Augmented Generation (RAG)** with multiple AI models to answer questions about your documents. Built with **Streamlit** and **LlamaIndex v0.12+**.

## ✨ Features

- 📄 **PDF Document Upload**: Upload multiple PDF files to create your knowledge base
- 🤖 **Multiple AI Models**: Choose between GPT-4, Claude 3, and other state-of-the-art models
- 🔍 **Intelligent Search**: Advanced vector-based document retrieval using FAISS
- ⚡ **Real-time Responses**: Get accurate answers with source citations
- 🔄 **Model Comparison**: Compare responses from different AI models side-by-side
- 📊 **Document Analytics**: View document statistics and processing information
- 📝 **Query History**: Track your questions and responses
- ⚙️ **Configurable Settings**: Adjust chunk size, overlap, and similarity parameters

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key and/or Anthropic API key

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd Agentic-RAG
```

### 2. Create Virtual Environment

```bash
python -m venv Environment
source Environment/bin/activate  # On Windows: Environment\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
# API Keys - Replace with your actual keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Model Configuration (Optional)
DEFAULT_LLM_MODEL=gpt-4
DEFAULT_EMBEDDING_MODEL=text-embedding-3-small

# RAG Configuration (Optional)
CHUNK_SIZE=1024
CHUNK_OVERLAP=200
SIMILARITY_TOP_K=5
```

### 5. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🎯 How to Use

### Step 1: Upload Documents
1. Click on the **"Upload PDF documents"** section
2. Select one or more PDF files from your computer
3. Click **"🚀 Process Documents"** to load them into the system

### Step 2: Create Vector Index
1. After documents are processed, you'll see document statistics
2. Click **"🔧 Create Vector Index"** to prepare the documents for querying
3. Wait for the indexing process to complete (this may take a few minutes)

### Step 3: Configure Settings
Use the sidebar to:
- **Select AI Model**: Choose from available GPT-4, Claude, or other models
- **Adjust RAG Settings**: Modify chunk size, overlap, and top-K results
- **Choose Embedding Model**: Select between OpenAI or HuggingFace embeddings
- **Enable Comparison Mode**: Compare responses from multiple models

### Step 4: Ask Questions
1. Enter your question in the text area
2. Click **"🔍 Ask Question"**
3. View the AI-generated response with source citations
4. Explore the sources to verify the information

## 🏗️ Architecture

The application follows a modular architecture:

```
Agentic-RAG/
├── app.py                 # Main Streamlit application
├── rag_pipeline.py        # RAG logic using LlamaIndex
├── llm_wrapper.py         # LLM abstraction layer
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
└── data/
    └── uploads/          # Uploaded documents (auto-created)
```

### Key Components

- **`app.py`**: User interface and application orchestration
- **`rag_pipeline.py`**: Document processing, embedding, and retrieval logic
- **`llm_wrapper.py`**: Unified interface for different AI models
- **FAISS Vector Store**: Fast similarity search for document chunks
- **LlamaIndex v0.12+**: Modern RAG framework using Settings API

## 🔧 Configuration Options

### RAG Parameters

- **Chunk Size**: Size of document chunks (512-2048 characters)
- **Chunk Overlap**: Overlap between chunks (50-500 characters)
- **Similarity Top-K**: Number of similar chunks to retrieve (3-10)

### Supported Models

#### OpenAI Models
- GPT-4
- GPT-4 Turbo
- GPT-3.5 Turbo

#### Anthropic Models
- Claude 3 Opus
- Claude 3 Sonnet
- Claude 3 Haiku

### Embedding Models
- **OpenAI**: `text-embedding-3-small` (requires API key)
- **HuggingFace**: `BAAI/bge-small-en-v1.5` (free, local)

## 🛠️ Advanced Features

### Comparison Mode
Enable comparison mode to see responses from multiple models simultaneously:
1. Check **"🔍 Comparison Mode"** in the sidebar
2. Ask a question
3. View side-by-side responses from different models

### Document Statistics
View detailed information about your uploaded documents:
- Number of documents and pages
- Total characters and estimated tokens
- File names and sizes

### Query History
Track your questions and responses in the **"📝 Query History"** section.

## 🔍 Troubleshooting

### Common Issues

**1. "No API keys found" error**
- Ensure your `.env` file is in the root directory
- Verify your API keys are correctly formatted
- Restart the application after adding keys

**2. "Failed to load documents" error**
- Check if your PDF files are readable
- Ensure files are not password-protected
- Try uploading smaller files

**3. "Model not available" error**
- Verify your API keys are valid and have sufficient credits
- Check if the selected model is accessible with your API key

**4. Slow indexing process**
- Large documents take time to process
- Consider using smaller chunk sizes for faster processing
- Ensure you have sufficient RAM for large document sets

### Performance Tips

- Use **OpenAI embeddings** for better accuracy (requires API key)
- Use **HuggingFace embeddings** for cost-free processing (slower)
- Reduce chunk size for faster indexing
- Increase chunk overlap for better context retention

## 📚 Technical Details

### Dependencies

The application uses modern versions of:
- **LlamaIndex v0.12+**: Latest RAG framework with Settings API
- **Streamlit**: Web application framework
- **FAISS**: Vector similarity search
- **Sentence Transformers**: Text embeddings
- **PyPDF2/PyMuPDF**: PDF processing

### Key Updates from LlamaIndex v0.12+

- Uses **Settings** object instead of deprecated ServiceContext
- Modular package structure with specific integrations
- Improved vector store abstractions
- Better error handling and progress tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with a clear description

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the [LlamaIndex documentation](https://docs.llamaindex.ai/)
3. Open an issue on GitHub with detailed error information

## 🙏 Acknowledgments

- [LlamaIndex](https://www.llamaindex.ai/) for the excellent RAG framework
- [Streamlit](https://streamlit.io/) for the web application framework
- [OpenAI](https://openai.com/) and [Anthropic](https://www.anthropic.com/) for the AI models

---

**Built with ❤️ using Streamlit + LlamaIndex v0.12+** 