# 📋 RAG AI Agent - Setup Guide

This guide will walk you through setting up and running the RAG-based AI Agent Web App step by step.

## 🎯 What You're Building

A modern web application that:
- Uploads PDF documents
- Creates a searchable knowledge base using vector embeddings
- Answers questions using GPT-4 or Claude models
- Shows source citations for all answers
- Compares responses from multiple AI models

## 📝 Step-by-Step Setup

### Step 1: Environment Setup

**Windows:**
```bash
# Navigate to project directory
cd Agentic-RAG

# Create virtual environment
python -m venv Environment

# Activate virtual environment
Environment\Scripts\activate
```

**Mac/Linux:**
```bash
# Navigate to project directory
cd Agentic-RAG

# Create virtual environment
python -m venv Environment

# Activate virtual environment
source Environment/bin/activate
```

### Step 2: Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

### Step 3: Get API Keys

**OpenAI API Key:**
1. Go to [OpenAI Platform](https://platform.openai.com/account/api-keys)
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key (starts with `sk-`)

**Anthropic API Key:**
1. Go to [Anthropic Console](https://console.anthropic.com/account/keys)
2. Sign in or create an account
3. Click "Create Key"
4. Copy the key (starts with `sk-ant-`)

> **Note:** You need at least one API key to use the application.

### Step 4: Configure Environment

Create a `.env` file in the project root:

```bash
# On Windows
type nul > .env

# On Mac/Linux
touch .env
```

Edit the `.env` file and add your API keys:

```
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
```

### Step 5: Test Your Setup

Run the setup test to verify everything is working:

```bash
python test_setup.py
```

You should see:
```
🚀 RAG AI Agent Setup Test
==================================================
🐍 Testing Python version...
✅ Python 3.x.x - Compatible
📦 Testing dependencies...
✅ All dependencies installed successfully!
...
🎉 All tests passed! Your setup is ready.
```

### Step 6: Launch the Application

**Option A: Use the launcher script (recommended)**
```bash
python run_app.py
```

**Option B: Launch directly**
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 🎮 Using the Application

### 1. Upload Documents
- Click "Upload PDF documents"
- Select one or more PDF files
- Click "🚀 Process Documents"

### 2. Create Index
- After documents are processed, click "🔧 Create Vector Index"
- Wait for indexing to complete (1-3 minutes depending on document size)

### 3. Configure Settings (Sidebar)
- **Select Model**: Choose GPT-4, Claude, etc.
- **Chunk Size**: 1024 (good default)
- **Top-K Results**: 5 (number of relevant chunks to retrieve)
- **Embedding Model**: OpenAI (better) or HuggingFace (free)

### 4. Ask Questions
- Type your question in the text area
- Click "🔍 Ask Question"
- View the AI response with source citations

### 5. Advanced Features
- **Comparison Mode**: Enable to see responses from multiple models
- **Query History**: View previous questions and answers
- **Source Citations**: Click to see which document chunks were used

## 🚨 Troubleshooting

### Common Issues

**"No API keys found" error:**
- Make sure `.env` file is in the correct location
- Check that API keys are valid and have no extra spaces
- Restart the application after adding keys

**"Module not found" errors:**
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again
- Check Python version (requires 3.8+)

**Slow performance:**
- Use smaller chunk sizes (512 instead of 1024)
- Use HuggingFace embeddings instead of OpenAI for free processing
- Ensure you have sufficient RAM for large documents

**Upload errors:**
- Check that PDF files are not password-protected
- Try smaller files first
- Ensure files are readable PDFs

### Performance Tips

**For Best Accuracy:**
- Use OpenAI embeddings (`text-embedding-3-small`)
- Use GPT-4 or Claude 3 Opus models
- Keep chunk overlap at 200-300 characters

**For Cost Efficiency:**
- Use HuggingFace embeddings (free)
- Use GPT-3.5 Turbo or Claude 3 Haiku
- Process smaller document sets

**For Speed:**
- Use smaller chunk sizes (512)
- Reduce top-K results to 3
- Use faster models like GPT-3.5 Turbo

## 🛠️ Advanced Configuration

### Custom Chunk Settings

Edit these values in the sidebar or modify `rag_pipeline.py`:

```python
chunk_size = 1024      # Size of each text chunk
chunk_overlap = 200    # Overlap between chunks
similarity_top_k = 5   # Number of chunks to retrieve
```

### Adding New Models

To add new LLM models, edit `llm_wrapper.py`:

```python
# Add to _initialize_models method
self.models.update({
    "new-model": YourModelClass(
        model="model-name",
        api_key=os.getenv("YOUR_API_KEY"),
        temperature=0.1
    )
})
```

### Custom Embeddings

To use different embedding models, edit `rag_pipeline.py`:

```python
# In _setup_embedding_model method
embed_model = YourEmbeddingModel(
    model_name="your-model-name"
)
```

## 📊 Understanding the Technology

### RAG Pipeline
1. **Document Loading**: PDFs are read and converted to text
2. **Chunking**: Text is split into overlapping chunks
3. **Embedding**: Each chunk is converted to a vector
4. **Storage**: Vectors are stored in FAISS index
5. **Retrieval**: Similar chunks are found for queries
6. **Generation**: LLM generates answer using retrieved chunks

### Model Comparison
- **GPT-4**: Best reasoning, higher cost
- **Claude 3 Opus**: Great analysis, longer context
- **GPT-3.5 Turbo**: Fast and cost-effective
- **Claude 3 Haiku**: Fastest, good for simple queries

## 🔐 Security Notes

- Never commit your `.env` file to version control
- Rotate API keys regularly
- Monitor API usage and costs
- Keep the application local or use proper authentication for production

## 📞 Getting Help

1. **Check the troubleshooting section above**
2. **Run the test script**: `python test_setup.py`
3. **Review logs**: Check terminal output for error messages
4. **Update dependencies**: `pip install -r requirements.txt --upgrade`
5. **Restart**: Try restarting the application

---

**Happy RAG-ing! 🤖✨** 