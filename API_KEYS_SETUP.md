# 🔑 API Keys Setup Guide

The application now supports both local development and Streamlit Cloud deployment with proper environment variable management.

## 🏠 Local Development Setup

### Step 1: Create `.env` File

Create a `.env` file in the project root directory:

```bash
# Create .env file
touch .env  # On Windows: type nul > .env
```

### Step 2: Add Your API Keys

Edit the `.env` file and add your API keys:

```env
# Required: OpenAI API Key (for embeddings)
OPENAI_API_KEY=sk-your-actual-openai-key-here

# Optional: Anthropic API Key (for Claude models)
ANTHROPIC_API_KEY=sk-ant-your-actual-anthropic-key-here

# Optional Configuration
DEFAULT_LLM_MODEL=gpt-4
DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
CHUNK_SIZE=1024
CHUNK_OVERLAP=200
SIMILARITY_TOP_K=5
```

## ☁️ Streamlit Cloud Deployment

### Step 1: Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Deploy your app

### Step 2: Configure Secrets

In your Streamlit Cloud app dashboard:

1. Go to **Settings** → **Secrets**
2. Add your secrets in TOML format:

```toml
# Streamlit Cloud Secrets
OPENAI_API_KEY = "sk-your-actual-openai-key-here"
ANTHROPIC_API_KEY = "sk-ant-your-actual-anthropic-key-here"
```

## 🔗 Get Your API Keys

- **OpenAI**: https://platform.openai.com/account/api-keys
- **Anthropic**: https://console.anthropic.com/account/keys

## ⚠️ Important Notes

1. **Required**: OpenAI API key is required for embeddings to work
2. **Optional**: Anthropic API key enables Claude models (recommended)
3. **Security**: Never commit `.env` files to version control
4. **Format**: No quotes needed in `.env` files or Streamlit secrets

## 🚀 Running the Application

### Local Development
```bash
# Make sure .env file is configured
streamlit run app.py
```

### Streamlit Cloud
- Just push to GitHub and deploy
- Configure secrets in Streamlit Cloud dashboard
- App will automatically use secrets

## 🔧 Troubleshooting

### Local Development Issues:
1. Ensure `.env` file is in the project root
2. Check for typos in environment variable names
3. Verify API keys are valid on provider websites
4. Restart the app after changing `.env` file

### Streamlit Cloud Issues:
1. Check secrets are properly formatted in TOML
2. Ensure secrets names match exactly
3. Redeploy app after adding secrets
4. Check app logs for specific error messages

## 📁 File Structure

```
your-project/
├── .env                 # Your local API keys (don't commit!)
├── .env.example         # Template for others (optional)
├── app.py              # Main application
├── llm_wrapper.py      # LLM configuration
├── rag_pipeline.py     # RAG logic
└── requirements.txt    # Dependencies
```

## 🎯 Quick Start Commands

```bash
# 1. Clone and setup
git clone your-repo
cd your-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file
cp .env.example .env  # Then edit with your keys

# 4. Run application
streamlit run app.py
``` 