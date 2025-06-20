# 🚀 Quick Start Guide - RAG AI Agent

## 🎯 Get Started in 5 Minutes

### 1️⃣ Setup API Keys
Create a `.env` file in the project root:
```bash
# Copy content from env_template.txt
OPENAI_API_KEY=your_actual_openai_api_key
ANTHROPIC_API_KEY=your_actual_anthropic_api_key
```

**Get API Keys:**
- OpenAI: https://platform.openai.com/account/api-keys
- Anthropic: https://console.anthropic.com/

### 2️⃣ Launch the App
```bash
streamlit run app.py
```

### 3️⃣ Use the App
1. **Upload PDF**: Click "Browse files" and select your PDF document
2. **Choose Model**: Select GPT-4 or Claude from the dropdown
3. **Ask Questions**: Type your question in the text box
4. **Get Answers**: View AI responses with source citations

## 🔧 Troubleshooting

**App won't start?**
- Check Python version: `python --version` (need 3.8+)
- Install dependencies: `pip install -r requirements.txt`

**No models available?**
- Verify API keys in `.env` file
- Ensure keys are valid and have credits

**Import errors?**
- Run: `python test_setup.py` to diagnose issues

## 🌟 Features

- 📄 **Multi-PDF Support**: Upload multiple documents
- 🤖 **Model Choice**: GPT-4, Claude 3, GPT-3.5
- 🔍 **Smart Search**: Advanced vector-based retrieval
- 📊 **Analytics**: View document statistics
- 🔄 **Compare Models**: Side-by-side model comparison
- 📝 **Query History**: Track your conversations

## 🆘 Need Help?

1. Run diagnostic test: `python test_setup.py`
2. Check setup guide: `SETUP_GUIDE.md`
3. View full documentation: `README.md`

## 🚀 Launch Command
```bash
# Standard launch
streamlit run app.py

# Custom port
streamlit run app.py --server.port 8502

# Run on network
streamlit run app.py --server.address 0.0.0.0
```

---
**Happy RAG-ing! 🤖✨** 