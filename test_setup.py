#!/usr/bin/env python3
"""
Setup Test Script for RAG AI Agent
Verifies that all dependencies are installed and configured correctly.
"""

import sys
import os
from pathlib import Path

def test_python_version():
    """Test Python version compatibility"""
    print("🐍 Testing Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def test_dependencies():
    """Test if all required dependencies are installed"""
    print("\n📦 Testing dependencies...")
    
    required_packages = [
        ('streamlit', 'Streamlit web framework'),
        ('llama_index.core', 'LlamaIndex core'),
        ('llama_index.llms.openai', 'LlamaIndex OpenAI integration'),
        ('llama_index.llms.anthropic', 'LlamaIndex Anthropic integration'),
        ('llama_index.embeddings.openai', 'LlamaIndex OpenAI embeddings'),
        ('llama_index.embeddings.huggingface', 'LlamaIndex HuggingFace embeddings'),
        ('llama_index.readers.file', 'LlamaIndex file readers'),
        ('llama_index.vector_stores.faiss', 'LlamaIndex FAISS integration'),
        ('faiss', 'FAISS vector search'),
        ('PyPDF2', 'PDF processing'),
        ('python_dotenv', 'Environment variables')
    ]
    
    missing_packages = []
    
    for package, description in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} - {description}")
        except ImportError:
            print(f"❌ {package} - {description} (MISSING)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed successfully!")
    return True

def test_environment_setup():
    """Test environment setup"""
    print("\n🔧 Testing environment setup...")
    
    # Check if .env file exists
    env_file = Path('.env')
    if env_file.exists():
        print("✅ .env file found")
        
        # Load environment variables
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            # Check API keys
            openai_key = os.getenv('OPENAI_API_KEY')
            anthropic_key = os.getenv('ANTHROPIC_API_KEY')
            
            if openai_key and openai_key != 'your_openai_api_key_here':
                print("✅ OpenAI API key configured")
            else:
                print("⚠️ OpenAI API key not configured")
            
            if anthropic_key and anthropic_key != 'your_anthropic_api_key_here':
                print("✅ Anthropic API key configured")
            else:
                print("⚠️ Anthropic API key not configured")
            
            if not openai_key and not anthropic_key:
                print("❌ No API keys configured. You need at least one API key.")
                return False
            
        except Exception as e:
            print(f"❌ Error loading environment: {e}")
            return False
    else:
        print("⚠️ .env file not found")
        print("Please create a .env file with your API keys:")
        print("""
# Example .env file content:
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
        """)
        return False
    
    return True

def test_directory_structure():
    """Test directory structure"""
    print("\n📁 Testing directory structure...")
    
    required_dirs = ['data', 'data/uploads']
    required_files = ['app.py', 'rag_pipeline.py', 'llm_wrapper.py', 'requirements.txt']
    
    # Check directories
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/ directory exists")
        else:
            print(f"❌ {dir_path}/ directory missing")
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"✅ Created {dir_path}/ directory")
    
    # Check files
    all_files_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_files_exist = False
    
    return all_files_exist

def test_model_availability():
    """Test if models can be initialized"""
    print("\n🤖 Testing model availability...")
    
    try:
        from llm_wrapper import llm_wrapper
        available_models = llm_wrapper.get_available_models()
        
        if available_models:
            print("✅ Available models:")
            for model_id, model_name in available_models.items():
                model_info = llm_wrapper.get_model_info(model_id)
                print(f"   • {model_name} ({model_info.get('provider', 'Unknown')})")
            return True
        else:
            print("❌ No models available. Check your API keys.")
            return False
            
    except Exception as e:
        print(f"❌ Error testing models: {e}")
        return False

def test_rag_pipeline():
    """Test RAG pipeline initialization"""
    print("\n🔍 Testing RAG pipeline...")
    
    try:
        from rag_pipeline import RAGPipeline
        
        # Test with HuggingFace embeddings (no API key required)
        pipeline = RAGPipeline(embedding_model="huggingface")
        print("✅ RAG pipeline initialized successfully")
        
        # Test basic functionality
        stats = pipeline.get_document_stats()
        if stats['total_docs'] == 0:
            print("✅ Document stats working (no documents loaded)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing RAG pipeline: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 RAG AI Agent Setup Test")
    print("=" * 50)
    
    tests = [
        test_python_version,
        test_dependencies,
        test_directory_structure,
        test_environment_setup,
        test_model_availability,
        test_rag_pipeline
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready.")
        print("\nTo run the application:")
        print("streamlit run app.py")
    else:
        print("⚠️ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Create .env file with your API keys")
        print("3. Ensure Python 3.8+ is installed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 