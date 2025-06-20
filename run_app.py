#!/usr/bin/env python3
"""
RAG AI Agent Launcher
Simple script to check setup and launch the Streamlit application.
"""

import sys
import os
import subprocess
from pathlib import Path

def check_basic_setup():
    """Quick setup check before launching"""
    print("🚀 RAG AI Agent Launcher")
    print("=" * 30)
    
    # Check if .env file exists
    if not Path('.env').exists():
        print("⚠️ No .env file found!")
        print("\nPlease create a .env file with your API keys:")
        print("""
# Create a .env file with the following content:
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
        """)
        print("Get API keys from:")
        print("- OpenAI: https://platform.openai.com/account/api-keys")
        print("- Anthropic: https://console.anthropic.com/account/keys")
        print("\nAfter creating .env file, run this script again.")
        return False
    
    # Check if main files exist
    required_files = ['app.py', 'rag_pipeline.py', 'llm_wrapper.py']
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ Missing file: {file_path}")
            return False
    
    # Check if data directory exists
    if not Path('data').exists():
        print("📁 Creating data directory...")
        Path('data').mkdir(exist_ok=True)
        Path('data/uploads').mkdir(exist_ok=True)
    
    print("✅ Basic setup looks good!")
    return True

def check_dependencies():
    """Check if key dependencies are installed"""
    try:
        import streamlit
        import llama_index
        print("✅ Key dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("\nPlease install dependencies:")
        print("pip install -r requirements.txt")
        return False

def launch_app():
    """Launch the Streamlit application"""
    print("\n🌟 Launching RAG AI Agent...")
    print("The app will open in your default browser.")
    print("Press Ctrl+C to stop the application.")
    print("-" * 50)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.headless', 'false',
            '--server.enableXsrfProtection', 'false',
            '--browser.gatherUsageStats', 'false'
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user.")
    except Exception as e:
        print(f"\n❌ Error launching application: {e}")
        print("\nTry running manually: streamlit run app.py")

def main():
    """Main launcher function"""
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check setup
    if not check_basic_setup():
        sys.exit(1)
    
    if not check_dependencies():
        print("\nRun setup test for more details: python test_setup.py")
        sys.exit(1)
    
    # Launch app
    launch_app()

if __name__ == "__main__":
    main() 