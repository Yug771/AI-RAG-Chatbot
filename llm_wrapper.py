"""
LLM Wrapper for GPT-4 and Claude models
Provides a unified interface for different language models using LlamaIndex
"""

import os
from typing import Optional, Dict, Any
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.core.llms import LLM
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
ANTHROPIC_API_KEY = get_api_key("ANTHROPIC_API_KEY")

class LLMWrapper:
    """Unified wrapper for different LLM providers"""
    
    def __init__(self):
        self.models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize available models"""
        # OpenAI models
        if OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here":
            self.models.update({
                "gpt-4": OpenAI(
                    model="gpt-4",
                    temperature=0.1,
                    api_key=OPENAI_API_KEY,
                    max_tokens=2048
                ),
                "gpt-4-turbo": OpenAI(
                    model="gpt-4-turbo-preview",
                    temperature=0.1,
                    api_key=OPENAI_API_KEY,
                    max_tokens=4096
                ),
                "gpt-3.5-turbo": OpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.1,
                    api_key=OPENAI_API_KEY,
                    max_tokens=2048
                )
            })
        
        # Anthropic models
        if ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != "your_anthropic_api_key_here":
            self.models.update({
                "claude-3-opus": Anthropic(
                    model="claude-3-opus-20240229",
                    temperature=0.1,
                    api_key=ANTHROPIC_API_KEY,
                    max_tokens=2048
                ),
                "claude-3-sonnet": Anthropic(
                    model="claude-3-sonnet-20240229",
                    temperature=0.1,
                    api_key=ANTHROPIC_API_KEY,
                    max_tokens=2048
                ),
                "claude-3-haiku": Anthropic(
                    model="claude-3-haiku-20240307",
                    temperature=0.1,
                    api_key=ANTHROPIC_API_KEY,
                    max_tokens=2048
                )
            })
    
    def get_available_models(self) -> Dict[str, str]:
        """Get list of available models with display names"""
        model_display_names = {
            "gpt-4": "GPT-4",
            "gpt-4-turbo": "GPT-4 Turbo",
            "gpt-3.5-turbo": "GPT-3.5 Turbo",
            "claude-3-opus": "Claude 3 Opus",
            "claude-3-sonnet": "Claude 3 Sonnet",
            "claude-3-haiku": "Claude 3 Haiku"
        }
        
        available = {}
        for model_id in self.models.keys():
            if model_id in model_display_names:
                available[model_id] = model_display_names[model_id]
        
        return available
    
    def get_model(self, model_name: str) -> Optional[LLM]:
        """Get a specific model instance"""
        return self.models.get(model_name)
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        model_info = {
            "gpt-4": {
                "provider": "OpenAI",
                "context_length": 8192,
                "description": "Most capable GPT model, best for complex reasoning"
            },
            "gpt-4-turbo": {
                "provider": "OpenAI", 
                "context_length": 128000,
                "description": "Latest GPT-4 with larger context window"
            },
            "gpt-3.5-turbo": {
                "provider": "OpenAI",
                "context_length": 4096,
                "description": "Fast and efficient for most tasks"
            },
            "claude-3-opus": {
                "provider": "Anthropic",
                "context_length": 200000,
                "description": "Most capable Claude model, excellent for analysis"
            },
            "claude-3-sonnet": {
                "provider": "Anthropic",
                "context_length": 200000,
                "description": "Balanced performance and speed"
            },
            "claude-3-haiku": {
                "provider": "Anthropic",
                "context_length": 200000,
                "description": "Fastest Claude model, good for simple tasks"
            }
        }
        
        return model_info.get(model_name, {})
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available and properly configured"""
        return model_name in self.models
    
    def get_provider_models(self, provider: str) -> Dict[str, str]:
        """Get models for a specific provider"""
        provider_mapping = {
            "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
            "anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
        }
        
        available_models = self.get_available_models()
        provider_models = {}
        
        for model_id in provider_mapping.get(provider.lower(), []):
            if model_id in available_models:
                provider_models[model_id] = available_models[model_id]
        
        return provider_models

# Global instance
llm_wrapper = LLMWrapper() 