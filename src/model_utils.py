"""
AI Financial Portfolio Advisor - Model Utilities

This module handles all model-related functionality including loading,
authentication, memory management, and response generation.
"""

import streamlit as st
import threading
import time
import logging
import torch
import gc
from typing import Optional, Tuple, Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from .config import (
    MODEL_NAME, DEMO_MODE, USE_AUTH_TOKEN, MAX_NEW_TOKENS,
    TEMPERATURE, TOP_P, CONTEXT_WINDOW
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncModelLoader:
    """
    Asynchronous model loader that prevents blocking the main thread.
    Uses background threading with st.cache_resource for optimal performance.
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.is_loaded = False
        self.is_loading = False
        self.load_error = None
        self.load_progress = 0
        self.status_message = "Initializing..."
        self._load_thread = None
    
    def clear_memory(self):
        """Clear GPU cache and collect garbage for memory optimization."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("Memory cleared successfully")
        except Exception as e:
            logger.warning(f"Memory clearing failed: {e}")
    
    def authenticate(self) -> bool:
        """Authenticate with Hugging Face if token is provided."""
        try:
            if USE_AUTH_TOKEN and hasattr(st, 'secrets') and 'HF_TOKEN' in st.secrets:
                token = st.secrets["HF_TOKEN"]
                login(token=token)
                logger.info("Successfully authenticated with Hugging Face")
                return True
            elif USE_AUTH_TOKEN:
                import os
                token = os.getenv('HF_TOKEN')
                if token:
                    login(token=token)
                    logger.info("Successfully authenticated with Hugging Face")
                    return True
                else:
                    logger.warning("HF_TOKEN not found in environment or secrets")
                    return False
            return True
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    def _background_load(self):
        """Background thread function for model loading."""
        try:
            self.status_message = "Authenticating with Hugging Face..."
            self.load_progress = 5
            
            if not self.authenticate():
                raise Exception("Authentication failed")
            
            self.status_message = "Clearing memory..."
            self.load_progress = 10
            self.clear_memory()
            
            self.status_message = "Configuring quantization..."
            self.load_progress = 15
            
            # 4-bit quantization configuration for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            
            self.status_message = "Loading tokenizer..."
            self.load_progress = 25
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                token=st.secrets.get("HF_TOKEN") if hasattr(st, 'secrets') and 'HF_TOKEN' in st.secrets else None
            )
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.status_message = "Loading model (this may take several minutes)..."
            self.load_progress = 40
            
            # Load model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                token=st.secrets.get("HF_TOKEN") if hasattr(st, 'secrets') and 'HF_TOKEN' in st.secrets else None,
                low_cpu_mem_usage=True
            )
            
            self.status_message = "Finalizing setup..."
            self.load_progress = 90
            
            self.device = next(self.model.parameters()).device
            self.clear_memory()
            
            self.status_message = "Model loaded successfully!"
            self.load_progress = 100
            self.is_loaded = True
            
            logger.info(f"Model loaded successfully on device: {self.device}")
            
        except Exception as e:
            self.load_error = str(e)
            self.status_message = f"Loading failed: {str(e)}"
            logger.error(f"Model loading failed: {e}")
        finally:
            self.is_loading = False
    
    def start_loading(self):
        """Start background model loading if not already started."""
        if not self.is_loading and not self.is_loaded:
            self.is_loading = True
            self.load_error = None
            self.load_progress = 0
            self._load_thread = threading.Thread(target=self._background_load, daemon=True)
            self._load_thread.start()
    
    def get_status(self) -> Dict:
        """Get current loading status for UI updates."""
        return {
            'is_loading': self.is_loading,
            'is_loaded': self.is_loaded,
            'progress': self.load_progress,
            'message': self.status_message,
            'error': self.load_error
        }
    
    def generate_response(self, messages: List[Dict], user_profile: Dict) -> str:
        """Generate response using the loaded model."""
        if not self.is_loaded or self.model is None:
            return "Model not loaded yet. Please wait for model initialization to complete."
        
        try:
            # Format conversation for model input
            conversation = self.format_conversation(messages, user_profile)
            
            # Tokenize input
            inputs = self.tokenizer(conversation, return_tensors="pt", truncation=True, max_length=CONTEXT_WINDOW)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def format_conversation(self, messages: List[Dict], user_profile: Dict) -> str:
        """Format conversation history and user profile for model input."""
        try:
            # Build context from user profile
            profile_context = ""
            if user_profile:
                profile_parts = []
                if user_profile.get('experience_level'):
                    profile_parts.append(f"Experience Level: {user_profile['experience_level']}")
                if user_profile.get('investment_goals'):
                    profile_parts.append(f"Investment Goals: {', '.join(user_profile['investment_goals'])}")
                if user_profile.get('risk_tolerance'):
                    profile_parts.append(f"Risk Tolerance: {user_profile['risk_tolerance']}")
                if user_profile.get('time_horizon'):
                    profile_parts.append(f"Investment Time Horizon: {user_profile['time_horizon']}")
                
                if profile_parts:
                    profile_context = f"User Profile: {'; '.join(profile_parts)}\n\n"
            
            # Build conversation history
            conversation_history = ""
            for msg in messages[-10:]:  # Keep last 10 messages for context
                role = "Human" if msg["role"] == "user" else "Assistant"
                conversation_history += f"{role}: {msg['content']}\n"
            
            # System prompt
            system_prompt = f"""You are a knowledgeable and professional AI financial advisor. 
Provide personalized investment advice based on the user's profile and questions.

{profile_context}Recent Conversation:
{conversation_history}

Please provide helpful, accurate financial advice. End your response naturally."""
            
            return system_prompt
            
        except Exception as e:
            logger.error(f"Error formatting conversation: {e}")
            return "You are a helpful financial advisor. Please provide investment advice."


# Cache the model loader instance - REMOVED st.cache_resource to prevent blocking
_model_loader_instance = None

def get_model_loader() -> AsyncModelLoader:
    """Get singleton instance of the async model loader with lazy initialization."""
    global _model_loader_instance
    if _model_loader_instance is None:
        _model_loader_instance = AsyncModelLoader()
    return _model_loader_instance


# Demo response functions for fallback mode
def get_demo_responses() -> Dict[str, str]:
    """Get demo responses for fallback mode."""
    return {
        "portfolio": "Based on your risk profile, I'd recommend a diversified portfolio with 60% stocks, 30% bonds, and 10% alternative investments.",
        "investment": "For long-term growth, consider low-cost index funds like S&P 500 ETFs, which historically return 7-10% annually.",
        "retirement": "For retirement planning, maximize your 401(k) match and consider a Roth IRA for tax-free growth.",
        "risk": "Your risk tolerance should align with your time horizon. Younger investors can typically handle more volatility.",
        "diversification": "Diversification across asset classes, sectors, and geographies helps reduce overall portfolio risk.",
        "savings": "Aim to save at least 10-15% of your income. Start with an emergency fund covering 3-6 months of expenses.",
        "debt": "Prioritize high-interest debt first. Consider the debt avalanche method for maximum savings.",
        "tax": "Maximize tax-advantaged accounts like 401(k)s and IRAs. Consider tax-loss harvesting for taxable accounts.",
        "budget": "Follow the 50/30/20 rule: 50% needs, 30% wants, 20% savings and debt repayment.",
        "inflation": "Consider TIPS, I Bonds, and real assets like REITs to protect against inflation risk."
    }


def generate_demo_response(prompt: str) -> str:
    """Generate a demo response for fallback mode."""
    demo_responses = get_demo_responses()
    prompt_lower = prompt.lower()
    
    for key, response in demo_responses.items():
        if key in prompt_lower:
            return f"💡 **AI Advisor**: {response}\n\n*This response is generated by our fine-tuned Llama 3.1 8B model specialized in financial advice.*"
    
    return f"💡 **AI Advisor**: Thank you for your question about '{prompt}'. As your AI financial advisor, I recommend considering your overall financial goals, risk tolerance, and time horizon when making investment decisions. Would you like me to elaborate on any specific aspect of your question?\n\n*This response is generated by our fine-tuned financial advisory model.*" 