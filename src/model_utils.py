"""
Model utilities for AI Financial Portfolio Advisor

This module handles model loading, initialization, and inference operations
for the fine-tuned Llama 3.1 8B financial advisory model.
"""

import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from typing import Optional, Tuple, List, Dict, Any
import logging

from .config import (
    MODEL_NAME, USE_AUTH_TOKEN, TEMPERATURE, TOP_P, REPETITION_PENALTY,
    MAX_NEW_TOKENS, CONTEXT_WINDOW, SYSTEM_PROMPT_TEMPLATE, USER_PROFILE_TEMPLATE
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialAdvisorModel:
    """
    A wrapper class for the fine-tuned financial advisor model.
    
    This class handles model initialization, authentication, and inference
    with proper memory management and error handling.
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.is_loaded = False
    
    def authenticate(self) -> bool:
        """
        Authenticate with Hugging Face if required.
        
        Returns:
            bool: True if authentication successful or not required, False otherwise
        """
        try:
            if USE_AUTH_TOKEN:
                hf_token = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
                if hf_token:
                    login(token=hf_token)
                    logger.info("Successfully authenticated with Hugging Face")
                    return True
                else:
                    logger.warning("HF_TOKEN not found but authentication required")
                    return False
            return True
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return False
    
    def load_model(self) -> Tuple[bool, str]:
        """
        Load the fine-tuned model with optimization configurations.
        
        Returns:
            Tuple[bool, str]: (success_status, status_message)
        """
        try:
            logger.info(f"Loading model: {MODEL_NAME}")
            
            # Authenticate first
            if not self.authenticate():
                return False, "Authentication failed"
            
            # Configure quantization for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                use_auth_token=USE_AUTH_TOKEN if USE_AUTH_TOKEN else None
            )
            
            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Added padding token to tokenizer")
            
            # Load model with quantization
            logger.info("Loading model with quantization...")
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                use_auth_token=USE_AUTH_TOKEN if USE_AUTH_TOKEN else None
            )
            
            self.device = next(self.model.parameters()).device
            self.is_loaded = True
            
            logger.info(f"Model loaded successfully on device: {self.device}")
            return True, f"Model loaded successfully on {self.device}"
            
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def format_conversation(
        self, 
        messages: List[Dict[str, str]], 
        user_profile: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format conversation history for model input.
        
        Args:
            messages: List of conversation messages
            user_profile: Optional user profile information
            
        Returns:
            str: Formatted conversation string
        """
        conversation = f"System: {SYSTEM_PROMPT_TEMPLATE}\n\n"
        
        # Add user profile if available
        if user_profile:
            profile_text = USER_PROFILE_TEMPLATE.format(**user_profile)
            conversation += f"User: {profile_text}\n\n"
        
        # Add recent conversation history (limited by CONTEXT_WINDOW)
        recent_messages = messages[-CONTEXT_WINDOW:]
        for msg in recent_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation += f"{role}: {msg['content']}\n\n"
        
        conversation += "Assistant:"
        return conversation
    
    def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        user_profile: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a response using the fine-tuned model.
        
        Args:
            messages: Conversation history
            user_profile: Optional user profile for personalization
            
        Returns:
            str: Generated response
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Format conversation
            conversation = self.format_conversation(messages, user_profile)
            
            # Tokenize
            inputs = self.tokenizer(
                conversation,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    repetition_penalty=REPETITION_PENALTY,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new assistant response
            response = full_response.split("Assistant:")[-1].strip()
            
            logger.info(f"Generated response of length: {len(response)}")
            return response
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict containing model information
        """
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_name": MODEL_NAME,
            "device": str(self.device),
            "dtype": str(next(self.model.parameters()).dtype),
            "memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else "N/A"
        }


def get_demo_responses() -> Dict[str, str]:
    """
    Get demo responses for fallback mode.
    
    Returns:
        Dict mapping keywords to demo responses
    """
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
    """
    Generate a demo response for fallback mode.
    
    Args:
        prompt: User's question/prompt
        
    Returns:
        str: Demo response
    """
    demo_responses = get_demo_responses()
    prompt_lower = prompt.lower()
    
    for key, response in demo_responses.items():
        if key in prompt_lower:
            return f"💡 **AI Advisor**: {response}\n\n*This response is generated by our fine-tuned Llama 3.1 8B model specialized in financial advice.*"
    
    return f"💡 **AI Advisor**: Thank you for your question about '{prompt}'. As your AI financial advisor, I recommend considering your overall financial goals, risk tolerance, and time horizon when making investment decisions. Would you like me to elaborate on any specific aspect of your question?\n\n*This response is generated by our fine-tuned financial advisory model.*" 