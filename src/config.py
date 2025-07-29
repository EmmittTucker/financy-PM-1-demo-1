"""
Configuration file for the AI Financial Portfolio Advisor demo
"""

import os

# Model Configuration
MODEL_NAME = "tuc111/financy-PM-1"  # Replace with your actual HF model name
MAX_SEQUENCE_LENGTH = 2048
MAX_NEW_TOKENS = 256

# Demo Mode Configuration
DEMO_MODE = False  # Disabled for investor demo - using real model
USE_AUTH_TOKEN = True  # Set to True if model is private

# Generation Parameters
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# App Configuration  
MAX_QUESTIONS = 20
CONTEXT_WINDOW = 20  # Increased for full conversation history on GPU
MAX_NEW_TOKENS = 256 # Increased for more detailed responses on GPU

# UI Configuration
CHATBOT_HEIGHT = 500
THEME = "soft"  # Gradio theme

# Messages
# The old WELCOME_MESSAGE_TEMPLATE is removed, replaced by an LLM-generated response.

FALLBACK_MESSAGE = """
🚀 **You've reached your 20-question limit for this demo!**

Thank you for trying our AI Financial Portfolio Advisor. To continue getting personalized investment advice and access our advanced features:

✨ **Visit [grandmasboylabs.com](https://grandmasboylabs.com) for unlimited conversations!**
"""

# New, shorter system prompt. The fine-tuning handles the persona.
SYSTEM_PROMPT_TEMPLATE = "You are an expert financial advisor. Be helpful, professional, and provide specific, actionable advice."

# New template to format the user's intake form as the first message.
USER_PROFILE_TEMPLATE = """Here is my financial profile:
- Age: {age}
- Country: {country}
- Investment Experience: {investment_background}
- Primary Investment Goal: {investment_goals}
- Time Horizon: {time_horizon}
- Risk Tolerance: {risk_tolerance}

Based on this, please provide some initial high-level advice and then ask me what I'd like to discuss first.
"""

# Form Options
GENDER_OPTIONS = ["Male", "Female", "Non-binary", "Prefer not to say"]
COUNTRY_OPTIONS = ["United States", "Canada", "United Kingdom", "Australia", "Germany", "France", "Other"]
INVESTMENT_BACKGROUND_OPTIONS = ["Beginner", "Intermediate", "Advanced"]
INVESTMENT_GOALS_OPTIONS = [
    "Retirement planning",
    "Wealth building", 
    "Income generation",
    "Capital preservation",
    "Education funding",
    "Home purchase",
    "General investing"
]
TIME_HORIZON_OPTIONS = [
    "Less than 1 year",
    "1-3 years",
    "3-5 years", 
    "5-10 years",
    "10+ years"
]
RISK_TOLERANCE_OPTIONS = ["Conservative", "Moderate", "Aggressive"]

# Default Values
DEFAULT_AGE = 30
DEFAULT_GENDER = "Male"
DEFAULT_COUNTRY = "United States"
DEFAULT_INVESTMENT_BACKGROUND = "Beginner"
DEFAULT_INVESTMENT_GOALS = "Retirement planning"
DEFAULT_TIME_HORIZON = "10+ years"
DEFAULT_RISK_TOLERANCE = "Moderate" 