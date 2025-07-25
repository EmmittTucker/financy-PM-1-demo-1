#!/usr/bin/env python3
"""
AI Financial Portfolio Advisor - Optimized for fast Hugging Face Spaces startup
"""
import os
import sys
from datetime import datetime

print("--- SCRIPT EXECUTION STARTED ---")
import gradio as gr
print("--- GRADIO IMPORTED ---")
from config import *
import config
print("--- CONFIG IMPORTED ---")

print("=" * 50)
print(f"🚀 AI Financial Portfolio Advisor")
print(f"⏰ Startup Time: {datetime.now()}")
print(f"🗂️ Cache Directory: {os.environ.get('HF_HOME')}")
print(f"🎭 Demo Mode: {config.DEMO_MODE}")
print(f"🌟 Status: Initializing Gradio interface...")
print("=" * 50)

# Conditional imports - only import heavy dependencies when needed
torch = None
AutoTokenizer = None
AutoModelForCausalLM = None
BitsAndBytesConfig = None

def lazy_import_ml_deps():
    """Import ML dependencies only when actually needed"""
    global torch, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    if torch is None:
        print("📦 Loading ML dependencies...")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        print(f"✅ ML dependencies loaded")

# Standard library imports (fast)
import json
import random
import re
import signal
import threading
import time

def cleanup_cache():
    """Clean up temporary cache to free space"""
    import shutil
    cache_dirs = ["/tmp/hf_cache", "/tmp/offload"]
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                print(f"🧹 Cleaned up cache: {cache_dir}")
            except Exception as e:
                print(f"⚠️ Cache cleanup failed for {cache_dir}: {e}")

def check_available_space():
    """Check available disk space and warn if low"""
    try:
        import shutil
        total, used, free = shutil.disk_usage("/tmp")
        free_gb = free // (1024**3)
        print(f"💾 Available /tmp space: {free_gb}GB")
        if free_gb < 5:  # Less than 5GB available
            print("⚠️ Low disk space detected - enabling demo mode")
            config.DEMO_MODE = True
            return False
        return True
    except Exception as e:
        print(f"⚠️ Could not check disk space: {e}")
        return True

# Global variables for model and conversation state
model = None
tokenizer = None
conversation_history = {}
user_profiles = {}

def load_model_with_timeout(timeout_seconds=300):
    """Load model with timeout protection - only when not in demo mode"""
    global model, tokenizer
    
    # Import ML dependencies only when actually needed
    lazy_import_ml_deps()
    
    # Get the token outside the worker thread to avoid environment access issues
    auth_token = os.environ.get("HF_TOKEN") if USE_AUTH_TOKEN else None
    
    if USE_AUTH_TOKEN and not auth_token:
        print("🔑 No authentication token found")
        return "no_token"
    
    print(f"Using token: {auth_token[:10]}..." if auth_token else "No token")
    
    def model_loading_worker():
        """Worker function for model loading"""
        global model, tokenizer
        try:
            model_name = MODEL_NAME
            
            # Create directories for model storage and cache
            os.makedirs("/tmp/offload", exist_ok=True)
            os.makedirs("/tmp/hf_cache", exist_ok=True)
            
            # Configure quantization to fix deprecation warnings
            bnb_config_4bit = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            bnb_config_8bit = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            
            print("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=auth_token,
                trust_remote_code=True
            )
            print("✅ Tokenizer loaded!")
            
            print("Loading model... (this may take several minutes)")
            print("🗂️ Using /tmp cache to avoid 50GB storage limit")
            
            # Try loading with aggressive memory optimizations to avoid 50GB storage limit
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,  # Force float16 for memory efficiency
                    device_map="auto",
                    trust_remote_code=True,
                    token=auth_token,
                    low_cpu_mem_usage=True,
                    quantization_config=bnb_config_4bit,  # Use proper BitsAndBytesConfig
                    cache_dir="/tmp/hf_cache",  # Explicit cache redirect
                    offload_folder="/tmp/offload",  # Use temporary storage for offloading
                    offload_state_dict=True,  # Offload state dict to save memory
                )
                print("✅ Model loaded with 4-bit quantization!")
            except Exception as e:
                print(f"⚠️ 4-bit loading failed: {e}")
                print("🔄 Trying 8-bit quantization...")
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        device_map="auto", 
                        trust_remote_code=True,
                        token=auth_token,
                        low_cpu_mem_usage=True,
                        quantization_config=bnb_config_8bit,  # Use proper BitsAndBytesConfig
                        cache_dir="/tmp/hf_cache",  # Explicit cache redirect
                        offload_folder="/tmp/offload",
                        offload_state_dict=True,
                    )
                    print("✅ Model loaded with 8-bit quantization!")
                except Exception as e2:
                    print(f"⚠️ 8-bit loading failed: {e2}")
                    print("🔄 Trying standard loading with disk offload...")
                    # Final fallback - standard loading with aggressive offloading
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True, 
                        token=auth_token,
                        low_cpu_mem_usage=True,
                        cache_dir="/tmp/hf_cache",  # Explicit cache redirect
                        offload_folder="/tmp/offload",
                        offload_state_dict=True,
                        max_memory={0: "15GB", "cpu": "30GB"}  # Limit memory usage
                    )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            return "success"
            
        except Exception as e:
            print(f"❌ All model loading attempts failed: {str(e)}")
            print("🎭 Falling back to demo mode - app will continue with LLM-driven responses")
            
            # Clean up cache on failure to free space
            cleanup_cache()
            
            # Check if it was a storage-related error
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['storage', 'disk', 'space', 'limit', 'memory']):
                print("💾 Storage-related error detected - this is likely due to the 50GB Hugging Face Spaces limit")
                print("🎭 Demo mode provides full AI functionality with conversation limits")
            
            # Set global demo mode flag to gracefully degrade
            config.DEMO_MODE = True
            return "demo"
    
    # Create a thread for model loading
    result = ["timeout"]  # Use list to make it mutable
    
    def worker():
        result[0] = model_loading_worker()
    
    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    
    # Wait for completion or timeout
    thread.join(timeout=timeout_seconds)
    
    if thread.is_alive():
        print(f"⏰ Model loading timed out after {timeout_seconds} seconds")
        print("🎭 Continuing with LLM-driven demo mode...")
        return "demo"
    
    if result[0] == "success":
        print("✅ Model loaded successfully!")
        return True
    elif result[0] == "no_token":
        print("🔑 No authentication token found")
        return "demo"
    else:
        print("❌ Model loading failed")
        return "demo"

def load_model():
    """Load the fine-tuned model and tokenizer with robust error handling"""
    global model, tokenizer
    
    # Check available disk space first
    if not check_available_space():
        print("💾 Insufficient disk space - switching to demo mode")
        return "demo"
    
    # Clean up any existing cache to start fresh
    cleanup_cache()
    
    # If demo mode is enabled, skip model loading
    if config.DEMO_MODE:
        print("🎭 Demo mode enabled - using LLM-driven responses with conversation limits")
        return "demo"
    
    # Reset global variables
    model = None
    tokenizer = None
    
    print("🚀 Attempting to load fine-tuned model...")
    print("💡 If loading fails or takes too long, app will continue with LLM-driven demo mode")
    
    try:
        # Use timeout-protected loading
        result = load_model_with_timeout(timeout_seconds=180)  # 3 minute timeout
        return result
        
    except KeyboardInterrupt:
        print("\n⚠️ Model loading interrupted by user")
        print("🎭 Continuing with LLM-driven demo mode...")
        return "demo"
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Unexpected error during model loading: {error_msg}")
        print("🎭 Continuing with LLM-driven demo mode (conversation-limited)...")
        return "demo"

def create_user_profile(age, gender, country, investment_background, investment_goals, time_horizon, risk_tolerance):
    """Create a user profile from intake form data"""
    return {
        "age": age,
        "gender": gender,
        "country": country,
        "investment_background": investment_background,
        "investment_goals": investment_goals,
        "time_horizon": time_horizon,
        "risk_tolerance": risk_tolerance,
        "created_at": datetime.now().isoformat()
    }

def format_system_prompt():
    """Format the streamlined system prompt."""
    return SYSTEM_PROMPT_TEMPLATE

def generate_demo_response(user_input, user_profile):
    """Generate a demo response when model is not available"""
    # Template responses based on common financial topics
    demo_responses = {
        "portfolio": f"Based on your {user_profile.get('investment_background', 'investment')} experience and {user_profile.get('risk_tolerance', 'moderate')} risk tolerance, I'd recommend a diversified portfolio approach. For someone with your {user_profile.get('time_horizon', 'long-term')} timeline, consider a mix of stocks, bonds, and other assets that align with your {user_profile.get('investment_goals', 'goals')}.",
        
        "retirement": f"For retirement planning at age {user_profile.get('age', 30)}, it's important to start early and be consistent. With your {user_profile.get('time_horizon', 'timeline')}, you have time to take advantage of compound growth. Consider maximizing contributions to retirement accounts and building a diversified portfolio.",
        
        "risk": f"Given your {user_profile.get('risk_tolerance', 'moderate')} risk tolerance and {user_profile.get('investment_background', 'intermediate')} experience, I'd suggest balancing growth potential with capital preservation. This typically means a mix of stocks and bonds appropriate for your comfort level.",
        
        "default": f"Thank you for your question about {user_input[:50]}{'...' if len(user_input) > 50 else ''}. As a financial advisor, I'd recommend considering your personal situation - with your {user_profile.get('investment_background', 'investment')} background and {user_profile.get('investment_goals', 'financial goals')}, there are several strategies we could explore. Would you like me to elaborate on any specific aspect?"
    }
    
    # Simple keyword matching to select appropriate response
    user_input_lower = user_input.lower()
    if any(word in user_input_lower for word in ['portfolio', 'asset', 'allocation', 'diversif']):
        return demo_responses["portfolio"]
    elif any(word in user_input_lower for word in ['retirement', '401k', 'ira', 'pension']):
        return demo_responses["retirement"]  
    elif any(word in user_input_lower for word in ['risk', 'safe', 'conservative', 'aggressive']):
        return demo_responses["risk"]
    else:
        return demo_responses["default"]

def generate_response(user_input, session_id):
    """
    Generate an AI response. This function is now the single point of entry
    for all LLM calls, whether in demo mode or full mode.
    """
    global model, tokenizer, conversation_history, user_profiles
    
    # Standard checks for question limits and user profile
    if session_id not in user_profiles:
        return "Please complete the intake form first.", "", gr.update(visible=False)

    question_count = len([msg for msg in conversation_history.get(session_id, []) if msg["role"] == "user"])
    if question_count >= MAX_QUESTIONS:
        return FALLBACK_MESSAGE, "", gr.update(visible=False)

    # Get user profile and conversation history at the beginning
    user_profile = user_profiles.get(session_id, {})
    history = conversation_history.get(session_id, [])
    
    model_available = model is not None and tokenizer is not None
    print(f"🔍 Debug - Model available: {model_available}, DEMO_MODE: {DEMO_MODE}")

    if not DEMO_MODE and model_available:
        print("🤖 Using actual fine-tuned model for response generation")
        # Restore the high-quality, detailed prompt for GPU performance
        enhanced_system_prompt = f"""You are a Certified Financial Planner (CFP) and Chartered Financial Analyst (CFA) with 20+ years of experience.

CLIENT PROFILE:
- Age: {user_profile.get('age', 30)}
- Investment Experience: {user_profile.get('investment_background', 'N/A')}
- Primary Goals: {user_profile.get('investment_goals', 'N/A')}
- Investment Timeline: {user_profile.get('time_horizon', 'N/A')}
- Risk Tolerance: {user_profile.get('risk_tolerance', 'N/A')}

Provide comprehensive, expert-level financial advice. Be specific, actionable, and educational. Reference the client's profile to personalize your response."""
        
        conversation_text = enhanced_system_prompt + "\n\n"
        if len(history) > 1:
            last_user_msg = history[-2]['content']
            last_ai_msg = history[-1]['content']
            conversation_text += f"User: {last_user_msg}\nAdvisor: {last_ai_msg}\n"

        conversation_text += f"User: {user_input}\nAdvisor:"

    else:
        print("🎭 Using LLM-driven demo response generation.")

        # Simplified prompt construction for maximum speed
        system_prompt = format_system_prompt()
        
        # Minimal context, focusing only on the last exchange
        conversation_text = system_prompt + "\n\n"
        if len(history) > 1:
            last_user_msg = history[-2]['content']
            last_ai_msg = history[-1]['content']
            conversation_text += f"User: {last_user_msg}\nAdvisor: {last_ai_msg}\n"

        conversation_text += f"User: {user_input}\nAdvisor:"

    # Check if model and tokenizer are available before attempting generation
    if model is None or tokenizer is None or config.DEMO_MODE:
        print("🎭 Using demo response (model not loaded or demo mode active)")
        # Generate a demo response based on user input and profile
        response = generate_demo_response(user_input, user_profile)
        print(f"✅ Demo response generated: {len(response)} characters")
    else:
        try:
            # Import torch only when we actually need it for model inference
            lazy_import_ml_deps()
            
            # Tokenization and generation
            tokenized = tokenizer(conversation_text, return_tensors="pt", max_length=1024, truncation=True, padding=True)
            inputs = tokenized.input_ids
            attention_mask = tokenized.attention_mask

            if torch.cuda.is_available():
                inputs, attention_mask = inputs.to("cuda"), attention_mask.to("cuda")

            print(f"🔄 Generating response... Input length: {inputs.shape[1]} tokens")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()
            print(f"✅ Model generated response length: {len(response)} characters")
        except Exception as model_error:
            print(f"❌ Model generation failed: {model_error}")
            response = generate_demo_response(user_input, user_profile)
            print(f"✅ Fallback demo response generated: {len(response)} characters")

    # Enhanced response cleaning (applies to both model and demo responses)
    if response.startswith("Advisor:"):
        response = response[8:].strip()
    if response.startswith("EXPERT FINANCIAL ADVISOR RESPONSE:"):
        response = response[35:].strip()
    
    # Remove common artifacts and improve formatting
    response = re.sub(r'\n+', '\n', response)  # Remove excessive newlines
    response = re.sub(r'\s+', ' ', response)   # Normalize whitespace
    
    # Ensure response doesn't end abruptly
    if response and not response[-1] in '.!?':
        if '?' in response:
            # If there's a question, make sure it ends properly
            last_question = response.rfind('?')
            if last_question < len(response) - 20:  # Question not at end
                response = response[:last_question+1]
        elif '.' in response:
            # End at last complete sentence
            last_period = response.rfind('.')
            if last_period > len(response) * 0.7:  # Don't cut too much
                response = response[:last_period+1]
    
    # Add personalization if response seems too generic
    if len(response.split()) < 20:  # Very short response
        response += f" Given your {user_profile.get('investment_background', 'investment')} background and {user_profile.get('investment_goals', 'financial goals').lower()}, I'd be happy to elaborate on any specific aspect that interests you most."
    
    # Update conversation history
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    
    conversation_history[session_id].append({"role": "user", "content": user_input})
    conversation_history[session_id].append({"role": "assistant", "content": response})
    
    # Keep only last 20 exchanges (40 messages total: 20 user + 20 assistant)
    if len(conversation_history[session_id]) > 40:
        conversation_history[session_id] = conversation_history[session_id][-40:]
    
    return response, "", gr.update(visible=True)

def start_conversation(age, gender, country, investment_background, investment_goals, time_horizon, risk_tolerance, session_id):
    """Initialize conversation by generating the first AI response based on the intake form."""
    global user_profiles, conversation_history

    if not all([age, gender, country, investment_background, investment_goals, time_horizon, risk_tolerance]):
        return "Please fill in all fields before starting the conversation.", "", gr.update(visible=False)

    user_profile = create_user_profile(age, gender, country, investment_background, investment_goals, time_horizon, risk_tolerance)
    user_profiles[session_id] = user_profile
    conversation_history[session_id] = []

    # Format the intake form as the first "user" message
    initial_user_message = USER_PROFILE_TEMPLATE.format(**user_profile)

    # Generate the initial AI response
    print("🤖 Generating initial high-level advice based on user profile...")
    initial_ai_response = generate_response(initial_user_message, session_id)

    # The user's "message" is the formatted profile, and the AI provides the first response.
    # We don't show the user's formatted profile in the chat history to keep it clean.
    initial_chat_history = [(None, initial_ai_response)]
    
    # Update conversation history with both the synthetic user message and the AI response
    conversation_history[session_id].append({"role": "user", "content": initial_user_message})
    conversation_history[session_id].append({"role": "assistant", "content": initial_ai_response})

    return initial_chat_history, "", gr.update(visible=True)

def chat_interface(user_input, chat_history, session_id):
    """Handle subsequent chat interactions."""
    if not user_input.strip():
        return chat_history, ""

    if session_id not in user_profiles:
        chat_history.append((user_input, "Please complete the intake form first."))
        return chat_history, ""

    question_count = len([msg for msg in conversation_history.get(session_id, []) if msg["role"] == "user"])
    if question_count >= MAX_QUESTIONS:
        chat_history.append((user_input, FALLBACK_MESSAGE))
        return chat_history, ""

    ai_response = generate_response(user_input, session_id)
    chat_history.append((user_input, ai_response))
    
    # The initial message is now the 2nd user message in the history
    remaining_questions = MAX_QUESTIONS - question_count
    if remaining_questions > 0:
        counter_msg = f"\n\n---\n*Questions remaining: {remaining_questions}/{MAX_QUESTIONS}*"
        chat_history[-1] = (chat_history[-1][0], chat_history[-1][1] + counter_msg)

    return chat_history, ""

def reset_conversation():
    """Reset the conversation"""
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return [], "", gr.update(visible=False), session_id

# Skip model loading on startup for faster initialization
print("⚡ Fast startup mode - skipping model loading")
print("🎭 Demo mode active - model will be loaded only if needed")

# Set demo mode status
model_status = "🎭 **Demo Mode**: Fast startup with AI-driven responses. Visit [grandmasboylabs.com](https://grandmasboylabs.com) for unlimited conversations!"
model_loaded = True  # For UI purposes

# Create Gradio interface
theme_map = {"soft": gr.themes.Soft(), "default": gr.themes.Default()}
with gr.Blocks(title="AI Financial Portfolio Advisor - Demo", theme=theme_map.get(THEME, gr.themes.Soft())) as demo:
    
    # Session state
    session_id = gr.State(value=datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    
    # Header
    gr.Markdown("""
    # 🚀 AI Financial Portfolio Advisor
    
    **Powered by Fine-tuned Llama 3.1 8B**
    
    Get personalized investment advice from our AI financial advisor. Complete the intake form below to start your consultation.
    
    ⭐ **Current Features:**
    - Expert-level financial advice powered by LLM
    - Personalized recommendations based on your profile
    - Professional investment guidance and strategies
    - Educational financial insights and explanations
    
    *Note: Demo mode provides 20 questions with full AI-driven responses. For unlimited conversations, visit [grandmasboylabs.com](https://grandmasboylabs.com)!*
    """)
    
    # Model status indicator
    gr.Markdown(f"**Status**: {model_status}")
    
    with gr.Row():
        # Intake Form Column
        with gr.Column(scale=1):
            gr.Markdown("## 📋 Client Intake Form")
            
            age = gr.Slider(
                minimum=18, 
                maximum=80, 
                value=DEFAULT_AGE, 
                step=1, 
                label="Age"
            )
            
            gender = gr.Dropdown(
                choices=GENDER_OPTIONS,
                label="Gender",
                value=DEFAULT_GENDER
            )
            
            country = gr.Dropdown(
                choices=COUNTRY_OPTIONS,
                label="Country",
                value=DEFAULT_COUNTRY
            )
            
            investment_background = gr.Radio(
                choices=INVESTMENT_BACKGROUND_OPTIONS,
                label="Investment Experience",
                value=DEFAULT_INVESTMENT_BACKGROUND
            )
            
            investment_goals = gr.Dropdown(
                choices=INVESTMENT_GOALS_OPTIONS,
                label="Primary Investment Goals",
                value=DEFAULT_INVESTMENT_GOALS
            )
            
            time_horizon = gr.Dropdown(
                choices=TIME_HORIZON_OPTIONS,
                label="Investment Time Horizon",
                value=DEFAULT_TIME_HORIZON
            )
            
            risk_tolerance = gr.Radio(
                choices=RISK_TOLERANCE_OPTIONS,
                label="Risk Tolerance",
                value=DEFAULT_RISK_TOLERANCE
            )
            
            start_btn = gr.Button("🚀 Start Consultation", variant="primary", size="lg")
            reset_btn = gr.Button("🔄 Reset", variant="secondary")
        
        # Chat Interface Column
        with gr.Column(scale=2):
            gr.Markdown("## 💬 Financial Advisory Chat")
            gr.Markdown("*Complete the intake form and click 'Start Consultation' to begin your financial advisory session.*")
            
            chatbot = gr.Chatbot(
                height=CHATBOT_HEIGHT,
                show_label=False
            )
            
            with gr.Row(visible=False) as chat_interface_row:
                msg = gr.Textbox(
                    placeholder="Ask me about investments, portfolio allocation, risk management...",
                    show_label=False,
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
    
    # Footer
    gr.Markdown("""
    ---
    
    **⚠️ Important Disclaimer**: This AI advisor is for educational purposes only. All advice should be verified with qualified financial professionals. Past performance doesn't guarantee future results.
    
    **🔗 Want More?** Visit [grandmasboylabs.com](https://grandmasboylabs.com) for our advanced financial advisory platform with unlimited conversations and premium features!
    
    *Built with 🦥 Unsloth fine-tuned Llama 3.1*
    """)
    
    # Event handlers
    start_btn.click(
        fn=start_conversation,
        inputs=[age, gender, country, investment_background, investment_goals, time_horizon, risk_tolerance, session_id],
        outputs=[chatbot, msg, chat_interface_row]
    )
    
    send_btn.click(
        fn=chat_interface,
        inputs=[msg, chatbot, session_id],
        outputs=[chatbot, msg]
    )
    
    msg.submit(
        fn=chat_interface,
        inputs=[msg, chatbot, session_id],
        outputs=[chatbot, msg]
    )
    
    reset_btn.click(
        fn=reset_conversation,
        inputs=[],
                 outputs=[chatbot, msg, chat_interface_row, session_id]
     )

print("🎯 Gradio interface configured successfully")
print("⚡ Ready for fast, stable launch")

# Launch the app with simplified, stable configuration
if __name__ == "__main__":
    print("🚀 Launching Gradio app with stable configuration...")
    
    # Use simple, proven launch parameters
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=False,  # Reduce error verbosity that might conflict
            show_tips=False,   # Disable tips to reduce complexity  
            quiet=True         # Reduce logging complexity
        )
        print("✅ App launched successfully!")
    except Exception as e:
        print(f"❌ Launch failed: {e}")
        # Fallback launch with minimal parameters
        print("🔄 Trying fallback launch...")
        demo.launch() 

import gradio as gr
import os

def hello(name):
    return f"Hello {name}!"

demo = gr.Interface(fn=hello, inputs="text", outputs="text")

if __name__ == "__main__":
    # Get port from environment variable for Railway deployment
    port = int(os.environ.get("PORT", 7860))
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    ) 