import streamlit as st
import os
import sys
import time
from typing import Optional

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

# Import configuration and model utilities
from src.config import (
    DEMO_MODE, MAX_QUESTIONS, FALLBACK_MESSAGE,
    GENDER_OPTIONS, COUNTRY_OPTIONS, INVESTMENT_BACKGROUND_OPTIONS,
    INVESTMENT_GOALS_OPTIONS, TIME_HORIZON_OPTIONS, RISK_TOLERANCE_OPTIONS,
    DEFAULT_AGE, DEFAULT_GENDER, DEFAULT_COUNTRY, DEFAULT_INVESTMENT_BACKGROUND,
    DEFAULT_INVESTMENT_GOALS, DEFAULT_TIME_HORIZON, DEFAULT_RISK_TOLERANCE
)
from src.model_utils import FinancialAdvisorModel, generate_demo_response

@st.cache_resource
def initialize_model():
    """Initialize the financial advisor model"""
    try:
        model_instance = FinancialAdvisorModel()
        success, message = model_instance.load_model()
        
        if success:
            st.success(f"✅ {message}")
            return model_instance
        else:
            st.error(f"❌ {message}")
            st.info("💡 **Fallback**: Using demo mode for this session.")
            return None
    except Exception as e:
        st.error(f"❌ Model initialization failed: {str(e)}")
        st.info("💡 **Fallback**: Using demo mode for this session.")
        return None



def show_user_profile_form():
    """Display user profile form in sidebar"""
    with st.sidebar:
        st.header("👤 Your Financial Profile")
        st.markdown("*Help me provide personalized advice*")
        
        with st.form("user_profile_form"):
            age = st.number_input(
                "Age", 
                min_value=18, 
                max_value=100, 
                value=DEFAULT_AGE,
                help="Your current age"
            )
            
            gender = st.selectbox(
                "Gender", 
                GENDER_OPTIONS, 
                index=GENDER_OPTIONS.index(DEFAULT_GENDER)
            )
            
            country = st.selectbox(
                "Country", 
                COUNTRY_OPTIONS, 
                index=COUNTRY_OPTIONS.index(DEFAULT_COUNTRY)
            )
            
            investment_background = st.selectbox(
                "Investment Experience", 
                INVESTMENT_BACKGROUND_OPTIONS,
                index=INVESTMENT_BACKGROUND_OPTIONS.index(DEFAULT_INVESTMENT_BACKGROUND)
            )
            
            investment_goals = st.selectbox(
                "Primary Investment Goal", 
                INVESTMENT_GOALS_OPTIONS,
                index=INVESTMENT_GOALS_OPTIONS.index(DEFAULT_INVESTMENT_GOALS)
            )
            
            time_horizon = st.selectbox(
                "Investment Time Horizon", 
                TIME_HORIZON_OPTIONS,
                index=TIME_HORIZON_OPTIONS.index(DEFAULT_TIME_HORIZON)
            )
            
            risk_tolerance = st.selectbox(
                "Risk Tolerance", 
                RISK_TOLERANCE_OPTIONS,
                index=RISK_TOLERANCE_OPTIONS.index(DEFAULT_RISK_TOLERANCE)
            )
            
            submitted = st.form_submit_button("💾 Save Profile", use_container_width=True)
            
            if submitted:
                st.session_state.user_profile = {
                    "age": age,
                    "gender": gender,
                    "country": country,
                    "investment_background": investment_background,
                    "investment_goals": investment_goals,
                    "time_horizon": time_horizon,
                    "risk_tolerance": risk_tolerance
                }
                st.success("✅ Profile saved!")
                st.rerun()



def main():
    st.set_page_config(
        page_title="AI Financial Portfolio Advisor - Investor Demo",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for professional appearance
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1f4e79, #2e7d32);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .model-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f4e79;
    }
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 AI Financial Portfolio Advisor</h1>
        <p><em>Powered by Fine-tuned Llama 3.1 8B | tuc111/financy-PM-1</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize model (only once)
    if "model_instance" not in st.session_state:
        st.session_state.model_instance = None
        st.session_state.model_loaded = False
        st.session_state.model_loading = False
        st.session_state.fallback_mode = False
    
    # Handle model loading state
    if not st.session_state.model_loaded and not DEMO_MODE and not st.session_state.model_loading:
        st.session_state.model_loading = True
        
        # Show loading UI first, then load model
        loading_container = st.container()
        with loading_container:
            st.info("🔄 **Loading AI Financial Advisor Model**")
            st.markdown("⏱️ **First-time loading**: The 8B model takes 3-5 minutes on CPU. Please be patient!")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Update progress indicators
            for i in range(10):
                progress_bar.progress((i + 1) * 10)
                status_text.text(f"Loading model components... ({i+1}/10)")
                time.sleep(0.1)  # Small delay to show progress
            
            # Now load the model
            try:
                model_instance = initialize_model()
                st.session_state.model_instance = model_instance
                st.session_state.model_loaded = True
                
                if model_instance is None:
                    st.session_state.fallback_mode = True
                    status_text.error("❌ Model loading failed - using demo mode")
                else:
                    st.session_state.fallback_mode = False
                    status_text.success("✅ Model loaded successfully!")
                    
                progress_bar.progress(100)
                time.sleep(1)
                
                # Clear loading UI and rerun
                loading_container.empty()
                st.rerun()
                
            except Exception as e:
                st.session_state.fallback_mode = True
                st.session_state.model_loaded = True
                status_text.error(f"❌ Error: {str(e)}")
                st.info("💡 **Fallback**: Using demo mode for this session.")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "question_count" not in st.session_state:
        st.session_state.question_count = 0
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = None
    
    # Show user profile form
    show_user_profile_form()
    
    # Model status info
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if DEMO_MODE or st.session_state.get("fallback_mode", False):
            st.markdown("""
            <div class="model-info">
                <h4>⚡ Demo Mode Active</h4>
                <p>Demonstrating AI responses with fallback system. Full model integration available for production deployment.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="model-info">
                <h4>🤖 AI Model Status: Active</h4>
                <p>Using fine-tuned Llama 3.1 8B model optimized for financial advisory services</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Questions Used", f"{st.session_state.question_count}/{MAX_QUESTIONS}")
        if st.session_state.user_profile:
            st.success("👤 Profile Set")
        else:
            st.warning("👤 Set Profile")
    
    # Chat interface
    st.markdown("### 💬 Financial Advisory Chat")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if st.session_state.question_count < MAX_QUESTIONS:
        if prompt := st.chat_input("Ask me about your financial portfolio, investments, retirement planning..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.question_count += 1
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Analyzing your financial question..."):
                    if DEMO_MODE or st.session_state.get("fallback_mode", False):
                        response = generate_demo_response(prompt)
                    else:
                        try:
                            response = st.session_state.model_instance.generate_response(
                                st.session_state.messages,
                                st.session_state.user_profile
                            )
                        except Exception as e:
                            st.error(f"Model error: {str(e)}")
                            response = generate_demo_response(prompt)
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.markdown(FALLBACK_MESSAGE)
        if st.button("🔄 Start New Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ["model_instance", "model_loaded"]:
                    del st.session_state[key]
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>AI Financial Portfolio Advisor</strong> | Investor Demo</p>
        <p><em>Powered by fine-tuned Llama 3.1 8B specialized for financial advisory services</em></p>
        <p>⚠️ This is an AI assistant. Always consult with qualified financial professionals for major decisions.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 