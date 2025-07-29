"""
AI Financial Portfolio Advisor

A professional financial advisory application powered by fine-tuned Llama 3.1 8B model.
Completely redesigned to prevent health check timeouts during model loading.
"""

import streamlit as st
import time
from datetime import datetime
from src.config import DEMO_MODE, APP_TITLE, APP_DESCRIPTION, FORM_OPTIONS
from src.model_utils import get_model_loader, generate_demo_response

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .status-container {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f0f8f0;
    }
    .loading-container {
        border: 2px solid #FF9800;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #fff8e1;
    }
    .error-container {
        border: 2px solid #f44336;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #ffebee;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {}
    
    if "model_loader" not in st.session_state:
        st.session_state.model_loader = None
    
    if "model_check_time" not in st.session_state:
        st.session_state.model_check_time = 0
    
    if "show_profile_form" not in st.session_state:
        st.session_state.show_profile_form = False


def get_model_status():
    """Get model loading status with caching to prevent excessive checks."""
    current_time = time.time()
    
    # Only check status every 5 seconds to prevent blocking (increased from 2 seconds)
    if current_time - st.session_state.model_check_time < 5:
        return st.session_state.get("last_model_status", {
            'is_loading': False,
            'is_loaded': False,
            'progress': 0,
            'message': 'Initializing...',
            'error': None
        })
    
    # Lazy initialization of model loader (non-blocking)
    if st.session_state.model_loader is None:
        try:
            st.session_state.model_loader = get_model_loader()
        except Exception as e:
            return {
                'is_loading': False,
                'is_loaded': False,
                'progress': 0,
                'message': 'Initialization failed',
                'error': str(e)
            }
    
    try:
        status = st.session_state.model_loader.get_status()
        st.session_state.last_model_status = status
        st.session_state.model_check_time = current_time
        return status
    except Exception as e:
        return {
            'is_loading': False,
            'is_loaded': False,
            'progress': 0,
            'message': 'Status check failed',
            'error': str(e)
        }


def start_model_loading():
    """Start model loading in background if not already started."""
    try:
        if st.session_state.model_loader is None:
            st.session_state.model_loader = get_model_loader()
        
        if not st.session_state.model_loader.is_loading and not st.session_state.model_loader.is_loaded:
            st.session_state.model_loader.start_loading()
    except Exception as e:
        st.error(f"Failed to start model loading: {e}")


def show_model_loading_status():
    """Display model loading status with progress indicators."""
    try:
        status = get_model_status()
        
        if status['is_loading']:
            st.markdown('<div class="loading-container">', unsafe_allow_html=True)
            st.info("🔄 **AI Model Loading**")
            st.markdown(f"**Status**: {status['message']}")
            
            # Progress bar
            progress = max(0, min(100, status['progress'])) / 100.0
            st.progress(progress)
            st.markdown(f"**Progress**: {status['progress']:.0f}%")
            
            if status['progress'] < 30:
                st.markdown("⏱️ **First-time loading**: This may take 3-5 minutes on CPU. Please be patient!")
            elif status['progress'] < 70:
                st.markdown("🧠 **Loading neural network**: The 8B parameter model is being initialized...")
            else:
                st.markdown("🔧 **Finalizing setup**: Almost ready!")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Auto-refresh every 5 seconds during loading (reduced frequency)
            time.sleep(0.1)
            st.rerun()
            
        elif status['error']:
            st.markdown('<div class="error-container">', unsafe_allow_html=True)
            st.error(f"❌ **Model Loading Failed**: {status['error']}")
            st.info("💡 **Fallback**: Using demo mode for this session.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        elif status['is_loaded']:
            st.markdown('<div class="status-container">', unsafe_allow_html=True)
            st.success("✅ **AI Model Ready**: Llama 3.1 8B Financial Advisor is online!")
            st.markdown('</div>', unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error displaying model status: {e}")
        # Fall back to demo mode on any error
        st.info("💡 **Fallback**: Using demo mode due to loading error.")


def show_user_profile_form():
    """Display user profile form in sidebar."""
    with st.sidebar:
        st.header("👤 User Profile")
        
        with st.form("user_profile_form"):
            experience = st.selectbox(
                "Investment Experience",
                FORM_OPTIONS["experience_levels"],
                index=FORM_OPTIONS["experience_levels"].index(st.session_state.user_profile.get("experience_level", "Beginner"))
            )
            
            goals = st.multiselect(
                "Investment Goals",
                FORM_OPTIONS["investment_goals"],
                default=st.session_state.user_profile.get("investment_goals", [])
            )
            
            risk_tolerance = st.selectbox(
                "Risk Tolerance",
                FORM_OPTIONS["risk_tolerances"],
                index=FORM_OPTIONS["risk_tolerances"].index(st.session_state.user_profile.get("risk_tolerance", "Moderate"))
            )
            
            time_horizon = st.selectbox(
                "Investment Time Horizon",
                FORM_OPTIONS["time_horizons"],
                index=FORM_OPTIONS["time_horizons"].index(st.session_state.user_profile.get("time_horizon", "5-10 years"))
            )
            
            submitted = st.form_submit_button("💾 Save Profile")
            
            if submitted:
                st.session_state.user_profile = {
                    "experience_level": experience,
                    "investment_goals": goals,
                    "risk_tolerance": risk_tolerance,
                    "time_horizon": time_horizon
                }
                st.success("✅ Profile saved!")
                st.rerun()


def generate_response(prompt: str) -> str:
    """Generate response using model or demo mode."""
    if DEMO_MODE:
        return generate_demo_response(prompt)
    
    status = get_model_status()
    
    if status['is_loaded'] and st.session_state.model_loader:
        return st.session_state.model_loader.generate_response(
            st.session_state.messages,
            st.session_state.user_profile
        )
    else:
        return generate_demo_response(prompt)


def main():
    """Main application logic with non-blocking model loading."""
    initialize_session_state()
    
    # Header
    st.title(APP_TITLE)
    st.markdown(APP_DESCRIPTION)
    
    # Sidebar
    show_user_profile_form()
    
    with st.sidebar:
        st.markdown("---")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 🔧 Status")
        
        if DEMO_MODE:
            st.info("📝 **Demo Mode Active**")
            st.markdown("Using simulated responses for demonstration.")
        else:
            # Start model loading if needed (non-blocking)
            start_model_loading()
            
            # Show current status
            status = get_model_status()
            if status['is_loaded']:
                st.success("🤖 **AI Model**: Online")
            elif status['is_loading']:
                st.warning("⏳ **AI Model**: Loading...")
            elif status['error']:
                st.error("❌ **AI Model**: Error")
            else:
                st.info("💤 **AI Model**: Initializing...")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Show model loading status if applicable
        if not DEMO_MODE:
            show_model_loading_status()
        
        # Chat interface
        st.subheader("💬 Financial Advisory Chat")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask your financial question..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = generate_response(prompt)
                st.markdown(response)
            
            # Add assistant message
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    with col2:
        st.subheader("📊 Quick Stats")
        
        # User profile summary
        if st.session_state.user_profile:
            st.markdown("**Profile Set** ✅")
            st.markdown(f"**Experience**: {st.session_state.user_profile.get('experience_level', 'Not set')}")
            st.markdown(f"**Risk**: {st.session_state.user_profile.get('risk_tolerance', 'Not set')}")
        else:
            st.markdown("**Profile**: Not configured")
            if st.button("⚙️ Set Profile"):
                st.session_state.show_profile_form = True
        
        # Chat stats
        st.markdown(f"**Messages**: {len(st.session_state.messages)}")
        
        # Session info
        st.markdown("---")
        st.markdown("### ℹ️ Session Info")
        st.markdown(f"**Started**: {datetime.now().strftime('%H:%M')}")
        
        if not DEMO_MODE:
            status = get_model_status()
            if status['is_loaded']:
                st.markdown("**Model**: Ready 🟢")
            elif status['is_loading']:
                st.markdown("**Model**: Loading 🟡")
            else:
                st.markdown("**Model**: Initializing 🔵")


if __name__ == "__main__":
    main() 