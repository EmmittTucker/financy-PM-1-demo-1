import streamlit as st
import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

# Import your existing functions
from config import DEMO_MODE, MAX_QUESTIONS, FALLBACK_MESSAGE

def main():
    st.set_page_config(
        page_title="AI Financial Portfolio Advisor",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 AI Financial Portfolio Advisor")
    st.markdown("*Powered by Fine-tuned Llama 3.1 8B*")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about your financial portfolio..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            if DEMO_MODE:
                response = generate_demo_response(prompt)
            else:
                response = "I'm currently in demo mode. The full model is temporarily unavailable."
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

def generate_demo_response(prompt):
    """Generate a demo response for financial questions"""
    demo_responses = {
        "portfolio": "Based on your risk profile, I'd recommend a diversified portfolio with 60% stocks, 30% bonds, and 10% alternative investments.",
        "investment": "For long-term growth, consider low-cost index funds like S&P 500 ETFs, which historically return 7-10% annually.",
        "retirement": "For retirement planning, maximize your 401(k) match and consider a Roth IRA for tax-free growth.",
        "risk": "Your risk tolerance should align with your time horizon. Younger investors can typically handle more volatility.",
        "diversification": "Diversification across asset classes, sectors, and geographies helps reduce overall portfolio risk."
    }
    
    prompt_lower = prompt.lower()
    for key, response in demo_responses.items():
        if key in prompt_lower:
            return f"💡 **Demo Response**: {response}\n\n*This is a demonstration. The full AI model provides personalized advice based on your specific financial situation.*"
    
    return f"💡 **Demo Response**: Thank you for your question about '{prompt}'. In demo mode, I can provide general financial guidance. For personalized advice, please wait for the full model to be available.\n\n*Key tip: Always consult with a qualified financial advisor for major financial decisions.*"

if __name__ == "__main__":
    main() 