# 🚀 Streamlit Setup Guide - AI Financial Portfolio Advisor

Complete guide for setting up and deploying the AI Financial Portfolio Advisor using Streamlit.

## 📋 Table of Contents
- [Prerequisites](#prerequisites)
- [Local Development Setup](#local-development-setup)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Deployment Options](#deployment-options)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

## 🔧 Prerequisites

### System Requirements
- **Python**: 3.10 or higher
- **RAM**: 16GB minimum (32GB recommended for GPU inference)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)
- **Storage**: 10GB+ free space for model files

### Required Accounts
- **Hugging Face Account**: For accessing the fine-tuned model `tuc111/financy-PM-1`
- **Git**: For cloning the repository

## 🏗️ Local Development Setup

### 1. Clone the Repository
```bash
git clone https://github.com/EmmittTucker/financy-PM-1-demo-1.git
cd financy-PM-1-demo-1
```

### 2. Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Verify Streamlit installation
streamlit --version
```

### 4. Verify Installation
```bash
# Run the comprehensive setup check
python run_demo.py --test
```

## ⚙️ Configuration

### 1. Environment Variables

Create a `.env` file in the project root:

```bash
# .env file
HF_TOKEN=your_huggingface_token_here
CUDA_VISIBLE_DEVICES=0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLE_CORS=false
```

### 2. Streamlit Configuration

Create `.streamlit/config.toml`:

```toml
[global]
developmentMode = false
showWarningOnDirectExecution = false

[server]
headless = true
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 10
port = 8501
address = "0.0.0.0"
baseUrlPath = ""

[browser]
serverAddress = "localhost"
gatherUsageStats = false

[theme]
primaryColor = "#1f4e79"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[runner]
magicEnabled = true
installTracer = false
fixMatplotlib = true
```

### 3. Secrets Management

Create `.streamlit/secrets.toml` for sensitive data:

```toml
# .streamlit/secrets.toml
HF_TOKEN = "your_huggingface_token_here"

# Optional: Database credentials, API keys, etc.
[database]
username = "your_db_user"
password = "your_db_password"

[external_apis]
openai_key = "your_openai_key"
```

**⚠️ Important**: Add `.streamlit/secrets.toml` to your `.gitignore` file!

### 4. Model Configuration

Edit `src/config.py` for your specific setup:

```python
# Model Configuration
MODEL_NAME = "tuc111/financy-PM-1"
DEMO_MODE = False  # Set to True for demo without model loading
USE_AUTH_TOKEN = True

# Performance Settings
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
CONTEXT_WINDOW = 20

# UI Configuration
MAX_QUESTIONS = 20
CHATBOT_HEIGHT = 500
```

## 🚀 Running the Application

### Option 1: Using the Demo Launcher (Recommended)
```bash
# Basic launch
python run_demo.py

# With system tests
python run_demo.py --test

# Development mode with auto-reload
python run_demo.py --dev
```

### Option 2: Direct Streamlit Command
```bash
# Basic run
streamlit run streamlit_app.py

# With specific port
streamlit run streamlit_app.py --server.port 8502

# Development mode
streamlit run streamlit_app.py --server.runOnSave true
```

### Option 3: Production Mode
```bash
# Set production environment
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false

# Run with production settings
streamlit run streamlit_app.py --server.headless true
```

## 🌐 Deployment Options

### 1. Streamlit Community Cloud

#### Setup Steps:
1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set secrets in the Streamlit Cloud dashboard
5. Deploy!

#### Required Secrets:
```
HF_TOKEN = "your_huggingface_token"
```

#### streamlit_app.py Requirements:
- Ensure the main file is named `streamlit_app.py`
- All dependencies must be in `requirements.txt`

### 2. Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Set environment variables
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Run the application
CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0"]
```

Build and run:
```bash
# Build the image
docker build -t financial-advisor .

# Run the container
docker run -p 8501:8501 -e HF_TOKEN=your_token financial-advisor
```

### 3. Cloud Platform Deployment

#### AWS EC2:
```bash
# Launch EC2 instance (Ubuntu 20.04+)
# SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install dependencies
sudo apt update
sudo apt install python3-pip git

# Clone and setup
git clone https://github.com/EmmittTucker/financy-PM-1-demo-1.git
cd financy-PM-1-demo-1
pip3 install -r requirements.txt

# Set environment variables
export HF_TOKEN=your_token

# Run application
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

#### Google Cloud Platform:
```yaml
# app.yaml for App Engine
runtime: python310

env_variables:
  HF_TOKEN: "your_huggingface_token"
  STREAMLIT_SERVER_HEADLESS: "true"

automatic_scaling:
  min_instances: 1
  max_instances: 10
```

Deploy:
```bash
gcloud app deploy
```

## ⚡ Performance Optimization

### 1. Memory Management

Add to `src/model_utils.py`:

```python
import gc
import torch

def clear_gpu_memory():
    """Clear GPU memory after model operations"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_model_cached():
    """Cached model loading with memory management"""
    model_instance = FinancialAdvisorModel()
    success, message = model_instance.load_model()
    return model_instance if success else None
```

### 2. Streamlit Optimization

Add to your main app:

```python
# Optimize page config
st.set_page_config(
    page_title="AI Financial Advisor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "AI Financial Portfolio Advisor v1.0"
    }
)

# Enable caching for expensive operations
@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_user_recommendations(user_profile):
    # Expensive computation here
    pass
```

### 3. Model Inference Optimization

```python
# Use torch.no_grad() for inference
with torch.no_grad():
    outputs = model.generate(...)

# Implement batch processing for multiple requests
def batch_generate(prompts, batch_size=4):
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        # Process batch
        results.extend(process_batch(batch))
    return results
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. Model Loading Errors
```python
# Issue: CUDA out of memory
# Solution: Reduce model precision or use CPU
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Use 8-bit instead of 4-bit
    device_map="auto"
)
```

#### 2. Streamlit Connection Issues
```bash
# Issue: Browser doesn't open automatically
# Solution: Manually navigate to http://localhost:8501

# Issue: Port already in use
streamlit run streamlit_app.py --server.port 8502
```

#### 3. Authentication Problems
```python
# Issue: HF_TOKEN not recognized
# Solution: Verify token format and permissions
import os
print(f"Token exists: {bool(os.getenv('HF_TOKEN'))}")
print(f"Token length: {len(os.getenv('HF_TOKEN', ''))}")
```

#### 4. Dependency Conflicts
```bash
# Issue: Package version conflicts
# Solution: Create fresh virtual environment
python -m venv fresh_env
fresh_env\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Add to streamlit_app.py
if st.checkbox("Debug Mode"):
    st.write("Session State:", st.session_state)
    st.write("Model Info:", st.session_state.get('model_instance', {}).get_model_info())
```

## 🔧 Advanced Configuration

### 1. Custom Themes

Create custom CSS:

```python
def load_custom_css():
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79, #2e7d32);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 4px solid #1f4e79;
    }
    
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
```

### 2. Multi-Page Apps

Organize into multiple pages:

```python
# pages/1_🏠_Home.py
import streamlit as st
st.title("Home Page")

# pages/2_📊_Analytics.py  
import streamlit as st
st.title("Analytics Dashboard")

# pages/3_⚙️_Settings.py
import streamlit as st
st.title("Application Settings")
```

### 3. Session Management

Implement user sessions:

```python
def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'user_id': None,
        'conversation_history': [],
        'user_profile': None,
        'model_instance': None,
        'question_count': 0
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def save_session(user_id):
    """Save session to persistent storage"""
    session_data = {
        'user_profile': st.session_state.user_profile,
        'conversation_history': st.session_state.messages,
        'timestamp': datetime.now().isoformat()
    }
    # Save to database or file
```

### 4. Monitoring and Analytics

Add application monitoring:

```python
import time
from datetime import datetime

def log_user_interaction(event_type, details):
    """Log user interactions for analytics"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'event_type': event_type,
        'details': details,
        'session_id': st.session_state.get('session_id'),
        'user_agent': st._config.get_option('browser.serverAddress')
    }
    # Log to file or database
    
# Usage
log_user_interaction('question_asked', {'question': prompt, 'response_time': response_time})
```

## 📞 Support and Resources

### Documentation Links
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Community Cloud](https://share.streamlit.io/)
- [Streamlit Components](https://streamlit.io/components)

### Community Resources
- [Streamlit Forum](https://discuss.streamlit.io/)
- [GitHub Issues](https://github.com/EmmittTucker/financy-PM-1-demo-1/issues)

### Contact
- **Technical Support**: support@grandmasboylabs.com
- **Documentation**: [Project Wiki](https://github.com/EmmittTucker/financy-PM-1-demo-1/wiki)

---

**© 2025 Financial AI Solutions | Streamlit Setup Guide v1.0** 