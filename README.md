---
title: AI Financial Portfolio Advisor
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.20.0
app_file: app.py
python_version: 3.10
disable_embedding: false
fullWidth: false
---

# 🚀 AI Financial Portfolio Advisor - Investor Demo

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)](https://streamlit.io)
[![Model](https://img.shields.io/badge/Model-Llama%203.1%208B-green.svg)](https://huggingface.co/tuc111/financy-PM-1)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Executive Summary

**AI Financial Portfolio Advisor** is a cutting-edge financial advisory platform powered by our custom fine-tuned **Llama 3.1 8B model** (`tuc111/financy-PM-1`). This sophisticated AI assistant provides personalized investment advice, portfolio optimization, and financial planning guidance with institutional-grade accuracy.

### 🏆 Key Value Propositions

- **🤖 Fine-tuned AI Model**: Specialized Llama 3.1 8B trained on financial advisory datasets
- **🎯 Personalized Advice**: User profiling system for tailored recommendations
- **⚡ Real-time Inference**: Optimized model deployment with 4-bit quantization
- **📱 Modern Interface**: Professional Streamlit web application
- **🔒 Enterprise-Ready**: Comprehensive testing suite and professional architecture

## 🛠 Technical Architecture

### Core Components

```
📁 financy-PM-1-demo-1/
├── 📁 src/                     # Core application modules
│   ├── config.py              # Configuration management
│   ├── model_utils.py         # AI model utilities
│   └── __init__.py           # Package initialization
├── 📁 tests/                  # Comprehensive test suite
│   ├── test_config.py        # Configuration tests
│   ├── test_model_utils.py   # Model utility tests
│   ├── run_all_tests.py      # CI/CD test runner
│   └── __init__.py          # Test package
├── streamlit_app.py          # Main application interface
├── run_demo.py              # Quick demo launcher
├── requirements.txt         # Dependencies
└── README.md               # Documentation
```

### Model Specifications

- **Base Model**: Meta Llama 3.1 8B
- **Fine-tuning**: Specialized for financial advisory tasks
- **Model ID**: `tuc111/financy-PM-1`
- **Deployment**: 4-bit quantization with BitsAndBytes
- **Memory Optimization**: GPU-efficient inference pipeline

## 🚀 Quick Start Guide

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM
- Hugging Face account (for private model access)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/EmmittTucker/financy-PM-1-demo-1.git
   cd financy-PM-1-demo-1
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Hugging Face authentication**
   ```bash
   # Set environment variable
   export HF_TOKEN="your_huggingface_token"
   
   # Or use Streamlit secrets (for deployment)
   echo 'HF_TOKEN = "your_token"' > .streamlit/secrets.toml
   ```

4. **Setup Streamlit (Automated)**
   ```bash
   # Run automated setup
   python setup_streamlit.py
   
   # Add your HF_TOKEN to .streamlit/secrets.toml
   ```

5. **Launch the application**
   ```bash
   # Quick demo launcher
   python run_demo.py
   
   # Or direct Streamlit launch
   streamlit run streamlit_app.py
   ```

## 💡 Product Features

### 🎯 Personalized Financial Profiling
- **Comprehensive User Assessment**: Age, investment experience, risk tolerance
- **Goal-Based Planning**: Retirement, wealth building, income generation
- **Geographic Customization**: Country-specific financial advice
- **Time Horizon Analysis**: Short-term to long-term investment strategies

### 🤖 AI-Powered Advisory
- **Fine-tuned Responses**: Trained specifically for financial advisory tasks
- **Context-Aware Conversations**: Maintains conversation history and user profile
- **Professional Expertise**: Covers portfolio allocation, risk management, tax optimization
- **Real-time Inference**: Sub-second response times with optimized deployment

### 📊 Advanced Portfolio Guidance
- **Asset Allocation Strategies**: Diversified portfolio recommendations
- **Risk Assessment**: Personalized risk tolerance evaluation
- **Investment Vehicles**: ETFs, mutual funds, individual stocks, bonds
- **Tax Optimization**: 401(k), IRA, and tax-loss harvesting strategies

## 🧪 Quality Assurance

### Comprehensive Testing Suite

```bash
# Run all tests
python tests/run_all_tests.py

# Run specific test modules
python tests/run_all_tests.py test_config
python tests/run_all_tests.py test_model_utils
```

### Test Coverage
- ✅ Configuration validation
- ✅ Model loading and inference
- ✅ User profile handling
- ✅ Demo response generation
- ✅ Error handling and fallbacks

## 🎯 Investor Demo Highlights

### 🔥 Live Demonstration Features

1. **Model Performance**
   - Real-time financial advice generation
   - Contextual conversation handling
   - Professional-grade responses

2. **User Experience**
   - Intuitive profile setup
   - Clean, modern interface
   - Mobile-responsive design

3. **Technical Excellence**
   - Professional code architecture
   - Comprehensive error handling
   - Production-ready deployment

### 📈 Business Metrics

- **Response Accuracy**: Fine-tuned for financial domain
- **User Engagement**: Interactive profiling system
- **Scalability**: GPU-optimized inference pipeline
- **Compliance**: Built-in disclaimers and professional standards

## 🛡 Security & Compliance

- **Data Privacy**: Local processing, no data retention
- **Financial Disclaimers**: Clear AI assistant limitations
- **Professional Standards**: Encourages consultation with qualified advisors
- **Secure Deployment**: Environment-based authentication

## 🔧 Development & Deployment

### Local Development
```bash
# Install development dependencies
pip install -r requirements.txt

# Setup Streamlit configuration
python setup_streamlit.py

# Run tests
python tests/run_all_tests.py

# Launch development server
streamlit run streamlit_app.py --server.runOnSave true
```

### 📖 Detailed Setup Guide
For comprehensive Streamlit setup instructions, see: **[STREAMLIT_SETUP_GUIDE.md](STREAMLIT_SETUP_GUIDE.md)**

### Production Deployment Options

1. **Streamlit Cloud**: Direct GitHub integration
2. **Docker Containers**: Containerized deployment
3. **Cloud Platforms**: AWS, GCP, Azure support
4. **On-Premise**: Private infrastructure deployment

## 📊 Performance Benchmarks

- **Model Load Time**: ~30-60 seconds (first load)
- **Response Generation**: <5 seconds average
- **Memory Usage**: ~8GB GPU memory (quantized)
- **Concurrent Users**: Scalable with load balancing

## 🎓 Model Training Details

Our fine-tuned model (`tuc111/financy-PM-1`) was trained on:
- Financial advisory conversations
- Investment strategy documents
- Portfolio optimization case studies
- Risk assessment frameworks
- Regulatory compliance guidelines

## 🚀 Future Roadmap

- **Multi-language Support**: International market expansion
- **Advanced Analytics**: Portfolio backtesting and simulation
- **API Integration**: Third-party financial data services
- **Mobile Application**: Native iOS and Android apps
- **Institutional Features**: Wealth management platform integration

## 📞 Support & Contact

For technical questions or investor inquiries:
- **Email**: support@grandmasboylabs.com
- **Website**: [grandmasboylabs.com](https://grandmasboylabs.com)
- **Demo**: Available upon request

---

### ⚠️ Important Disclaimer

This AI Financial Portfolio Advisor is designed to provide general financial guidance and should not replace professional financial advice. Users should always consult with qualified financial advisors for major investment decisions. The AI model's responses are based on training data and may not reflect current market conditions or individual circumstances.

**© 2025 Financial AI Solutions | All Rights Reserved**

