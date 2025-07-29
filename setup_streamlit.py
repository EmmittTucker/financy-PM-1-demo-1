#!/usr/bin/env python3
"""
Streamlit Setup Automation Script

This script automates the initial setup process for the AI Financial Portfolio Advisor
Streamlit application, including directory creation, configuration files, and environment setup.
"""

import os
import sys
from pathlib import Path
import subprocess

def create_directory_structure():
    """Create necessary directories for Streamlit"""
    directories = [
        ".streamlit",
        "pages",
        "static",
        "static/images",
        "static/css"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_streamlit_config():
    """Create Streamlit configuration file"""
    config_content = """[global]
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

[logger]
level = "info"
messageFormat = "%(asctime)s %(message)s"
"""
    
    config_file = Path(".streamlit/config.toml")
    if not config_file.exists():
        config_file.write_text(config_content)
        print("✅ Created Streamlit config file: .streamlit/config.toml")
    else:
        print("ℹ️  Streamlit config file already exists")

def create_secrets_template():
    """Create a template for Streamlit secrets"""
    secrets_template = """# Streamlit Secrets Configuration
# Copy this file to secrets.toml and add your actual values

HF_TOKEN = "your_huggingface_token_here"

# Optional: Add other secrets as needed
[database]
username = "your_db_user"
password = "your_db_password"

[external_apis]
openai_key = "your_openai_key"
other_api_key = "your_other_api_key"
"""
    
    template_file = Path(".streamlit/secrets.toml.template")
    template_file.write_text(secrets_template)
    print("✅ Created secrets template: .streamlit/secrets.toml.template")
    print("ℹ️  Copy this to .streamlit/secrets.toml and add your actual tokens")

def create_environment_file():
    """Create environment file template"""
    env_content = """# Environment Variables for AI Financial Portfolio Advisor
# Copy this to .env and update with your values

HF_TOKEN=your_huggingface_token_here
CUDA_VISIBLE_DEVICES=0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLE_CORS=false

# Optional: Set these for production deployment
# STREAMLIT_SERVER_PORT=8501
# STREAMLIT_SERVER_ADDRESS=0.0.0.0
"""
    
    env_file = Path(".env.template")
    env_file.write_text(env_content)
    print("✅ Created environment template: .env.template")
    print("ℹ️  Copy this to .env and add your actual values")

def check_streamlit_installation():
    """Check if Streamlit is installed and get version"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "streamlit", "--version"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        print(f"✅ Streamlit is installed: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Streamlit is not installed")
        return False

def install_streamlit():
    """Install Streamlit if not present"""
    print("Installing Streamlit...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "streamlit"
        ], check=True)
        print("✅ Streamlit installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install Streamlit")
        return False

def create_quick_start_script():
    """Create a quick start script"""
    if os.name == 'nt':  # Windows
        script_content = """@echo off
echo Starting AI Financial Portfolio Advisor...
echo.
echo Make sure you have set your HF_TOKEN in .env or .streamlit/secrets.toml
echo.
python -m streamlit run streamlit_app.py
pause
"""
        script_file = Path("start_app.bat")
    else:  # Unix/Linux/macOS
        script_content = """#!/bin/bash
echo "Starting AI Financial Portfolio Advisor..."
echo ""
echo "Make sure you have set your HF_TOKEN in .env or .streamlit/secrets.toml"
echo ""
python -m streamlit run streamlit_app.py
"""
        script_file = Path("start_app.sh")
    
    script_file.write_text(script_content)
    if not os.name == 'nt':
        os.chmod(script_file, 0o755)
    
    print(f"✅ Created quick start script: {script_file}")

def update_gitignore():
    """Update .gitignore with Streamlit-specific entries"""
    gitignore_additions = """
# Streamlit secrets and config
.streamlit/secrets.toml
.streamlit/credentials.toml

# Environment files
.env
.env.local
.env.production
"""
    
    gitignore_file = Path(".gitignore")
    if gitignore_file.exists():
        current_content = gitignore_file.read_text()
        if ".streamlit/secrets.toml" not in current_content:
            with open(gitignore_file, "a") as f:
                f.write(gitignore_additions)
            print("✅ Updated .gitignore with Streamlit entries")
        else:
            print("ℹ️  .gitignore already contains Streamlit entries")
    else:
        gitignore_file.write_text(gitignore_additions.strip())
        print("✅ Created .gitignore with Streamlit entries")

def main():
    """Main setup function"""
    print("🚀 AI Financial Portfolio Advisor - Streamlit Setup")
    print("=" * 55)
    
    # Check if we're in the right directory
    if not Path("streamlit_app.py").exists():
        print("❌ streamlit_app.py not found. Please run this script from the project root.")
        sys.exit(1)
    
    print("📁 Creating directory structure...")
    create_directory_structure()
    
    print("\n⚙️  Creating configuration files...")
    create_streamlit_config()
    create_secrets_template()
    create_environment_file()
    
    print("\n🔍 Checking Streamlit installation...")
    if not check_streamlit_installation():
        install_streamlit()
    
    print("\n📝 Creating utility scripts...")
    create_quick_start_script()
    update_gitignore()
    
    print("\n" + "=" * 55)
    print("✅ Streamlit setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Copy .env.template to .env and add your HF_TOKEN")
    print("2. Copy .streamlit/secrets.toml.template to .streamlit/secrets.toml")
    print("3. Add your Hugging Face token to the secrets file")
    print("4. Run: python run_demo.py")
    print("\n📚 For detailed instructions, see: STREAMLIT_SETUP_GUIDE.md")

if __name__ == "__main__":
    main() 