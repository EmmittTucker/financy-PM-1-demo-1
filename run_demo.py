#!/usr/bin/env python3
"""
AI Financial Portfolio Advisor - Demo Launcher

Professional demo launcher script for investor presentations.
Handles environment setup, dependency checking, and application startup.
"""

import os
import sys
import subprocess
import time
import platform
from pathlib import Path

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header():
    """Print application header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}")
    print("🚀 AI Financial Portfolio Advisor - Investor Demo")
    print("=" * 50)
    print("Powered by Fine-tuned Llama 3.1 8B | tuc111/financy-PM-1")
    print(f"{Colors.ENDC}")

def print_status(message, status="info"):
    """Print colored status message"""
    color = {
        "info": Colors.OKBLUE,
        "success": Colors.OKGREEN,
        "warning": Colors.WARNING,
        "error": Colors.FAIL
    }.get(status, Colors.OKBLUE)
    
    icon = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌"
    }.get(status, "ℹ️")
    
    print(f"{color}{icon} {message}{Colors.ENDC}")

def check_python_version():
    """Check if Python version meets requirements"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print_status("Python 3.10+ required. Current version: {}.{}.{}".format(
            version.major, version.minor, version.micro), "error")
        return False
    
    print_status(f"Python version: {version.major}.{version.minor}.{version.micro} ✓", "success")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        "streamlit",
        "torch",
        "transformers",
        "accelerate",
        "huggingface_hub",
        "bitsandbytes"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print_status(f"Package '{package}' found ✓", "success")
        except ImportError:
            missing_packages.append(package)
            print_status(f"Package '{package}' missing", "error")
    
    if missing_packages:
        print_status("Installing missing dependencies...", "warning")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ])
            print_status("Dependencies installed successfully", "success")
        except subprocess.CalledProcessError:
            print_status("Failed to install dependencies", "error")
            return False
    
    return True

def check_gpu_availability():
    """Check GPU availability for model acceleration"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print_status(f"GPU available: {gpu_name} ({gpu_count} device(s))", "success")
            return True
        else:
            print_status("No GPU detected. Using CPU (slower inference)", "warning")
            return False
    except ImportError:
        print_status("PyTorch not available for GPU check", "warning")
        return False

def check_hf_token():
    """Check Hugging Face token availability"""
    hf_token = os.getenv("HF_TOKEN")
    secrets_file = Path(".streamlit/secrets.toml")
    
    if hf_token:
        print_status("HF_TOKEN environment variable found ✓", "success")
        return True
    elif secrets_file.exists():
        print_status("Streamlit secrets file found ✓", "success")
        return True
    else:
        print_status("No HF_TOKEN found. Demo mode will be used", "warning")
        print_status("For full model access, set HF_TOKEN environment variable", "info")
        return False

def run_tests():
    """Run the test suite"""
    print_status("Running test suite...", "info")
    try:
        result = subprocess.run([
            sys.executable, "tests/run_all_tests.py"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print_status("All tests passed ✓", "success")
            return True
        else:
            print_status("Some tests failed", "warning")
            print(result.stdout)
            return False
    except subprocess.TimeoutExpired:
        print_status("Tests timed out", "warning")
        return False
    except Exception as e:
        print_status(f"Test execution error: {e}", "error")
        return False

def launch_streamlit():
    """Launch the Streamlit application"""
    print_status("Launching AI Financial Portfolio Advisor...", "info")
    
    # Set environment variables for optimal performance
    env = os.environ.copy()
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_SERVER_ENABLE_CORS"] = "false"
    env["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "false"
    
    try:
        # Launch Streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app.py"]
        
        # Add development flags if needed
        if "--dev" in sys.argv:
            cmd.extend(["--server.runOnSave", "true"])
            cmd.extend(["--server.port", "8501"])
        
        print_status("Starting Streamlit server...", "info")
        print_status("Application will open in your default browser", "info")
        print_status("Press Ctrl+C to stop the server", "info")
        
        subprocess.run(cmd, env=env)
        
    except KeyboardInterrupt:
        print_status("\nApplication stopped by user", "info")
    except Exception as e:
        print_status(f"Failed to launch application: {e}", "error")

def main():
    """Main demo launcher function"""
    print_header()
    
    # System requirements check
    print_status("Checking system requirements...", "info")
    
    if not check_python_version():
        sys.exit(1)
    
    if not check_dependencies():
        sys.exit(1)
    
    # Optional checks
    check_gpu_availability()
    check_hf_token()
    
    # Run tests if requested
    if "--test" in sys.argv or "--tests" in sys.argv:
        if not run_tests():
            print_status("Consider fixing test issues before demo", "warning")
            response = input(f"{Colors.WARNING}Continue anyway? (y/N): {Colors.ENDC}")
            if response.lower() != 'y':
                sys.exit(1)
    
    # Launch application
    print_status("All checks completed. Ready for demo!", "success")
    time.sleep(1)
    
    launch_streamlit()

def show_help():
    """Show help message"""
    print(f"""
{Colors.BOLD}AI Financial Portfolio Advisor - Demo Launcher{Colors.ENDC}

Usage:
    python run_demo.py [options]

Options:
    --test, --tests    Run test suite before launching
    --dev             Launch in development mode
    --help, -h        Show this help message

Examples:
    python run_demo.py                    # Quick demo launch
    python run_demo.py --test             # Run tests then launch
    python run_demo.py --dev              # Development mode
    
For investor demonstrations:
    1. Ensure HF_TOKEN is set for full model access
    2. Run with --test flag to verify system health
    3. Present the professional interface to stakeholders

Support: support@grandmasboylabs.com
""")

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help()
    else:
        main() 