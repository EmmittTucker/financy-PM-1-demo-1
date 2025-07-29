#!/usr/bin/env python3
"""
App Restart Script with Health Check Fix

This script properly stops any existing Streamlit processes and restarts
the application with proper configuration to avoid health check issues.
"""

import os
import sys
import subprocess
import time
import psutil
import signal

def kill_streamlit_processes():
    """Kill any existing Streamlit processes"""
    killed_count = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cmdline = proc.info['cmdline'] or []
                if any('streamlit' in str(arg) for arg in cmdline):
                    print(f"🔄 Killing Streamlit process: PID {proc.info['pid']}")
                    proc.kill()
                    killed_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if killed_count > 0:
        print(f"✅ Killed {killed_count} Streamlit processes")
        time.sleep(2)  # Wait for processes to fully terminate
    else:
        print("ℹ️  No existing Streamlit processes found")

def check_port_available(port=8501):
    """Check if the port is available"""
    try:
        result = subprocess.run(
            ["netstat", "-an"], 
            capture_output=True, 
            text=True, 
            shell=True
        )
        if f":{port}" in result.stdout and "LISTENING" in result.stdout:
            print(f"⚠️  Port {port} is still in use")
            return False
        else:
            print(f"✅ Port {port} is available")
            return True
    except Exception as e:
        print(f"⚠️  Could not check port status: {e}")
        return True

def start_streamlit():
    """Start Streamlit with optimized settings"""
    print("🚀 Starting AI Financial Portfolio Advisor with health check fixes...")
    
    # Set environment variables for better performance
    env = os.environ.copy()
    env.update({
        "STREAMLIT_SERVER_HEADLESS": "true",
        "STREAMLIT_SERVER_ENABLE_CORS": "false",
        "STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION": "false",
        "STREAMLIT_SERVER_MAX_MESSAGE_SIZE": "200",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false"
    })
    
    # Command to start Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.headless", "true",
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--logger.level", "info"
    ]
    
    print("📝 Starting with command:", " ".join(cmd))
    
    try:
        # Start the process
        process = subprocess.Popen(cmd, env=env)
        
        print("⏱️  Waiting for server to start...")
        time.sleep(5)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ Streamlit server started successfully!")
            print("🌐 Open your browser to: http://localhost:8501")
            print("⚠️  If you see 'Demo Mode Active', wait 3-5 minutes for model loading")
            print("🔄 Press Ctrl+C to stop the server")
            
            # Wait for the process to complete
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Stopping server...")
                process.terminate()
                process.wait()
                print("✅ Server stopped")
        else:
            print("❌ Streamlit server failed to start")
            return False
            
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("🔧 AI Financial Portfolio Advisor - Restart Script")
    print("=" * 55)
    
    print("🛑 Step 1: Stopping existing processes...")
    kill_streamlit_processes()
    
    print("\n🔍 Step 2: Checking port availability...")
    if not check_port_available():
        print("⚠️  Waiting for port to be released...")
        time.sleep(5)
        if not check_port_available():
            print("❌ Port 8501 is still in use. You may need to manually kill processes.")
    
    print("\n🚀 Step 3: Starting application...")
    success = start_streamlit()
    
    if not success:
        print("\n❌ Failed to start application")
        sys.exit(1)

if __name__ == "__main__":
    main() 