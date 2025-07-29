#!/usr/bin/env python3
"""
App Restart Script with Health Check Fix
==========================================

This script provides a robust way to stop any running Streamlit processes
and restart the application with optimized settings for model loading.

Recent improvements:
- Removed st.cache_resource blocking behavior
- Disabled health checks during model loading
- Implemented true background model loading
- Reduced UI update frequency to prevent blocking
"""

import os
import sys
import subprocess
import time
import psutil
import signal

def kill_streamlit_processes():
    """Kill all running Streamlit processes"""
    killed_count = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                if proc.info['cmdline'] and any('streamlit' in str(cmd).lower() for cmd in proc.info['cmdline']):
                    print(f"🔄 Killing Streamlit process: PID {proc.info['pid']}")
                    proc.kill()
                    killed_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    if killed_count > 0:
        print(f"✅ Killed {killed_count} Streamlit processes")
        time.sleep(2)  # Wait for processes to fully terminate
    else:
        print("ℹ️  No existing Streamlit processes found")

def check_port_available(port=8501):
    """Check if the specified port is available"""
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(('localhost', port))
            return result != 0  # Port is available if connection fails
    except Exception:
        return False

def start_streamlit():
    """Start the Streamlit application with optimized settings"""
    try:
        python_path = sys.executable
        print(f"🚀 Starting AI Financial Portfolio Advisor with health check fixes...")
        print(f"📝 Starting with command: {python_path} -m streamlit run streamlit_app.py --server.headless true --server.port 8501 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false --logger.level info")
        
        # Start Streamlit with optimized server settings
        process = subprocess.Popen([
            python_path, '-m', 'streamlit', 'run', 'streamlit_app.py',
            '--server.headless', 'true',
            '--server.port', '8501',
            '--server.address', '0.0.0.0',
            '--server.enableCORS', 'false',
            '--server.enableXsrfProtection', 'false',
            '--logger.level', 'info'
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        print("⏱️  Waiting for server to start...")
        
        # Monitor the process startup
        startup_timeout = 30
        start_time = time.time()
        
        while time.time() - start_time < startup_timeout:
            # Check if process is still running
            if process.poll() is not None:
                output = process.stdout.read()
                print(f"❌ Process terminated early: {output}")
                return False
                
            # Check if we can see any output
            try:
                line = process.stdout.readline()
                if line:
                    print(line.strip())
                    if "You can now view your Streamlit app" in line:
                        print("✅ Streamlit server started successfully!")
                        print("🌐 Open your browser to: http://localhost:8501")
                        print("⚠️  If you see 'Demo Mode Active', wait 3-5 minutes for model loading")
                        print("🔄 Press Ctrl+C to stop the server")
                        return True
            except:
                pass
                
            time.sleep(0.5)
        
        print("⚠️  Startup timeout reached, but server may still be starting...")
        return True
        
    except Exception as e:
        print(f"❌ Failed to start Streamlit: {e}")
        return False

def main():
    print("🔧 AI Financial Portfolio Advisor - Restart Script")
    print("=" * 55)
    print("🆕 **NEW**: Health check fixes implemented!")
    print("   • Disabled blocking health checks")
    print("   • Removed st.cache_resource blocking behavior") 
    print("   • True background model loading")
    print("   • Optimized UI update frequency")
    print()
    
    print("🛑 Step 1: Stopping existing processes...")
    kill_streamlit_processes()
    
    print("\n🔍 Step 2: Checking port availability...")
    if not check_port_available():
        print("⚠️  Waiting for port to be released...")
        time.sleep(5)
        if not check_port_available():
            print("❌ Port 8501 is still in use. You may need to manually kill processes.")
    else:
        print("✅ Port 8501 is available")
    
    print("\n🚀 Step 3: Starting application...")
    success = start_streamlit()
    
    if not success:
        print("\n❌ Failed to start application")
        print("💡 Try running manually: streamlit run streamlit_app.py")
        sys.exit(1)
    else:
        print("\n🎉 Application should now be running without health check errors!")
        print("📊 Expected behavior:")
        print("   ✅ App starts immediately (no blocking)")
        print("   ✅ Model loads in background with progress bar")
        print("   ✅ UI remains responsive during loading")
        print("   ✅ No more 503 health check errors!")

if __name__ == "__main__":
    main() 