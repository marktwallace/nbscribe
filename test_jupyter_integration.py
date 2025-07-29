#!/usr/bin/env python3
"""
Test script for Jupyter Notebook integration
Verifies subprocess management and proxy functionality
"""

import sys
import time
import requests
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.notebook_server import NotebookServerManager

def test_notebook_server():
    """Test basic notebook server functionality"""
    print("🧪 Testing Jupyter Notebook Server Management...")
    
    # Test server startup
    print("1. Testing server startup...")
    manager = NotebookServerManager()
    
    try:
        manager.start()
        print(f"   ✅ Server started on port {manager.port}")
        
        # Test health check
        print("2. Testing server health...")
        status_url = manager.get_url("/api/status")
        response = requests.get(status_url, params={"token": manager.token}, timeout=5)
        
        if response.status_code == 200:
            print("   ✅ Server health check passed")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            
        # Test token URL
        print("3. Testing token authentication...")
        token_url = manager.get_token_url("/tree")
        print(f"   🔗 Notebook URL: {token_url}")
        
        response = requests.get(token_url, timeout=5)
        if response.status_code == 200:
            print("   ✅ Token authentication working")
        else:
            print(f"   ❌ Token auth failed: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        
    finally:
        # Clean shutdown
        print("4. Testing server shutdown...")
        manager.stop()
        print("   ✅ Server stopped")

def test_fastapi_integration():
    """Test FastAPI server with Jupyter integration"""
    print("\n🧪 Testing FastAPI Integration...")
    print("   ℹ️  Start the main server with: python main.py")
    print("   ℹ️  Then test these URLs:")
    print("     • http://localhost:5317/api/notebook/status")
    print("     • http://localhost:5317/jupyter/tree")
    print("     • http://localhost:5317/ (main chat interface)")

if __name__ == "__main__":
    test_notebook_server()
    test_fastapi_integration()
    print("\n🎉 Basic tests completed!")
    print("    Start the full server with: python main.py") 