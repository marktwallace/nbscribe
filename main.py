#!/usr/bin/env python3
"""
nbscribe - AI-powered Jupyter Notebook Assistant
Main entry point for the application
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from src.server import run_server

def main():
    """Main entry point"""
    # Default port from the README
    port = int(os.getenv("PORT", 5317))
    
    # Run the server
    run_server(port=port, reload=True)

if __name__ == "__main__":
    main() 