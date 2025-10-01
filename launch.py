#!/usr/bin/env python3
"""
Simple launcher for the Menu Taste Guide interactive interface.
"""

import os
import webbrowser
import time
import threading
from start_api import app
import uvicorn

def open_browser():
    """Open browser after a short delay."""
    time.sleep(2)
    webbrowser.open('http://localhost:8000/app')

def main():
    print("🍜 Menu Taste Guide - Interactive Demo Launcher")
    print("=" * 60)
    print("🚀 Starting server...")
    print("📱 Interactive App: http://localhost:8000/app")
    print("📊 API Overview: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("=" * 60)
    print("Press Ctrl+C to stop the server")
    print()

    # Start browser in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

    # Start the server
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Thanks for trying Menu Taste Guide!")

if __name__ == "__main__":
    main()