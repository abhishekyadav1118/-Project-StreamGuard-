"""
Project StreamGuard: Standalone Platform Launcher.
Runs the Flask application server on http://127.0.0.1:5000
"""

import os
import sys

# Ensure local imports work reliably
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 65)
    print(" STREAMGUARD: OTT CHURN & REVENUE INTELLIGENCE PLATFORM")
    print("=" * 65)
    print(f" Web Server live at: http://127.0.0.1:{port}")
    print(" REST API Endpoints active for Real-Time ML Inference.")
    print("=" * 65)
    app.run(host="0.0.0.0", port=port, debug=False)
