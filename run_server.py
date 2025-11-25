#!/usr/bin/env python3
"""
Server entry point for LLM Neuron Attribution Tool.
"""

import os
import uvicorn
from fastapi.staticfiles import StaticFiles
from LKN.server import app, STATIC_DIR

# Create static directory if it doesn't exist
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static files (only if not already mounted)
if os.path.exists(STATIC_DIR):
    try:
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    except ValueError:
        # Already mounted, ignore
        pass

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 LLM Neuron Attribution Tool Server")
    print("=" * 60)
    print("\n서버가 시작되었습니다!")
    print("브라우저에서 http://localhost:8000 을 열어주세요.\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

