#!/usr/bin/env python3
"""
nbscribe FastAPI Server
Lightweight server for AI-powered Jupyter notebook assistance
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
from pathlib import Path
import os

def create_app():
    """Factory function to create FastAPI app"""
    app = FastAPI(
        title="nbscribe",
        description="AI-powered Jupyter Notebook assistant",
        version="0.1.0"
    )

    # Create static directory if it doesn't exist (relative to project root)
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)

    # Mount static files
    app.mount("/static", StaticFiles(directory="static"), name="static")

    # Set up Jinja2 templates
    templates = Jinja2Templates(directory="templates")

    # Data models for API
    class ChatMessage(BaseModel):
        message: str
        notebook_path: str = None

    class ChatResponse(BaseModel):
        response: str
        success: bool = True

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint to verify server is running"""
        return JSONResponse({
            "status": "healthy",
            "service": "nbscribe",
            "version": "0.1.0"
        })

    # Basic chat endpoint
    @app.post("/api/chat", response_model=ChatResponse)
    async def chat_endpoint(message: ChatMessage):
        """
        Main chat endpoint for receiving user messages and generating AI responses
        """
        try:
            # Import here to avoid circular imports and lazy loading
            from src.llm_interface import generate_response
            
            # Generate response using LLM (no conversation context for now)
            response_text = generate_response(message.message)
            
            return ChatResponse(
                response=response_text,
                success=True
            )
        except Exception as e:
            # Log the full error with stack trace for debugging
            import traceback
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Chat endpoint error: {e}")
            logger.error(traceback.format_exc())
            
            raise HTTPException(status_code=500, detail=str(e))

    # Serve the main chat interface
    @app.get("/", response_class=HTMLResponse)
    async def serve_chat_interface(request: Request):
        """Serve the main chat interface using Jinja2 template"""
        from datetime import datetime
        
        return templates.TemplateResponse("chat.html", {
            "request": request,
            "title": "nbscribe - Conversation Log",
            "service_name": "nbscribe",
            "version": "0.1.0",
            "conversation_id": "session_001",  # TODO: Generate unique session ID
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    # API info endpoint
    @app.get("/api/info")
    async def api_info():
        """Get API information"""
        return JSONResponse({
            "service": "nbscribe",
            "version": "0.1.0",
            "endpoints": {
                "health": "/health",
                "chat": "/api/chat",
                "info": "/api/info"
            }
        })

    return app

def run_server(host="0.0.0.0", port=5317, reload=True):
    """Run the FastAPI server"""
    print(f"🚀 Starting nbscribe server on port {port}")
    print(f"📖 Open http://localhost:{port} in your browser")
    
    uvicorn.run(
        "src.server:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    # Default port from the README
    port = int(os.getenv("PORT", 5317))
    run_server(port=port) 