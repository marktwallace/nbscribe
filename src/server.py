#!/usr/bin/env python3
"""
nbscribe FastAPI Server
Lightweight server for AI-powered Jupyter notebook assistance
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
from pathlib import Path
import os
from datetime import datetime
import json

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

    # Session management utilities
    def generate_session_id() -> str:
        """Generate a unique session ID with timestamp and milliseconds"""
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        milliseconds = now.microsecond // 1000  # Convert microseconds to milliseconds
        return f"{timestamp}_{milliseconds:03d}"

    def get_latest_session() -> str:
        """Get the most recent session ID, or create new if none exist"""
        from src.conversation_logger import ConversationLogger
        logger = ConversationLogger()
        
        log_files = logger.list_conversation_logs()
        if log_files:
            # Extract session ID from most recent file
            latest_file = log_files[0]  # Already sorted by mtime (most recent first)
            
            # Session ID is now the filename without .html extension
            session_id = latest_file.stem
            return session_id
        
        # No existing sessions, create new
        return generate_session_id()

    def load_session_data(session_id: str) -> dict:
        """Load session data from HTML file or create new session"""
        from src.conversation_logger import ConversationLogger
        logger = ConversationLogger()
        
        # Construct filename directly from session ID
        session_file = logger.logs_dir / f"{session_id}.html"
        
        if session_file.exists():
            # Load existing session
            conversation_data = logger.load_conversation_log(session_file)
            if conversation_data:
                return {
                    'session_id': session_id,
                    'created_at': conversation_data['created_at'],
                    'messages': conversation_data['messages'],
                    'log_file': session_file,
                    'is_new': False
                }
        
        # Create new session
        log_file = logger.create_conversation_log(session_id)
        
        # Add initial greeting message for new sessions
        initial_message = {
            'role': 'assistant',
            'content': "Hello! I'm your AI Jupyter assistant. Type a message below to get started.",
            'timestamp': datetime.now().isoformat()
        }
        
        initial_messages = [initial_message]
        
        # Save initial message to file immediately
        logger.save_conversation_state(log_file, initial_messages)
        
        return {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'messages': initial_messages,
            'log_file': log_file,
            'is_new': True
        }

    # Data models for API
    class ChatMessage(BaseModel):
        message: str
        session_id: str = None
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

    # Session-based chat endpoint
    @app.post("/api/chat", response_model=ChatResponse)
    async def chat_endpoint(message: ChatMessage):
        """
        Main chat endpoint for receiving user messages and generating AI responses
        """
        try:
            # Import here to avoid circular imports and lazy loading
            from src.llm_interface import generate_response
            from src.conversation_logger import ConversationLogger
            
            # Get session ID (fallback to latest if not provided)
            session_id = message.session_id or get_latest_session()
            
            # Load existing conversation context
            session_data = load_session_data(session_id)
            
            # Format conversation context for LLM
            conversation_context = ""
            if session_data['messages']:
                context_lines = []
                for msg in session_data['messages']:
                    role = msg['role'].title()
                    content = msg['content']
                    context_lines.append(f"{role}: {content}")
                conversation_context = "\n\n".join(context_lines)
            
            # Generate response using LLM with conversation context
            response_text = generate_response(message.message, conversation_context)
            
            # Add new messages to conversation
            timestamp = datetime.now().isoformat()
            session_data['messages'].extend([
                {
                    'role': 'user',
                    'content': message.message,
                    'timestamp': timestamp
                },
                {
                    'role': 'assistant', 
                    'content': response_text,
                    'timestamp': timestamp
                }
            ])
            
            # Save updated conversation to HTML file
            logger = ConversationLogger()
            logger.save_conversation_state(session_data['log_file'], session_data['messages'])
            
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

    # Streaming chat endpoint for real-time responses
    @app.post("/api/chat/stream")
    async def chat_stream_endpoint(message: ChatMessage):
        """
        Streaming chat endpoint using Server-Sent Events (SSE)
        """
        try:
            from src.llm_interface import get_llm_interface
            from src.conversation_logger import ConversationLogger
            
            # Get session ID (fallback to latest if not provided)
            session_id = message.session_id or get_latest_session()
            
            # Load existing conversation context
            session_data = load_session_data(session_id)
            
            # Format conversation context for LLM
            conversation_context = ""
            if session_data['messages']:
                context_lines = []
                for msg in session_data['messages']:
                    role = msg['role'].title()
                    content = msg['content']
                    context_lines.append(f"{role}: {content}")
                conversation_context = "\n\n".join(context_lines)
            
            def stream_response():
                """Generator function for SSE streaming"""
                try:
                    llm = get_llm_interface()
                    full_response = ""
                    
                    # Stream the response
                    for chunk in llm.generate_response_stream(message.message, conversation_context):
                        full_response += chunk
                        # Send chunk as SSE
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                    
                    # Response complete - save to file
                    timestamp = datetime.now().isoformat()
                    session_data['messages'].extend([
                        {
                            'role': 'user',
                            'content': message.message,
                            'timestamp': timestamp
                        },
                        {
                            'role': 'assistant', 
                            'content': full_response,
                            'timestamp': timestamp
                        }
                    ])
                    
                    # Save updated conversation to HTML file
                    logger = ConversationLogger()
                    logger.save_conversation_state(session_data['log_file'], session_data['messages'])
                    
                    # Send completion signal
                    yield f"data: {json.dumps({'type': 'complete', 'content': full_response})}\n\n"
                    
                except Exception as e:
                    import traceback
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Streaming error: {e}")
                    logger.error(traceback.format_exc())
                    
                    # Send error as SSE
                    yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            
            return StreamingResponse(
                stream_response(),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"  # Disable nginx buffering
                }
            )
            
        except Exception as e:
            # Log the full error with stack trace for debugging
            import traceback
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Streaming endpoint error: {e}")
            logger.error(traceback.format_exc())
            
            raise HTTPException(status_code=500, detail=str(e))

    # Root route - redirect to latest session or create new
    @app.get("/")
    async def root_redirect():
        """Redirect to latest session or create new session"""
        session_id = get_latest_session()
        return RedirectResponse(url=f"/session/{session_id}", status_code=302)

    # Create new session route
    @app.get("/new")
    async def new_session():
        """Force create a new session"""
        session_id = generate_session_id()
        return RedirectResponse(url=f"/session/{session_id}", status_code=302)

    # Session-specific chat interface
    @app.get("/session/{session_id}", response_class=HTMLResponse)
    async def serve_session_interface(request: Request, session_id: str):
        """Serve chat interface for specific session"""
        # Load session data (creates new if doesn't exist)
        session_data = load_session_data(session_id)
        
        return templates.TemplateResponse("chat.html", {
            "request": request,
            "title": f"nbscribe - Session {session_id}",
            "service_name": "nbscribe",
            "version": "0.1.0",
            "conversation_id": session_id,
            "created_at": session_data['created_at'],
            "messages": session_data['messages']  # Pass existing messages to template
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
                "chat_stream": "/api/chat/stream (SSE)",
                "info": "/api/info",
                "sessions": "GET / (latest), GET /session/{id}, GET /new"
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