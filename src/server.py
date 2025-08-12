#!/usr/bin/env python3
"""
nbscribe FastAPI Server (htmx Migration)
Lightweight server for AI-powered Jupyter notebook assistance with htmx interface
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
import uvicorn
from pathlib import Path
import os
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


def create_app():
    """Factory function to create FastAPI app"""
    app = FastAPI(
        title="nbscribe",
        description="AI-powered Jupyter Notebook assistant with htmx interface",
        version="0.2.0"
    )
    
    # Create static directory if it doesn't exist
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)

    # Mount static files
    app.mount("/static", StaticFiles(directory="static"), name="static")

    # Set up Jinja2 templates
    templates = Jinja2Templates(directory="templates")
    app.state.templates = templates

    # Data models for API
    class ChatMessage(BaseModel):
        message: str
        session_id: str = None
        notebook_path: str = None

    class ChatResponse(BaseModel):
        response: str
        success: bool = True
    
    class DirectiveRequest(BaseModel):
        id: str
        tool: str
        pos: Optional[int] = None
        cell_id: Optional[str] = None
        before: Optional[str] = None
        after: Optional[str] = None
        group_id: Optional[str] = None
        seq: Optional[int] = None
        code: str
        language: str = "python"
        session_id: Optional[str] = None
    
    class CreateNotebookRequest(BaseModel):
        name: str

    # Session management utilities
    def generate_session_id() -> str:
        """Generate a unique session ID with timestamp and milliseconds"""
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        milliseconds = now.microsecond // 1000
        return f"{timestamp}_{milliseconds:03d}"

    def extract_notebook_path_from_session(session_id: str) -> Optional[str]:
        """Extract notebook path from session ID if it's a notebook session"""
        if session_id.startswith("notebook_"):
            path_part = session_id[9:]
            
            if path_part.endswith('_ipynb'):
                notebook_path = path_part[:-6] + '.ipynb'
            elif path_part.endswith('_py'):
                notebook_path = path_part[:-3] + '.py'
            else:
                possible_paths = [
                    path_part.replace('_', '.') + '.ipynb',
                    path_part.replace('_', '/') + '.ipynb',
                    path_part + '.ipynb'
                ]
                
                for path in possible_paths:
                    if Path(path).exists():
                        return path
                
                notebook_path = path_part.replace('_', '.') + '.ipynb'
            
            logger.info(f"SESSION->PATH: {session_id} -> {notebook_path}")
            return notebook_path
        
        return None

    def get_latest_session() -> str:
        """Get the most recent session ID, or create new if none exist"""
        from src.conversation_logger import ConversationLogger
        logger_instance = ConversationLogger()
        
        log_files = logger_instance.list_conversation_logs()
        if log_files:
            latest_file = log_files[0]
            session_id = latest_file.stem
            return session_id
        
        return generate_session_id()

    def get_recent_notebooks() -> list:
        """Get list of recent notebook files in current directory"""
        try:
            notebook_files = []
            current_dir = Path(".")
            
            for notebook_path in current_dir.glob("*.ipynb"):
                if notebook_path.is_file():
                    stat = notebook_path.stat()
                    notebook_files.append({
                        "name": notebook_path.name,
                        "path": str(notebook_path),
                        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                    })
            
            notebook_files.sort(key=lambda x: Path(x["path"]).stat().st_mtime, reverse=True)
            return notebook_files[:5]
            
        except Exception as e:
            logger.error(f"Error getting recent notebooks: {e}")
            return []

    def load_session_data(session_id: str) -> dict:
        """Load session data from HTML file or create new session"""
        from src.conversation_logger import ConversationLogger
        logger_instance = ConversationLogger()
        
        session_file = logger_instance.logs_dir / f"{session_id}.html"
        
        if session_file.exists():
            conversation_data = logger_instance.load_conversation_log(session_file)
            if conversation_data:
                return {
                    'session_id': session_id,
                    'created_at': conversation_data['created_at'],
                    'messages': conversation_data['messages'],
                    'log_file': session_file,
                    'is_new': False
                }
        
        log_file = logger_instance.create_conversation_log(session_id)
        
        initial_message = {
            'role': 'assistant',
            'content': "Hello! I'm your AI Jupyter assistant. Type a message below to get started.",
            'timestamp': datetime.now().isoformat()
        }
        
        initial_messages = [initial_message]
        logger_instance.save_conversation_state(log_file, initial_messages)
        
        return {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'messages': initial_messages,
            'log_file': log_file,
            'is_new': True
        }

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint to verify server is running"""
        return JSONResponse({
            "status": "healthy",
            "service": "nbscribe",
            "version": "0.2.0"
        })

    # Include notebook routes
    from src.notebook_routes import create_notebook_router
    notebook_router = create_notebook_router(templates)
    app.include_router(notebook_router)

    # Root route - show notebook picker
    @app.get("/")
    async def notebook_picker(request: Request):
        """Show notebook file picker landing page"""
        recent_notebooks = get_recent_notebooks()
        return templates.TemplateResponse("notebook_picker.html", {
            "request": request,
            "recent_notebooks": recent_notebooks
        })
    
    # Legacy chat-only route
    @app.get("/chat")
    async def chat_only():
        """Redirect to latest session for chat-only mode"""
        session_id = get_latest_session()
        return RedirectResponse(url=f"/session/{session_id}", status_code=302)

    # Create new session route
    @app.get("/new")
    async def new_session():
        """Force create a new session"""
        session_id = generate_session_id()
        return RedirectResponse(url=f"/session/{session_id}", status_code=302)

    # Session-based chat endpoint (PRESERVED from original)
    @app.post("/api/chat", response_model=ChatResponse)
    async def chat_endpoint(message: ChatMessage):
        """Main chat endpoint for receiving user messages and generating AI responses"""
        try:
            from src.llm_interface import get_llm_interface
            from src.conversation_logger import ConversationLogger
            from src.conversation_manager import get_conversation_manager
            
            session_id = message.session_id or get_latest_session()
            logger.info(f"CHAT LINEAR DEBUG - Session ID: {session_id}")
            
            session_data = load_session_data(session_id)
            
            conv_manager = get_conversation_manager()
            llm = get_llm_interface()
            
            linear_messages = conv_manager.build_linear_conversation(
                session_data['messages'], 
                llm.system_prompt or "You are a helpful AI assistant for Jupyter notebooks."
            )
            
            logger.info(f"CHAT LINEAR DEBUG - Built linear conversation with {len(linear_messages)} messages")
            
            response_text = llm.generate_response(linear_messages, message.message)
            
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
            
            logger_instance = ConversationLogger()
            logger_instance.save_conversation_state(session_data['log_file'], session_data['messages'])
            
            return ChatResponse(
                response=response_text,
                success=True
            )
        except Exception as e:
            import traceback
            logger.error(f"Chat endpoint error: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    # Streaming chat endpoint (PRESERVED from original)
    @app.post("/api/chat/stream")
    async def chat_stream_endpoint(message: ChatMessage):
        """Streaming chat endpoint using Server-Sent Events (SSE)"""
        try:
            from fastapi.responses import StreamingResponse
            from src.llm_interface import get_llm_interface
            from src.conversation_logger import ConversationLogger
            from src.conversation_manager import get_conversation_manager
            
            session_id = message.session_id or get_latest_session()
            logger.info(f"STREAM LINEAR DEBUG - Session ID: {session_id}")
            
            session_data = load_session_data(session_id)
            
            conv_manager = get_conversation_manager()
            llm = get_llm_interface()
            
            linear_messages = conv_manager.build_linear_conversation(
                session_data['messages'], 
                llm.system_prompt or "You are a helpful AI assistant for Jupyter notebooks."
            )
            
            logger.info(f"STREAM LINEAR DEBUG - Built linear conversation with {len(linear_messages)} messages")
            
            def stream_response():
                """Generator function for SSE streaming"""
                try:
                    llm = get_llm_interface()
                    full_response = ""
                    
                    for chunk in llm.generate_response_stream(linear_messages, message.message):
                        full_response += chunk
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                    
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
                    
                    logger_instance = ConversationLogger()
                    logger_instance.save_conversation_state(session_data['log_file'], session_data['messages'])
                    
                    yield f"data: {json.dumps({'type': 'complete', 'content': full_response})}\n\n"
                    
                except Exception as e:
                    import traceback
                    logger.error(f"Streaming error: {e}")
                    logger.error(traceback.format_exc())
                    yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            
            return StreamingResponse(
                stream_response(),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
            
        except Exception as e:
            import traceback
            logger.error(f"Streaming endpoint error: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    # Session-specific chat interface (PRESERVED from original)
    @app.get("/session/{session_id}", response_class=HTMLResponse)
    async def serve_session_interface(request: Request, session_id: str):
        """Serve chat interface for specific session"""
        session_data = load_session_data(session_id)
        
        return templates.TemplateResponse("chat.html", {
            "request": request,
            "title": f"nbscribe - Session {session_id}",
            "service_name": "nbscribe",
            "version": "0.2.0",
            "conversation_id": session_id,
            "created_at": session_data['created_at'],
            "messages": session_data['messages'],
            "notebook_iframe_url": None,
            "chat_only": True
        })

    # API info endpoint
    @app.get("/api/info")
    async def api_info():
        """Get API information"""
        return JSONResponse({
            "service": "nbscribe",
            "version": "0.2.0",
            "endpoints": {
                "health": "/health",
                "chat": "/api/chat",
                "chat_stream": "/api/chat/stream (SSE)",
                "info": "/api/info",
                "sessions": "GET / (latest), GET /session/{id}, GET /new",
                "notebooks": "GET /nb/{path} (htmx interface)"
            }
        })

    @app.get("/api/system-prompt")
    async def get_system_prompt():
        """Get the current system prompt for transparency"""
        try:
            from src.llm_interface import get_llm_interface
            llm = get_llm_interface()
            return JSONResponse({
                "prompt": llm.system_prompt,
                "source": "prompts/system_prompt.txt"
            })
        except Exception as e:
            logger.error(f"Error getting system prompt: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/notebooks/create")
    async def create_notebook(request: CreateNotebookRequest):
        """Create a new Jupyter notebook"""
        try:
            from src import nbio
            
            # Create empty notebook
            nb = nbio.create_empty_notebook()
            
            # Add a simple welcome cell
            welcome_cell = nbio.create_code_cell(
                "# Welcome to your new notebook!\n# Start coding below..."
            )
            nbio.insert_cell(nb, welcome_cell, 0)
            
            # Save to current directory
            notebook_path = Path(request.name)
            
            # Ensure unique filename
            counter = 1
            original_stem = notebook_path.stem
            while notebook_path.exists():
                notebook_path = Path(f"{original_stem}-{counter}.ipynb")
                counter += 1
            
            # Write notebook file
            nbio.write_nb(str(notebook_path), nb)
            
            return JSONResponse({
                "success": True,
                "path": str(notebook_path),
                "name": notebook_path.name
            })
            
        except Exception as e:
            logger.error(f"Error creating notebook: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/directives/approve")
    async def approve_directive(directive: DirectiveRequest):
        """Apply an approved tool directive to the notebook using new nbio system"""
        try:
            from src import nbio
            from src.conversation_manager import get_conversation_manager
            from src.conversation_logger import ConversationLogger
            
            # Extract notebook path from session context
            session_id = directive.session_id
            if session_id:
                notebook_path = extract_notebook_path_from_session(session_id)
            else:
                notebooks = [f for f in Path(".").glob("*.ipynb")]
                if not notebooks:
                    raise Exception("No notebook found - please open a specific notebook")
                notebook_path = str(notebooks[0])
            
            if not notebook_path:
                raise Exception("Cannot determine notebook path from session context")
            
            logger.info(f"Working with notebook: {notebook_path}")
            
            result_message = ""
            cell_data = None
            operation_type = ""
            
            if directive.tool == "insert_cell":
                # Read notebook
                nb = nbio.read_nb(notebook_path)
                
                # Resolve insertion position
                cells = nb.get("cells", [])
                if directive.before:
                    pos = nbio.find_cell_index(nb, directive.before)
                    if pos is not None:
                        insert_position = pos
                    else:
                        raise Exception(f"Cell ID '{directive.before}' not found for BEFORE positioning")
                elif directive.after:
                    pos = nbio.find_cell_index(nb, directive.after)
                    if pos is not None:
                        insert_position = pos + 1
                    else:
                        raise Exception(f"Cell ID '{directive.after}' not found for AFTER positioning")
                elif directive.pos is not None:
                    if 0 <= directive.pos <= len(cells):
                        insert_position = directive.pos
                    else:
                        raise Exception(f"Position {directive.pos} is out of bounds (0-{len(cells)})")
                else:
                    raise Exception("No position specified for insert_cell (need BEFORE, AFTER, or POS)")
                
                # Create new cell
                if directive.language == "python":
                    new_cell = nbio.create_code_cell(directive.code)
                else:
                    new_cell = nbio.create_markdown_cell(directive.code)
                
                # Insert cell
                nbio.insert_cell(nb, new_cell, insert_position)
                
                # Write notebook
                nbio.write_nb(notebook_path, nb)
                
                result_message = f"Cell inserted at position {insert_position} in {notebook_path}"
                operation_type = "inserted"
                cell_data = new_cell
                
            elif directive.tool == "edit_cell":
                # Read notebook
                nb = nbio.read_nb(notebook_path)
                
                # Update cell
                success = nbio.update_cell_source(nb, directive.cell_id, directive.code)
                if not success:
                    raise Exception(f"Cell ID '{directive.cell_id}' not found")
                
                # Write notebook
                nbio.write_nb(notebook_path, nb)
                
                result_message = f"Cell {directive.cell_id} edited in {notebook_path}"
                operation_type = "updated"
                cell_data = nbio.get_cell_by_id(nb, directive.cell_id)
                
            elif directive.tool == "delete_cell":
                # Read notebook
                nb = nbio.read_nb(notebook_path)
                
                # Delete cell
                success = nbio.delete_cell(nb, directive.cell_id)
                if not success:
                    raise Exception(f"Cell ID '{directive.cell_id}' not found")
                
                # Write notebook
                nbio.write_nb(notebook_path, nb)
                
                result_message = f"Cell {directive.cell_id} deleted from {notebook_path}"
                operation_type = "deleted"
                
            else:
                raise HTTPException(status_code=400, detail=f"Unknown tool: {directive.tool}")
            
            # Add post-modification message to conversation history
            try:
                conv_manager = get_conversation_manager()
                
                if operation_type == "deleted":
                    update_message_content = conv_manager.format_cell_deletion(directive.cell_id)
                elif cell_data:
                    update_message_content = conv_manager.format_cell_update(cell_data, operation_type)
                else:
                    update_message_content = conv_manager.format_operation_failure(
                        directive.tool, 
                        "Success but cell data unavailable for conversation", 
                        directive.cell_id
                    )
                
                if directive.session_id:
                    session_data = load_session_data(directive.session_id)
                    update_message = {
                        'role': 'user',
                        'content': update_message_content,
                        'timestamp': datetime.now().isoformat()
                    }
                    session_data['messages'].append(update_message)
                    
                    logger_instance = ConversationLogger()
                    logger_instance.save_conversation_state(session_data['log_file'], session_data['messages'])
                    
                    logger.info(f"Added {operation_type} cell message to conversation: {directive.session_id}")
                
            except Exception as e:
                logger.error(f"Error adding post-modification message to conversation: {e}")
            
            return JSONResponse({
                "success": True,
                "message": result_message,
                "directive_id": directive.id
            })
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error applying directive: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # App lifecycle events
    @app.on_event("startup")
    async def startup_event():
        """Initialize kernel pool on startup"""
        logger.info("Starting nbscribe with htmx interface...")
        # Note: Kernel pool is initialized lazily when first needed

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown"""
        try:
            logger.info("Shutting down nbscribe...")
            from src.kernel import get_kernel_manager
            kernel_manager = get_kernel_manager()
            kernel_manager.shutdown_all_kernels()
            logger.info("Kernel shut down")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    return app


def run_server(host="0.0.0.0", port=5317, reload=True):
    """Run the FastAPI server"""
    print(f"🚀 Starting nbscribe server (htmx) on port {port}")
    print(f"📖 Open http://localhost:{port} in your browser")
    
    uvicorn.run(
        "src.server:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5317))
    run_server(port=port)
