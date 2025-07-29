#!/usr/bin/env python3
"""
nbscribe FastAPI Server
Lightweight server for AI-powered Jupyter notebook assistance
"""

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
import uvicorn
import httpx
from pathlib import Path
import os
from datetime import datetime
import json
import logging

# Surgical log filter - suppress only the identified repetitive polling patterns
class JupyterRepetitiveFilter(logging.Filter):
    """Suppress only the specific repetitive Jupyter polling we've identified"""
    def filter(self, record):
        if hasattr(record, 'getMessage'):
            message = record.getMessage()
            
            # Only suppress these specific repetitive patterns
            repetitive_patterns = [
                '/checkpoints?',                    # Checkpoint polling every 2s
                '/contents?content=1&hash=0&',      # File system monitoring  
                '/kernels?',                        # Kernel status polling
                '/sessions?',                       # Session status polling
                '/kernelspecs?',                    # Kernel specs polling
                '/me?',                             # User info polling
            ]
            
            # Suppress if it's a GET request matching these patterns
            if 'GET /jupyter/api/' in message:
                if any(pattern in message for pattern in repetitive_patterns):
                    return False
                    
        return True

# Apply surgical filter to suppress only identified noise
logging.getLogger("uvicorn.access").addFilter(JupyterRepetitiveFilter())

def create_app():
    """Factory function to create FastAPI app"""
    app = FastAPI(
        title="nbscribe",
        description="AI-powered Jupyter Notebook assistant",
        version="0.1.0"
    )
    
    # Initialize notebook server manager
    from src.notebook_server import NotebookServerManager
    notebook_manager = NotebookServerManager()
    
    # Store reference for access in routes
    app.state.notebook_manager = notebook_manager

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

    # Notebook modification helper functions
    async def insert_notebook_cell(position: int, code: str, language: str, notebook_manager) -> str:
        """Insert a new cell into the notebook at the specified position"""
        try:
            # For now, we'll implement a simple approach by reading the notebook,
            # modifying it, and writing it back
            # TODO: In future, we could use Jupyter's live kernel API for real-time updates
            
            # This is a placeholder implementation
            # In a real implementation, we'd need to:
            # 1. Get the current notebook path from the session
            # 2. Read the notebook file via Jupyter API
            # 3. Insert the new cell at the specified position
            # 4. Write the notebook back
            
            logging.info(f"Insert cell requested: position={position}, language={language}")
            logging.info(f"Code: {code[:100]}...")  # Log first 100 chars
            
            return f"Cell inserted at position {position} (placeholder implementation)"
            
        except Exception as e:
            logging.error(f"Error inserting cell: {e}")
            raise Exception(f"Failed to insert cell: {e}")
    
    async def edit_notebook_cell(cell_id: str, code: str, notebook_manager) -> str:
        """Edit an existing cell in the notebook"""
        try:
            logging.info(f"Edit cell requested: cell_id={cell_id}")
            logging.info(f"New code: {code[:100]}...")  # Log first 100 chars
            
            return f"Cell {cell_id} edited (placeholder implementation)"
            
        except Exception as e:
            logging.error(f"Error editing cell: {e}")
            raise Exception(f"Failed to edit cell: {e}")
    
    async def delete_notebook_cell(cell_id: str, notebook_manager) -> str:
        """Delete a cell from the notebook"""
        try:
            logging.info(f"Delete cell requested: cell_id={cell_id}")
            
            return f"Cell {cell_id} deleted (placeholder implementation)"
            
        except Exception as e:
            logging.error(f"Error deleting cell: {e}")
            raise Exception(f"Failed to delete cell: {e}")

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

    def get_recent_notebooks() -> list:
        """Get list of recent notebook files in current directory"""
        try:
            notebook_files = []
            current_dir = Path(".")
            
            # Find all .ipynb files
            for notebook_path in current_dir.glob("*.ipynb"):
                if notebook_path.is_file():
                    stat = notebook_path.stat()
                    notebook_files.append({
                        "name": notebook_path.name,
                        "path": str(notebook_path),
                        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                    })
            
            # Sort by modification time (newest first)
            notebook_files.sort(key=lambda x: Path(x["path"]).stat().st_mtime, reverse=True)
            
            # Return top 5
            return notebook_files[:5]
            
        except Exception as e:
            logging.error(f"Error getting recent notebooks: {e}")
            return []

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
    
    class DirectiveRequest(BaseModel):
        id: str
        tool: str
        pos: Optional[int] = None
        cell_id: Optional[str] = None
        code: str
        language: str = "python"
    
    class CreateNotebookRequest(BaseModel):
        name: str

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

    # Notebook-specific interface
    @app.get("/notebook/{notebook_path:path}", response_class=HTMLResponse)
    async def serve_notebook_interface(request: Request, notebook_path: str):
        """Serve split-pane interface for specific notebook"""
        if not notebook_manager.is_running():
            raise HTTPException(status_code=503, detail="Jupyter server not running")
        
        # Ensure notebook exists
        notebook_file = Path(notebook_path)
        if not notebook_file.exists():
            raise HTTPException(status_code=404, detail=f"Notebook not found: {notebook_path}")
        
        # Create or get session for this notebook
        session_id = f"notebook_{notebook_path.replace('/', '_').replace('.', '_')}"
        session_data = load_session_data(session_id)
        
        # Update greeting for notebook context
        if session_data['is_new']:
            initial_message = {
                'role': 'assistant',
                'content': f"Hello! I'm ready to help you with `{notebook_file.name}`. I can analyze your notebook and suggest code edits. What would you like to work on?",
                'timestamp': datetime.now().isoformat()
            }
            session_data['messages'] = [initial_message]
            
            # Save updated greeting
            from src.conversation_logger import ConversationLogger
            logger = ConversationLogger()
            logger.save_conversation_state(session_data['log_file'], session_data['messages'])
        
        return templates.TemplateResponse("chat.html", {
            "request": request,
            "title": f"nbscribe - {notebook_file.name}",
            "service_name": "nbscribe",
            "version": "0.1.0",
            "conversation_id": session_id,
            "created_at": session_data['created_at'],
            "messages": session_data['messages'],
            "notebook_path": notebook_path,
            "notebook_iframe_url": f"/jupyter/notebooks/{notebook_path}"
        })
    
    # Session-specific chat interface (legacy)
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
            "messages": session_data['messages'],  # Pass existing messages to template
            "notebook_iframe_url": None,  # No notebook pane for chat-only sessions
            "chat_only": True  # Flag to hide notebook pane
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
            logging.error(f"Error getting system prompt: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Jupyter proxy routes - must be added before other routes to avoid conflicts
    @app.api_route("/jupyter/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def jupyter_proxy(request: Request, path: str):
        """Proxy all /jupyter/* requests to the Jupyter Notebook server"""
        if not notebook_manager.is_running():
            raise HTTPException(status_code=503, detail="Jupyter server not running")
        
        # Build target URL
        target_url = f"http://localhost:{notebook_manager.port}/jupyter/{path}"
        
        # Handle query parameters with auto-token injection for GET requests
        query_params = dict(request.query_params)
        
        # Auto-inject token for GET requests if not already present
        if request.method == "GET" and "token" not in query_params:
            query_params["token"] = notebook_manager.token
        
        # Add query parameters to URL
        if query_params:
            query_string = "&".join([f"{k}={v}" for k, v in query_params.items()])
            target_url += f"?{query_string}"
        
        # Get request body if present
        body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            body = await request.body()
        
        # Forward headers (exclude host-related headers)
        headers = {
            key: value for key, value in request.headers.items()
            if key.lower() not in ["host", "content-length"]
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    content=body,
                    follow_redirects=False
                )
                
                # Forward response headers (exclude problematic ones)
                response_headers = {
                    key: value for key, value in response.headers.items()
                    if key.lower() not in ["content-encoding", "content-length", "transfer-encoding"]
                }
                
                return StreamingResponse(
                    content=response.aiter_bytes(),
                    status_code=response.status_code,
                    headers=response_headers,
                    media_type=response.headers.get("content-type")
                )
                
        except httpx.RequestError as e:
            logging.error(f"Jupyter proxy error: {e}")
            raise HTTPException(status_code=503, detail=f"Failed to connect to Jupyter server: {e}")
    
    # Notebook server management endpoints
    @app.post("/api/notebook/start")
    async def start_notebook_server():
        """Start the Jupyter Notebook server"""
        try:
            if notebook_manager.is_running():
                return JSONResponse({"status": "already_running", "port": notebook_manager.port})
            
            notebook_manager.start()
            return JSONResponse({
                "status": "started",
                "port": notebook_manager.port,
                "url": notebook_manager.get_token_url()
            })
        except Exception as e:
            logging.error(f"Failed to start notebook server: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/notebook/stop")
    async def stop_notebook_server():
        """Stop the Jupyter Notebook server"""
        try:
            notebook_manager.stop()
            return JSONResponse({"status": "stopped"})
        except Exception as e:
            logging.error(f"Failed to stop notebook server: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/notebook/status")
    async def notebook_server_status():
        """Get Jupyter Notebook server status"""
        return JSONResponse({
            "running": notebook_manager.is_running(),
            "port": notebook_manager.port,
            "url": notebook_manager.get_token_url() if notebook_manager.is_running() else None
        })
    
    @app.get("/api/notebook/logs")
    async def notebook_server_logs(lines: int = 50):
        """Get recent Jupyter server logs"""
        if not notebook_manager.is_running():
            raise HTTPException(status_code=503, detail="Jupyter server not running")
        
        logs = notebook_manager.get_recent_logs(lines)
        return JSONResponse({
            "stdout": logs["stdout"],
            "stderr": logs["stderr"],
            "total_lines": len(logs["stdout"]) + len(logs["stderr"])
        })
    
    @app.post("/api/notebooks/create")
    async def create_notebook(request: CreateNotebookRequest):
        """Create a new Jupyter notebook"""
        try:
            import nbformat
            
            # Create empty notebook
            nb = nbformat.v4.new_notebook()
            
            # Add a simple welcome cell
            welcome_cell = nbformat.v4.new_code_cell(
                "# Welcome to your new notebook!\n# Start coding below..."
            )
            nb.cells.append(welcome_cell)
            
            # Save to current directory
            notebook_path = Path(request.name)
            
            # Ensure unique filename
            counter = 1
            original_stem = notebook_path.stem
            while notebook_path.exists():
                notebook_path = Path(f"{original_stem}-{counter}.ipynb")
                counter += 1
            
            # Write notebook file
            with open(notebook_path, 'w') as f:
                nbformat.write(nb, f)
            
            return JSONResponse({
                "success": True,
                "path": str(notebook_path),
                "name": notebook_path.name
            })
            
        except Exception as e:
            logging.error(f"Error creating notebook: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/directives/approve")
    async def approve_directive(directive: DirectiveRequest):
        """Apply an approved tool directive to the notebook"""
        try:
            if not notebook_manager.is_running():
                raise HTTPException(status_code=503, detail="Jupyter server not running")
            
            # Parse the tool directive and execute the corresponding notebook operation
            result_message = ""
            
            if directive.tool == "insert_cell":
                # Insert a new cell at the specified position
                result_message = await insert_notebook_cell(
                    directive.pos, 
                    directive.code, 
                    directive.language,
                    notebook_manager
                )
                
            elif directive.tool == "edit_cell":
                # Edit an existing cell
                result_message = await edit_notebook_cell(
                    directive.cell_id,
                    directive.code,
                    notebook_manager
                )
                
            elif directive.tool == "delete_cell":
                # Delete a cell
                result_message = await delete_notebook_cell(
                    directive.cell_id,
                    notebook_manager
                )
                
            else:
                raise HTTPException(status_code=400, detail=f"Unknown tool: {directive.tool}")
            
            return JSONResponse({
                "success": True,
                "message": result_message,
                "directive_id": directive.id
            })
            
        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"Error applying directive: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # WebSocket proxy endpoints - forward to actual Jupyter server
    @app.websocket("/jupyter/api/events/subscribe")
    async def jupyter_events_websocket(websocket: WebSocket):
        """Proxy WebSocket for Jupyter event subscriptions"""
        if not notebook_manager.is_running():
            await websocket.close(code=1011, reason="Jupyter server not running")
            return
            
        target_url = f"ws://localhost:{notebook_manager.port}/jupyter/api/events/subscribe"
        
        upstream = None
        try:
            import websockets
            import asyncio
            
            await websocket.accept()
            
            # Connect to upstream Jupyter server
            upstream = await websockets.connect(f"{target_url}?token={notebook_manager.token}")
            
            # Track connection state
            client_closed = asyncio.Event()
            upstream_closed = asyncio.Event()
            
            async def forward_to_upstream():
                try:
                    while not client_closed.is_set() and not upstream_closed.is_set():
                        message = await websocket.receive_text()
                        if not upstream_closed.is_set():
                            await upstream.send(message)
                except WebSocketDisconnect:
                    client_closed.set()
                except websockets.exceptions.ConnectionClosed:
                    upstream_closed.set()
                except Exception as e:
                    logging.error(f"Events forward to upstream error: {e}")
                    client_closed.set()
                    upstream_closed.set()
            
            async def forward_to_client():
                try:
                    async for message in upstream:
                        if not client_closed.is_set():
                            try:
                                await websocket.send_text(message)
                            except Exception:
                                # Client disconnected while sending
                                client_closed.set()
                                break
                except websockets.exceptions.ConnectionClosed:
                    upstream_closed.set()
                except Exception as e:
                    logging.error(f"Events forward to client error: {e}")
                    upstream_closed.set()
            
            # Run both forwarding tasks with proper cleanup
            await asyncio.gather(
                forward_to_upstream(), 
                forward_to_client(),
                return_exceptions=True
            )
                
        except Exception as e:
            logging.error(f"Events WebSocket proxy error: {e}")
        finally:
            # Clean up connections
            if upstream:
                try:
                    await upstream.close()
                except Exception:
                    pass
            
            # Only close websocket if not already closed
            try:
                if websocket.client_state.name != "DISCONNECTED":
                    await websocket.close()
            except Exception:
                pass
    
    @app.websocket("/jupyter/api/kernels/{kernel_id}/channels")
    async def jupyter_kernel_websocket(websocket: WebSocket, kernel_id: str):
        """Proxy WebSocket for Jupyter kernel channels - this is crucial for cell execution"""
        if not notebook_manager.is_running():
            await websocket.close(code=1011, reason="Jupyter server not running")
            return
            
        # Get query parameters (especially session_id)
        query_params = dict(websocket.query_params)
        query_string = "&".join([f"{k}={v}" for k, v in query_params.items()])
        if "token" not in query_params:
            query_string += f"&token={notebook_manager.token}"
        
        target_url = f"ws://localhost:{notebook_manager.port}/jupyter/api/kernels/{kernel_id}/channels"
        if query_string:
            target_url += f"?{query_string}"
        
        upstream = None
        try:
            import websockets
            import asyncio
            
            await websocket.accept()
            
            # Connect to upstream Jupyter server
            upstream = await websockets.connect(target_url)
            
            # Track connection state
            client_closed = asyncio.Event()
            upstream_closed = asyncio.Event()
            
            async def forward_to_upstream():
                try:
                    while not client_closed.is_set() and not upstream_closed.is_set():
                        message = await websocket.receive_text()
                        if not upstream_closed.is_set():
                            await upstream.send(message)
                except WebSocketDisconnect:
                    client_closed.set()
                except websockets.exceptions.ConnectionClosed:
                    upstream_closed.set()
                except Exception as e:
                    logging.error(f"Forward to upstream error: {e}")
                    client_closed.set()
                    upstream_closed.set()
            
            async def forward_to_client():
                try:
                    async for message in upstream:
                        if not client_closed.is_set():
                            try:
                                await websocket.send_text(message)
                            except Exception:
                                # Client disconnected while sending
                                client_closed.set()
                                break
                except websockets.exceptions.ConnectionClosed:
                    upstream_closed.set()
                except Exception as e:
                    logging.error(f"Forward to client error: {e}")
                    upstream_closed.set()
            
            # Run both forwarding tasks with proper cleanup
            await asyncio.gather(
                forward_to_upstream(), 
                forward_to_client(),
                return_exceptions=True
            )
                
        except Exception as e:
            logging.error(f"Kernel WebSocket proxy error: {e}")
        finally:
            # Clean up connections
            if upstream:
                try:
                    await upstream.close()
                except Exception:
                    pass
            
            # Only close websocket if not already closed
            try:
                if websocket.client_state.name != "DISCONNECTED":
                    await websocket.close()
            except Exception:
                pass
    
    # App lifecycle events
    @app.on_event("startup")
    async def startup_event():
        """Start Jupyter server when app starts"""
        try:
            logging.info("Starting Jupyter Notebook server...")
            notebook_manager.start()
            logging.info(f"Jupyter server started on port {notebook_manager.port}")
        except Exception as e:
            logging.error(f"Failed to start Jupyter server: {e}")
            # Don't fail app startup, but log the error
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Stop Jupyter server when app shuts down"""
        try:
            logging.info("Stopping Jupyter Notebook server...")
            notebook_manager.stop()
            logging.info("Jupyter server stopped")
        except Exception as e:
            logging.error(f"Error stopping Jupyter server: {e}")

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
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    # Default port from the README
    port = int(os.getenv("PORT", 5317))
    run_server(port=port) 