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
import time
from collections import defaultdict, deque

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

# Kernel protection system - prevent 404 flooding
class KernelProtection:
    """Circuit breaker and rate limiting for kernel requests"""
    
    def __init__(self):
        # Track failed kernel requests (kernel_id -> list of failure timestamps)
        self.kernel_failures = defaultdict(lambda: deque(maxlen=10))
        # Blocked kernels (kernel_id -> block_until_timestamp)
        self.blocked_kernels = {}
        # Recently seen requests for deduplication
        self.recent_requests = {}
        
    def should_block_kernel(self, kernel_id: str) -> bool:
        """Check if a kernel should be blocked due to repeated failures"""
        now = time.time()
        
        # Check if kernel is currently blocked
        if kernel_id in self.blocked_kernels:
            if now < self.blocked_kernels[kernel_id]:
                return True
            else:
                # Block period expired, remove from blocked list
                del self.blocked_kernels[kernel_id]
                
        return False
    
    def record_kernel_failure(self, kernel_id: str):
        """Record a kernel failure and potentially block future requests"""
        now = time.time()
        failures = self.kernel_failures[kernel_id]
        
        # Add current failure
        failures.append(now)
        
        # Check if we should block this kernel
        # If 5+ failures in last 30 seconds, block for 60 seconds
        recent_failures = [f for f in failures if now - f < 30]
        if len(recent_failures) >= 5:
            self.blocked_kernels[kernel_id] = now + 60  # Block for 60 seconds
            logging.warning(f"Blocking kernel {kernel_id} due to repeated failures (5+ in 30s)")
            return True
            
        return False
    
    def is_duplicate_request(self, request_key: str, window_seconds: float = 1.0) -> bool:
        """Check if this is a duplicate request within the time window"""
        now = time.time()
        
        # Clean old requests
        self.recent_requests = {k: v for k, v in self.recent_requests.items() 
                              if now - v < window_seconds}
        
        # Check if request is duplicate
        if request_key in self.recent_requests:
            return True
            
        # Record this request
        self.recent_requests[request_key] = now
        return False

# Global protection instance
kernel_protection = KernelProtection()

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

    # Data models for API - must be defined before functions that reference them
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
        code: str
        language: str = "python"
        session_id: Optional[str] = None  # Add session context
    
    class CreateNotebookRequest(BaseModel):
        name: str

    # Session management utilities
    def generate_session_id() -> str:
        """Generate a unique session ID with timestamp and milliseconds"""
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        milliseconds = now.microsecond // 1000  # Convert microseconds to milliseconds
        return f"{timestamp}_{milliseconds:03d}"

    def extract_notebook_path_from_session(session_id: str) -> Optional[str]:
        """Extract notebook path from session ID if it's a notebook session"""
        if session_id.startswith("notebook_"):
            # Remove prefix
            path_part = session_id[9:]  # Remove "notebook_" prefix
            
            # Smart reconstruction: handle the encoding we used in session creation
            # Original: notebook_path.replace('/', '_').replace('.', '_')
            # So we need to reverse this carefully
            
            # Check if it ends with common notebook patterns
            if path_part.endswith('_ipynb'):
                # This is likely "filename.ipynb" encoded as "filename_ipynb"
                notebook_path = path_part[:-6] + '.ipynb'  # Replace "_ipynb" with ".ipynb"
            elif path_part.endswith('_py'):
                # This is likely "filename.py" encoded as "filename_py"  
                notebook_path = path_part[:-3] + '.py'
            else:
                # Fallback: try both approaches
                # 1. All underscores to dots (for simple filenames)
                # 2. Underscores to slashes (for paths with directories)
                possible_paths = [
                    path_part.replace('_', '.') + '.ipynb',     # Simple filename approach
                    path_part.replace('_', '/') + '.ipynb',     # Directory approach
                    path_part + '.ipynb'                        # Direct approach
                ]
                
                for path in possible_paths:
                    if Path(path).exists():
                        return path
                
                # Default to the most likely
                notebook_path = path_part.replace('_', '.') + '.ipynb'
            
            logging.info(f"SESSION->PATH: {session_id} -> {notebook_path}")
            return notebook_path
        
        return None

    async def get_notebook_content(notebook_path: str, notebook_manager) -> dict:
        """Read notebook content via Jupyter Contents API"""
        try:
            if not notebook_manager.is_running():
                raise Exception("Jupyter server not running")
            
            # Use Jupyter Contents API to read the notebook
            url = f"http://localhost:{notebook_manager.port}/jupyter/api/contents/{notebook_path}"
            params = {"content": "1", "token": notebook_manager.token}
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            logging.error(f"Error reading notebook {notebook_path}: {e}")
            raise Exception(f"Failed to read notebook: {e}")

    async def write_notebook_content(notebook_path: str, content: dict, notebook_manager) -> dict:
        """Write notebook content via Jupyter Contents API"""
        try:
            if not notebook_manager.is_running():
                raise Exception("Jupyter server not running")
            
            # Use Jupyter Contents API to write the notebook
            url = f"http://localhost:{notebook_manager.port}/jupyter/api/contents/{notebook_path}"
            params = {"token": notebook_manager.token}
            
            # Prepare the content for the API
            api_content = {
                "type": "notebook",
                "format": "json",
                "content": content
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.put(url, json=api_content, params=params)
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            logging.error(f"Error writing notebook {notebook_path}: {e}")
            raise Exception(f"Failed to write notebook: {e}")

    def ensure_cell_ids(notebook_content: dict) -> dict:
        """Ensure all cells have stable UUIDs in metadata"""
        import uuid
        
        cells = notebook_content.get("cells", [])
        
        for cell in cells:
            if "metadata" not in cell:
                cell["metadata"] = {}
            
            # Add cell ID if it doesn't exist
            if "id" not in cell["metadata"]:
                cell["metadata"]["id"] = str(uuid.uuid4())
        
        return notebook_content

    def find_cell_position(cells: list, cell_id: str) -> Optional[int]:
        """Find the index of a cell by its ID"""
        for i, cell in enumerate(cells):
            if cell.get("metadata", {}).get("id") == cell_id:
                return i
        return None

    def resolve_insert_position(cells: list, directive: DirectiveRequest) -> int:
        """Resolve the position where to insert a new cell based on BEFORE/AFTER/POS"""
        if directive.before:
            # Insert before the specified cell
            pos = find_cell_position(cells, directive.before)
            if pos is not None:
                return pos
            else:
                raise Exception(f"Cell ID '{directive.before}' not found for BEFORE positioning")
        
        elif directive.after:
            # Insert after the specified cell
            pos = find_cell_position(cells, directive.after)
            if pos is not None:
                return pos + 1
            else:
                raise Exception(f"Cell ID '{directive.after}' not found for AFTER positioning")
        
        elif directive.pos is not None:
            # Direct position insertion (fallback)
            if 0 <= directive.pos <= len(cells):
                return directive.pos
            else:
                raise Exception(f"Position {directive.pos} is out of bounds (0-{len(cells)})")
        
        else:
            raise Exception("No position specified for insert_cell (need BEFORE, AFTER, or POS)")

    # Notebook modification helper functions
    async def get_notebook_structure(notebook_path: str, notebook_manager) -> Optional[str]:
        """Get a summary of notebook structure for AI context"""
        try:
            if not notebook_path or not notebook_manager.is_running():
                return None
            
            # Read current notebook content
            notebook_data = await get_notebook_content(notebook_path, notebook_manager)
            notebook_content = notebook_data["content"]
            
            # Ensure all cells have IDs
            notebook_content = ensure_cell_ids(notebook_content)
            
            # Build structure summary
            cells = notebook_content.get("cells", [])
            if not cells:
                return f"Current notebook '{notebook_path}' is empty (0 cells)."
            
            structure_lines = [f"Current notebook structure ({len(cells)} cells):"]
            
            for i, cell in enumerate(cells):
                cell_id = cell.get("metadata", {}).get("id", f"cell-{i}")
                cell_type = cell.get("cell_type", "unknown")
                
                # Get first line of source for preview
                source = cell.get("source", [])
                if isinstance(source, list):
                    preview = source[0] if source else ""
                else:
                    preview = str(source)
                
                # Truncate preview to reasonable length
                preview = preview.strip()[:60]
                if len(preview) == 60:
                    preview += "..."
                
                structure_lines.append(f"- {cell_id}: [{cell_type}] {preview}")
            
            structure_lines.append(f"\nAvailable for BEFORE/AFTER references: {', '.join([cell.get('metadata', {}).get('id', f'cell-{i}') for i, cell in enumerate(cells)])}")
            
            return "\n".join(structure_lines)
            
        except Exception as e:
            logging.error(f"Error getting notebook structure: {e}")
            return None

    async def insert_notebook_cell(directive: DirectiveRequest, notebook_manager) -> str:
        """Insert a new cell into the notebook using BEFORE/AFTER or POS"""
        try:
            import uuid
            import nbformat
            
            # Extract notebook path from session context
            session_id = directive.session_id
            if session_id:
                notebook_path = extract_notebook_path_from_session(session_id)
            else:
                # Fallback: look for any open notebook in current directory
                notebooks = [f for f in Path(".").glob("*.ipynb")]
                if not notebooks:
                    raise Exception("No notebook found - please open a specific notebook")
                notebook_path = str(notebooks[0])  # Use first found notebook
            
            if not notebook_path:
                raise Exception("Cannot determine notebook path from session context")
            
            logging.info(f"Working with notebook: {notebook_path}")
            
            # Read current notebook content
            notebook_data = await get_notebook_content(notebook_path, notebook_manager)
            notebook_content = notebook_data["content"]
            
            # Ensure all cells have IDs
            notebook_content = ensure_cell_ids(notebook_content)
            
            # Resolve where to insert the new cell
            cells = notebook_content.get("cells", [])
            insert_position = resolve_insert_position(cells, directive)
            
            # Debug logging for newline investigation
            logging.info(f"CELL CREATION - Directive code length: {len(directive.code) if directive.code else 0}")
            logging.info(f"CELL CREATION - Code contains \\n: {'\\n' in directive.code if directive.code else False}")
            logging.info(f"CELL CREATION - Code repr: {repr(directive.code[:100]) if directive.code else 'None'}")
            
            # Split code into lines for notebook format
            source_lines = directive.code.split('\n') if directive.code else [""]
            logging.info(f"CELL CREATION - Source lines count: {len(source_lines)}")
            logging.info(f"CELL CREATION - First line: {repr(source_lines[0]) if source_lines else 'None'}")
            
            # Create new cell with UUID
            new_cell = {
                "cell_type": "code" if directive.language == "python" else "markdown",
                "metadata": {
                    "id": str(uuid.uuid4())
                },
                "source": source_lines
            }
            
            # Add cell-specific fields based on type
            if new_cell["cell_type"] == "code":
                new_cell["execution_count"] = None
                new_cell["outputs"] = []
            
            # Insert the new cell
            cells.insert(insert_position, new_cell)
            
            # Write back to notebook
            await write_notebook_content(notebook_path, notebook_content, notebook_manager)
            
            # Describe what happened
            if directive.before:
                position_desc = f"before cell {directive.before}"
            elif directive.after:
                position_desc = f"after cell {directive.after}"
            else:
                position_desc = f"at position {directive.pos}"
            
            return f"Cell inserted {position_desc} in {notebook_path}"
            
        except Exception as e:
            logging.error(f"Error inserting cell: {e}")
            raise Exception(f"Failed to insert cell: {e}")
    
    async def edit_notebook_cell(cell_id: str, code: str, notebook_manager, session_id: str = None) -> str:
        """Edit an existing cell in the notebook"""
        try:
            # Extract notebook path from session context
            notebook_path = None
            if session_id:
                notebook_path = extract_notebook_path_from_session(session_id)
            
            if not notebook_path:
                # Fallback: look for any open notebook in current directory
                notebooks = [f for f in Path(".").glob("*.ipynb")]
                if not notebooks:
                    raise Exception("No notebook found - please open a specific notebook")
                notebook_path = str(notebooks[0])
            
            logging.info(f"Editing cell {cell_id} in notebook: {notebook_path}")
            
            # Read current notebook content
            notebook_data = await get_notebook_content(notebook_path, notebook_manager)
            notebook_content = notebook_data["content"]
            
            # Ensure all cells have IDs
            notebook_content = ensure_cell_ids(notebook_content)
            
            # Find the cell to edit
            cells = notebook_content.get("cells", [])
            cell_position = find_cell_position(cells, cell_id)
            
            if cell_position is None:
                raise Exception(f"Cell ID '{cell_id}' not found")
            
            # Update the cell source
            cell = cells[cell_position]
            cell["source"] = code.split('\n') if code else [""]
            
            # Clear outputs for code cells when editing
            if cell.get("cell_type") == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
            
            # Write back to notebook
            await write_notebook_content(notebook_path, notebook_content, notebook_manager)
            
            return f"Cell {cell_id} edited in {notebook_path}"
            
        except Exception as e:
            logging.error(f"Error editing cell: {e}")
            raise Exception(f"Failed to edit cell: {e}")
    
    async def delete_notebook_cell(cell_id: str, notebook_manager, session_id: str = None) -> str:
        """Delete a cell from the notebook"""
        try:
            # Extract notebook path from session context
            notebook_path = None
            if session_id:
                notebook_path = extract_notebook_path_from_session(session_id)
            
            if not notebook_path:
                # Fallback: look for any open notebook in current directory
                notebooks = [f for f in Path(".").glob("*.ipynb")]
                if not notebooks:
                    raise Exception("No notebook found - please open a specific notebook")
                notebook_path = str(notebooks[0])
            
            logging.info(f"Deleting cell {cell_id} from notebook: {notebook_path}")
            
            # Read current notebook content
            notebook_data = await get_notebook_content(notebook_path, notebook_manager)
            notebook_content = notebook_data["content"]
            
            # Find the cell to delete
            cells = notebook_content.get("cells", [])
            cell_position = find_cell_position(cells, cell_id)
            
            if cell_position is None:
                raise Exception(f"Cell ID '{cell_id}' not found")
            
            # Remove the cell
            deleted_cell = cells.pop(cell_position)
            
            # Write back to notebook
            await write_notebook_content(notebook_path, notebook_content, notebook_manager)
            
            return f"Cell {cell_id} deleted from {notebook_path}"
            
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
            logging.info(f"CHAT CONTEXT DEBUG - Session ID: {session_id}")
            
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
            
            # Get notebook context if this is a notebook session
            notebook_context = None
            notebook_path = extract_notebook_path_from_session(session_id)
            logging.info(f"CHAT CONTEXT DEBUG - Extracted notebook path: {notebook_path}")
            if notebook_path:
                notebook_context = await get_notebook_structure(notebook_path, notebook_manager)
                logging.info(f"CHAT CONTEXT DEBUG - Notebook context length: {len(notebook_context) if notebook_context else 0}")
                if notebook_context:
                    logging.info(f"CHAT CONTEXT DEBUG - Notebook context preview: {notebook_context[:200]}...")
            else:
                logging.info(f"CHAT CONTEXT DEBUG - No notebook path found for session")
            
            # Generate response using LLM with conversation and notebook context
            response_text = generate_response(message.message, conversation_context, notebook_context)
            
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
            logging.info(f"STREAM CONTEXT DEBUG - Session ID: {session_id}")
            
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
            
            # Get notebook context if this is a notebook session
            notebook_context = None
            notebook_path = extract_notebook_path_from_session(session_id)
            logging.info(f"STREAM CONTEXT DEBUG - Extracted notebook path: {notebook_path}")
            if notebook_path:
                notebook_context = await get_notebook_structure(notebook_path, notebook_manager)
                logging.info(f"STREAM CONTEXT DEBUG - Notebook context length: {len(notebook_context) if notebook_context else 0}")
                if notebook_context:
                    logging.info(f"STREAM CONTEXT DEBUG - Notebook context preview: {notebook_context[:200]}...")
            else:
                logging.info(f"STREAM CONTEXT DEBUG - No notebook path found for session")
            
            def stream_response():
                """Generator function for SSE streaming"""
                try:
                    llm = get_llm_interface()
                    full_response = ""
                    
                    # Stream the response
                    for chunk in llm.generate_response_stream(message.message, conversation_context, notebook_context):
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
        
        # Kernel protection: Check for blocked kernels and duplicate requests
        if "api/kernels" in path:
            # Extract kernel ID from path if present
            kernel_id = None
            path_parts = path.split('/')
            if len(path_parts) >= 3 and path_parts[1] == "api" and path_parts[2] == "kernels":
                if len(path_parts) >= 4:
                    kernel_id = path_parts[3]
            
            # Create request key for deduplication
            request_key = f"{request.method}:{path}:{kernel_id}"
            
            # Check for duplicate requests (rapid retries)
            if kernel_protection.is_duplicate_request(request_key, window_seconds=0.5):
                logging.info(f"BLOCKED DUPLICATE: {request_key}")
                raise HTTPException(status_code=429, detail="Too many requests - duplicate detected")
            
            # Check if kernel should be blocked
            if kernel_id and kernel_protection.should_block_kernel(kernel_id):
                logging.info(f"BLOCKED KERNEL: {kernel_id} is temporarily blocked due to repeated failures")
                raise HTTPException(status_code=503, detail=f"Kernel {kernel_id} temporarily unavailable")
            
            logging.info(f"KERNEL API: {request.method} {path} -> {target_url}")
        
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
                
                # Kernel protection: Handle failures
                if "api/kernels" in path and response.status_code == 404:
                    # Extract kernel ID and record failure
                    path_parts = path.split('/')
                    if len(path_parts) >= 4 and path_parts[1] == "api" and path_parts[2] == "kernels":
                        kernel_id = path_parts[3]
                        kernel_protection.record_kernel_failure(kernel_id)
                        logging.warning(f"KERNEL 404: Recorded failure for kernel {kernel_id}")
                
                # Debug logging for kernel API responses
                if "api/kernels" in path:
                    logging.info(f"KERNEL API RESPONSE: {response.status_code} for {request.method} {path}")
                    if response.status_code >= 400:
                        try:
                            response_text = await response.aread()
                            logging.error(f"KERNEL API ERROR RESPONSE: {response_text}")
                        except Exception:
                            pass
                
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
                    directive,
                    notebook_manager
                )
                
            elif directive.tool == "edit_cell":
                # Edit an existing cell
                result_message = await edit_notebook_cell(
                    directive.cell_id,
                    directive.code,
                    notebook_manager,
                    directive.session_id
                )
                
            elif directive.tool == "delete_cell":
                # Delete a cell
                result_message = await delete_notebook_cell(
                    directive.cell_id,
                    notebook_manager,
                    directive.session_id
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
        
        # Kernel protection: Check if this kernel is blocked
        if kernel_protection.should_block_kernel(kernel_id):
            logging.info(f"BLOCKED WEBSOCKET: Kernel {kernel_id} is temporarily blocked")
            await websocket.close(code=1011, reason=f"Kernel {kernel_id} temporarily unavailable")
            return
            
        # Check for duplicate WebSocket requests (same kernel, rapid succession)
        ws_request_key = f"WS:{kernel_id}"
        if kernel_protection.is_duplicate_request(ws_request_key, window_seconds=2.0):
            logging.info(f"BLOCKED DUPLICATE WEBSOCKET: {kernel_id}")
            await websocket.close(code=1011, reason="Duplicate WebSocket connection blocked")
            return
            
        # Get query parameters (especially session_id)
        query_params = dict(websocket.query_params)
        query_string = "&".join([f"{k}={v}" for k, v in query_params.items()])
        if "token" not in query_params:
            query_string += f"&token={notebook_manager.token}"
        
        target_url = f"ws://localhost:{notebook_manager.port}/jupyter/api/kernels/{kernel_id}/channels"
        if query_string:
            target_url += f"?{query_string}"
        
        logging.info(f"WEBSOCKET CONNECTING: {target_url}")
        logging.info(f"WEBSOCKET QUERY PARAMS: {query_params}")
        
        upstream = None
        try:
            import websockets
            import asyncio
            
            await websocket.accept()
            
            # Connect to upstream Jupyter server - token already in URL
            logging.info(f"WEBSOCKET FINAL URL: {target_url}")
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
            # Enhanced error logging for WebSocket connection issues
            logging.error(f"WEBSOCKET CONNECTION ERROR: {type(e).__name__}: {e}")
            logging.error(f"WEBSOCKET FAILED URL: {target_url}")
            
            # Record kernel failure if it's a connection error (likely 404 or 403)
            if any(code in str(e) for code in ["404", "403", "rejected", "forbidden"]):
                kernel_protection.record_kernel_failure(kernel_id)
                logging.warning(f"WEBSOCKET AUTH/404: Recorded failure for kernel {kernel_id}")
            
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

    @app.get("/debug/jupyter-status")
    async def debug_jupyter_status():
        """Debug endpoint to check Jupyter server status and test kernel creation"""
        if not notebook_manager.is_running():
            return {"error": "Jupyter server not running"}
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Test basic API status
                status_url = f"http://localhost:{notebook_manager.port}/jupyter/api/status?token={notebook_manager.token}"
                status_response = await client.get(status_url)
                
                # Test kernel list
                kernels_url = f"http://localhost:{notebook_manager.port}/jupyter/api/kernels?token={notebook_manager.token}"
                kernels_response = await client.get(kernels_url)
                
                # Try creating a kernel
                create_kernel_url = f"http://localhost:{notebook_manager.port}/jupyter/api/kernels?token={notebook_manager.token}"
                create_response = await client.post(create_kernel_url, json={"name": "python3"})
                
                return {
                    "jupyter_port": notebook_manager.port,
                    "jupyter_token": notebook_manager.token,
                    "protection_status": {
                        "blocked_kernels": len(kernel_protection.blocked_kernels),
                        "failed_kernels": len(kernel_protection.kernel_failures),
                        "recent_requests": len(kernel_protection.recent_requests)
                    },
                    "status_check": {
                        "status_code": status_response.status_code,
                        "response": status_response.json() if status_response.status_code < 400 else status_response.text
                    },
                    "kernels_list": {
                        "status_code": kernels_response.status_code,
                        "response": kernels_response.json() if kernels_response.status_code < 400 else kernels_response.text
                    },
                    "create_kernel": {
                        "status_code": create_response.status_code,
                        "response": create_response.json() if create_response.status_code < 400 else create_response.text
                    }
                }
        except Exception as e:
            return {"error": f"Failed to test Jupyter API: {e}"}
    
    @app.post("/debug/clear-kernel-blocks")
    async def clear_kernel_blocks():
        """Clear all blocked kernels and reset protection state"""
        kernel_protection.blocked_kernels.clear()
        kernel_protection.kernel_failures.clear()
        kernel_protection.recent_requests.clear()
        
        return {
            "success": True,
            "message": "All kernel blocks and protection state cleared"
        }

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