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
                '/checkpoints?',
                '/contents?content=1&hash=0&',
                '/contents?type=notebook&content=1',
                '/contents?content=0&hash=1&',
                '/kernels?',
                '/sessions?',
                '/kernelspecs?',
                '/me?',
                '/lab/api/settings',
                '/lab/api/translations',
                '/lsp/status',
                '/lab/extensions/',
                '/static/notebook/',
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
        version="0.1.4"
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

    # --- Debug helpers for message flow ---
    def _summarize_message(msg: dict, max_len: int = 160) -> str:
        try:
            role = msg.get('role')
            ts = msg.get('timestamp', '')
            content = str(msg.get('content', ''))
            flat = content.replace('\n', ' ').strip()
            if len(flat) > max_len:
                flat = flat[:max_len] + '…'
            has_tool = 'TOOL:' in content
            has_fence = '```' in content
            return f"{role or '?'} @ {ts}: {flat} [len={len(content)} tool={int(has_tool)} fence={int(has_fence)}]"
        except Exception:
            return str(msg)

    def _log_messages_debug(tag: str, messages: list, limit: int = 5) -> None:
        try:
            logging.info(f"MSG DEBUG [{tag}] total={len(messages)} show_last={min(limit, len(messages))}")
            print(f"MSG DEBUG [{tag}] total={len(messages)} show_last={min(limit, len(messages))}")
            for m in messages[-limit:]:
                line = _summarize_message(m)
                logging.info(f"MSG DEBUG [{tag}] {line}")
                print(f"MSG DEBUG [{tag}] {line}")
        except Exception as e:
            logging.warning(f"MSG DEBUG [{tag}] failed: {e}")
            print(f"MSG DEBUG [{tag}] failed: {e}")

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
        group_id: Optional[str] = None
        seq: Optional[int] = None
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
        """Read notebook content via Jupyter Contents API and return actual .ipynb content"""
        try:
            if not notebook_manager.is_running():
                raise Exception("Jupyter server not running")
            
            # Use Jupyter Contents API to read the notebook
            url = f"http://localhost:{notebook_manager.port}/jupyter/api/contents/{notebook_path}"
            params = {"content": "1", "token": notebook_manager.token}
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                contents_response = response.json()
                
                # Extract and return the actual .ipynb content (not the wrapper)
                return contents_response["content"]
                
        except Exception as e:
            logging.error(f"Error reading notebook {notebook_path}: {e}")
            raise Exception(f"Failed to read notebook: {e}")

    async def write_notebook_content(notebook_path: str, content: dict, notebook_manager) -> dict:
        """Write notebook content via Jupyter Contents API"""
        try:
            if not notebook_manager.is_running():
                raise Exception("Jupyter server not running")
            
            # Ensure all cells have IDs before writing (prevents nbformat warnings)
            content_with_ids = ensure_cell_ids(content)
            
            # Use Jupyter Contents API to write the notebook
            url = f"http://localhost:{notebook_manager.port}/jupyter/api/contents/{notebook_path}"
            params = {"token": notebook_manager.token}
            
            # Prepare the content for the API
            api_content = {
                "type": "notebook",
                "format": "json",
                "content": content_with_ids
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.put(url, json=api_content, params=params)
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            logging.error(f"Error writing notebook {notebook_path}: {e}")
            raise Exception(f"Failed to write notebook: {e}")

    async def trigger_notebook_refresh(notebook_path: str, notebook_manager):
        """Trigger Jupyter UI to refresh notebook content by sending file change event"""
        try:
            if not notebook_manager.is_running():
                return
            
            # Send file change event to Jupyter to trigger UI refresh
            # This mimics what Jupyter does when files are modified externally
            import json
            
            # Simpler approach: Just touch the Contents API to trigger refresh
            # This should signal to Jupyter UI that the file has changed
            contents_url = f"http://localhost:{notebook_manager.port}/jupyter/api/contents/{notebook_path}"
            params = {"token": notebook_manager.token}
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(contents_url, params=params)
                response.raise_for_status()
                logging.info(f"Triggered notebook refresh by touching Contents API for {notebook_path}")
                
        except Exception as e:
            # Don't fail the whole operation if refresh fails
            logging.warning(f"Failed to trigger notebook refresh: {e}")

    def ensure_cell_ids(notebook_content: dict) -> dict:
        """Ensure all cells have stable UUIDs in the format expected by nbformat"""
        import uuid
        
        cells = notebook_content.get("cells", [])
        
        for cell in cells:
            # Check if cell already has a valid top-level ID
            if "id" not in cell or not cell["id"]:
                # Generate a new ID in the format nbformat expects (short hex string)
                cell["id"] = str(uuid.uuid4()).replace("-", "")[:8]
                
            # Ensure metadata exists
            if "metadata" not in cell:
                cell["metadata"] = {}
                
            # Also add ID to metadata for backward compatibility
            if "id" not in cell["metadata"]:
                cell["metadata"]["id"] = cell["id"]
        
        logging.info(f"Ensured cell IDs for {len(cells)} cells")
        return notebook_content

    def find_cell_position(cells: list, cell_id: str) -> Optional[int]:
        """Find the index of a cell by its ID (checks both top-level and metadata locations)"""
        for i, cell in enumerate(cells):
            # Check for ID in multiple locations (notebook format variations)
            current_id = cell.get("id") or cell.get("metadata", {}).get("id")
            if current_id == cell_id:
                return i
        return None

    def resolve_insert_position(cells: list, directive: DirectiveRequest) -> int:
        """Resolve the position where to insert a new cell based on BEFORE/AFTER/POS"""
        if directive.before:
            # Insert before the specified cell
            pos = find_cell_position(cells, directive.before)
            if pos is not None:
                return pos
            raise Exception(f"Cell ID '{directive.before}' not found for BEFORE positioning")

        elif directive.after:
            # Insert after the specified cell
            pos = find_cell_position(cells, directive.after)
            if pos is None:
                raise Exception(f"Cell ID '{directive.after}' not found for AFTER positioning")

            base = pos + 1
            # If group info provided, maintain emission order for same-group AFTERs
            if directive.group_id is not None and directive.seq is not None:
                index = base
                while index < len(cells):
                    meta = cells[index].get("metadata", {})
                    nbmeta = meta.get("nbscribe", {}) if isinstance(meta.get("nbscribe", {}), dict) else {}
                    if (
                        nbmeta.get("anchor_type") == "AFTER"
                        and nbmeta.get("anchor_id") == directive.after
                        and nbmeta.get("group_id") == directive.group_id
                        and isinstance(nbmeta.get("seq"), int)
                        and nbmeta.get("seq") < directive.seq
                    ):
                        index += 1
                        continue
                    # Stop if we hit an item from same group but with seq >= ours, or a different anchor/group
                    break
                return index
            return base

        elif directive.pos is not None:
            # Direct position insertion (fallback)
            if 0 <= directive.pos <= len(cells):
                return directive.pos
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
            notebook_content = await get_notebook_content(notebook_path, notebook_manager)
            
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
            notebook_content = await get_notebook_content(notebook_path, notebook_manager)
            
            # Ensure all cells have IDs
            notebook_content = ensure_cell_ids(notebook_content)
            
            # Resolve where to insert the new cell
            cells = notebook_content.get("cells", [])
            insert_position = resolve_insert_position(cells, directive)
            
            # Debug logging for newline investigation
            logging.info(f"CELL CREATION - Directive code length: {len(directive.code) if directive.code else 0}")
            logging.info(f"CELL CREATION - Code contains \\n: {'\\n' in directive.code if directive.code else False}")
            logging.info(f"CELL CREATION - Code repr: {repr(directive.code[:100]) if directive.code else 'None'}")
            
            # Split code into lines for notebook format (Jupyter expects lines with \n endings)
            if directive.code:
                lines = directive.code.split('\n')
                # Add \n to all lines except empty ones, preserving Jupyter notebook format
                source_lines = []
                for i, line in enumerate(lines):
                    if i == len(lines) - 1 and line == "":
                        # Skip empty last line (result of trailing \n in original code)
                        continue
                    elif i == len(lines) - 1:
                        # Last non-empty line doesn't get \n
                        source_lines.append(line)
                    else:
                        # All other lines get \n
                        source_lines.append(line + "\n")
            else:
                source_lines = [""]
                
            logging.info(f"CELL CREATION - Source lines count: {len(source_lines)}")
            logging.info(f"CELL CREATION - First line: {repr(source_lines[0]) if source_lines else 'None'}")
            logging.info(f"CELL CREATION - Last line: {repr(source_lines[-1]) if source_lines else 'None'}")
            
            # Create new cell with consistent ID format
            cell_id = str(uuid.uuid4()).replace("-", "")[:8]
            new_cell = {
                "cell_type": "code" if directive.language == "python" else "markdown",
                "id": cell_id,  # Top-level ID (nbformat standard)
                "metadata": {
                    "id": cell_id,  # Also in metadata for compatibility
                    "nbscribe": {
                        **({"group_id": directive.group_id, "seq": directive.seq} if directive.group_id is not None and directive.seq is not None else {}),
                        **({"anchor_type": "AFTER", "anchor_id": directive.after} if directive.after else {}),
                        **({"anchor_type": "BEFORE", "anchor_id": directive.before} if (not directive.after and directive.before) else {}),
                        **({"anchor_type": "POS", "anchor_pos": directive.pos} if (directive.after is None and directive.before is None and directive.pos is not None) else {})
                    }
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
            
            # Trigger notebook refresh in Jupyter UI
            await trigger_notebook_refresh(notebook_path, notebook_manager)
            
            # Describe what happened (always include actual position for post-modification lookup)
            if directive.before:
                position_desc = f"before cell {directive.before} at position {insert_position}"
            elif directive.after:
                position_desc = f"after cell {directive.after} at position {insert_position}"
            else:
                position_desc = f"at position {insert_position}"
            
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
            notebook_content = await get_notebook_content(notebook_path, notebook_manager)
            
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
            
            # Trigger notebook refresh in Jupyter UI
            await trigger_notebook_refresh(notebook_path, notebook_manager)
            
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
            notebook_content = await get_notebook_content(notebook_path, notebook_manager)
            
            # Find the cell to delete
            cells = notebook_content.get("cells", [])
            cell_position = find_cell_position(cells, cell_id)
            
            if cell_position is None:
                raise Exception(f"Cell ID '{cell_id}' not found")
            
            # Remove the cell
            deleted_cell = cells.pop(cell_position)
            
            # Write back to notebook
            await write_notebook_content(notebook_path, notebook_content, notebook_manager)
            
            # Trigger notebook refresh in Jupyter UI
            await trigger_notebook_refresh(notebook_path, notebook_manager)
            
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
            from src.llm_interface import get_llm_interface
            from src.conversation_logger import ConversationLogger
            from src.conversation_manager import get_conversation_manager
            
            # Get session ID (fallback to latest if not provided)
            session_id = message.session_id or get_latest_session()
            logging.info(f"CHAT LINEAR DEBUG - Session ID: {session_id}")
            
            # Load existing conversation context
            session_data = load_session_data(session_id)
            logging.info(f"STREAM INCOMING session={session_id} user_len={len(message.message or '')}")
            _log_messages_debug("pre-stream", session_data['messages'], limit=5)
            logging.info(f"CHAT INCOMING session={session_id} user_len={len(message.message or '')}")
            _log_messages_debug("pre-chat", session_data['messages'], limit=5)
            
            # Use linear conversation architecture
            conv_manager = get_conversation_manager()
            llm = get_llm_interface()
            
            # Build linear conversation messages from session data
            linear_messages = conv_manager.build_linear_conversation(
                session_data['messages'], 
                llm.system_prompt or "You are a helpful AI assistant for Jupyter notebooks."
            )
            
            logging.info(f"CHAT LINEAR DEBUG - Built linear conversation with {len(linear_messages)} messages")
            
            # Generate response using new linear conversation interface
            response_text = llm.generate_response(linear_messages, message.message)
            try:
                import re as _re
                has_tool = 'TOOL:' in response_text or 'tool:' in response_text
                fence_count = response_text.count('```')
                dual_fence_tool = bool(_re.search(r"```[\w-]*[\t\x20]*\r?\n[\s\S]*?```[\t\x20]*\r?\n[\t\x20]*```[\w-]*[\t\x20]*\r?\n[\s\S]*?TOOL\s*:\s*.+?```", response_text, _re.IGNORECASE | _re.DOTALL))
                logging.info(f"ASSISTANT TOOL DEBUG (non-stream) has_tool={int(has_tool)} fences={fence_count} dual_fence_tool={int(dual_fence_tool)} len={len(response_text)}")
            except Exception as _e:
                logging.warning(f"ASSISTANT TOOL DEBUG (non-stream) failed: {_e}")
            
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
            _log_messages_debug("post-chat", session_data['messages'], limit=6)
            
            return ChatResponse(
                response=response_text,
                success=True
            )
        except Exception as e:
            # Log the full error with stack trace for debugging
            import traceback
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
            from src.conversation_manager import get_conversation_manager
            
            # Get session ID (fallback to latest if not provided)
            session_id = message.session_id or get_latest_session()
            logging.info(f"STREAM LINEAR DEBUG - Session ID: {session_id}")
            
            # Load existing conversation context
            session_data = load_session_data(session_id)
            
            # Use linear conversation architecture
            conv_manager = get_conversation_manager()
            llm = get_llm_interface()
            
            # Build linear conversation messages from session data
            linear_messages = conv_manager.build_linear_conversation(
                session_data['messages'], 
                llm.system_prompt or "You are a helpful AI assistant for Jupyter notebooks."
            )
            
            logging.info(f"STREAM LINEAR DEBUG - Built linear conversation with {len(linear_messages)} messages")
            
            def stream_response():
                """Generator function for SSE streaming"""
                try:
                    llm = get_llm_interface()
                    full_response = ""
                    
                    # Stream the response using new linear conversation interface
                    for chunk in llm.generate_response_stream(linear_messages, message.message):
                        full_response += chunk
                        # Send chunk as SSE
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                    
                    # Response complete - save to file
                    timestamp = datetime.now().isoformat()
                    try:
                        import re as _re
                        has_tool = 'TOOL:' in full_response or 'tool:' in full_response
                        fence_count = full_response.count('```')
                        dual_fence_tool = bool(_re.search(r"```[\w-]*[\t\x20]*\r?\n[\s\S]*?```[\t\x20]*\r?\n[\t\x20]*```[\w-]*[\t\x20]*\r?\n[\s\S]*?TOOL\s*:\s*.+?```", full_response, _re.IGNORECASE | _re.DOTALL))
                        logging.info(f"ASSISTANT TOOL DEBUG (stream) has_tool={int(has_tool)} fences={fence_count} dual_fence_tool={int(dual_fence_tool)} len={len(full_response)}")
                    except Exception as _e:
                        logging.warning(f"ASSISTANT TOOL DEBUG (stream) failed: {_e}")
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
                    _log_messages_debug("post-stream", session_data['messages'], limit=6)
                    
                    # Send completion signal
                    yield f"data: {json.dumps({'type': 'complete', 'content': full_response})}\n\n"
                    
                except Exception as e:
                    import traceback
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
        _log_messages_debug("serve-notebook-pre", session_data['messages'], limit=5)
        
        # For new notebook sessions, inject complete notebook JSON and greeting
        if session_data['is_new']:
            try:
                # Get the complete notebook JSON
                notebook_content = await get_notebook_content(notebook_path, notebook_manager)
                if notebook_content:
                    from src.conversation_manager import get_conversation_manager
                    conv_manager = get_conversation_manager()

                    # Create initial notebook JSON message as User message
                    notebook_json_message = {
                        'role': 'user',
                        'content': conv_manager.format_notebook_for_ai(notebook_content),
                        'timestamp': datetime.now().isoformat()
                    }

                    # Create assistant greeting that acknowledges seeing the notebook
                    notebook_cell_count = len(notebook_content.get('cells', []))
                    initial_message = {
                        'role': 'assistant',
                        'content': f"Hello! I can see your notebook `{notebook_file.name}` with {notebook_cell_count} cells. I understand the current structure and I'm ready to help you analyze, modify, or extend your code. What would you like to work on?",
                        'timestamp': datetime.now().isoformat()
                    }

                    # Set the messages in proper order: notebook JSON first, then greeting
                    session_data['messages'] = [notebook_json_message, initial_message]
                else:
                    # Fallback if we can't read the notebook
                    initial_message = {
                        'role': 'assistant',
                        'content': f"Hello! I'm ready to help you with `{notebook_file.name}`. I can analyze your notebook and suggest code edits. What would you like to work on?",
                        'timestamp': datetime.now().isoformat()
                    }
                    session_data['messages'] = [initial_message]

                # Save updated conversation
                from src.conversation_logger import ConversationLogger
                logger = ConversationLogger()
                logger.save_conversation_state(session_data['log_file'], session_data['messages'])

            except Exception as e:
                logging.error(f"Error initializing notebook session: {e}")
                # Fallback to simple greeting
                initial_message = {
                    'role': 'assistant',
                    'content': f"Hello! I'm ready to help you with `{notebook_file.name}`. I can analyze your notebook and suggest code edits. What would you like to work on?",
                    'timestamp': datetime.now().isoformat()
                }
                session_data['messages'] = [initial_message]
                from src.conversation_logger import ConversationLogger
                logger = ConversationLogger()
                logger.save_conversation_state(session_data['log_file'], session_data['messages'])
        
        # Build a per-session workspace ID so layout is isolated per session
        workspace_id = f"nbscribe-embed-{session_id}"
        # Seed the workspace layout so JupyterLab restores with left collapsed and single-document mode
        try:
            await seed_workspace_layout(workspace_id, notebook_manager)
        except Exception as e:
            logging.warning(f"Workspace seeding failed (non-fatal): {e}")

        return templates.TemplateResponse("chat.html", {
            "request": request,
            "title": f"nbscribe - {notebook_file.name}",
            "service_name": "nbscribe",
            "version": "0.1.4",
            "conversation_id": session_id,
            "created_at": session_data['created_at'],
            "messages": session_data['messages'],
            "notebook_path": notebook_path,
            # Use JupyterLab route with per-session workspace, so we can use Lab commands and keep UI minimal
            "notebook_iframe_url": f"/jupyter/lab/workspaces/{workspace_id}/tree/{notebook_path}"
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
            "version": "0.1.4",
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
            "version": "0.1.4",
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

    # Debug route to test if our routing is working
    @app.get("/debug/websocket-routes")
    async def debug_websocket_routes():
        """Debug endpoint to check WebSocket routes"""
        routes = []
        for route in app.routes:
            routes.append({
                "path": getattr(route, 'path', str(route)),
                "methods": getattr(route, 'methods', []),
                "type": type(route).__name__
            })
        return {"routes": routes}

    # WebSocket proxy endpoints - MUST be defined before HTTP proxy to take precedence
    @app.websocket("/jupyter/api/events/subscribe") 
    async def jupyter_events_websocket(websocket: WebSocket):
        """Proxy WebSocket for Jupyter event subscriptions"""
        try:
            logging.info(f"📡 EVENTS WEBSOCKET HANDLER CALLED")
            logging.info(f"📡 EVENTS WEBSOCKET QUERY PARAMS: {dict(websocket.query_params)}")
            
            if not notebook_manager.is_running():
                logging.error("📡 EVENTS WEBSOCKET REJECTED: Jupyter server not running")
                await websocket.close(code=1011, reason="Jupyter server not running")
                return
        except Exception as e:
            logging.error(f"📡 EVENTS WEBSOCKET HANDLER ERROR: {e}")
            logging.exception("Full traceback:")
            try:
                await websocket.close(code=1011, reason="Handler error")
            except:
                pass
            return
            
        # Ensure token is included for events WebSocket
        target_url = f"ws://localhost:{notebook_manager.port}/jupyter/api/events/subscribe?token={notebook_manager.token}"
        
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
        logging.info(f"🔌 WEBSOCKET HANDLER CALLED: kernel_id={kernel_id}")
        logging.info(f"🔌 WEBSOCKET QUERY PARAMS: {dict(websocket.query_params)}")
        
        if not notebook_manager.is_running():
            logging.error("🔌 WEBSOCKET REJECTED: Jupyter server not running")
            await websocket.close(code=1011, reason="Jupyter server not running")
            return
        
        # Kernel protection: Check if this kernel is blocked
        if kernel_protection.should_block_kernel(kernel_id):
            logging.info(f"BLOCKED WEBSOCKET: Kernel {kernel_id} is temporarily blocked")
            await websocket.close(code=1011, reason=f"Kernel {kernel_id} temporarily unavailable")
            return
            
        # Allow rapid reconnects for kernel WebSocket channels; do not dedupe here
            
        # Get query parameters (especially session_id) and ensure token is always included
        query_params = dict(websocket.query_params)
        # Always add/override token to ensure authentication
        query_params["token"] = notebook_manager.token
        query_string = "&".join([f"{k}={v}" for k, v in query_params.items()])
        
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
            upstream = await websockets.connect(target_url, max_size=None)
            
            # Track connection state
            client_closed = asyncio.Event()
            upstream_closed = asyncio.Event()
            
            async def forward_to_upstream():
                try:
                    while not client_closed.is_set() and not upstream_closed.is_set():
                        data = await websocket.receive()
                        msg_type = data.get('type')
                        if msg_type == 'websocket.disconnect':
                            client_closed.set()
                            break
                        # Prefer explicit branches for text vs bytes
                        if 'text' in data and data['text'] is not None:
                            if not upstream_closed.is_set():
                                await upstream.send(data['text'])
                        elif 'bytes' in data and data['bytes'] is not None:
                            if not upstream_closed.is_set():
                                await upstream.send(data['bytes'])
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
                        if client_closed.is_set():
                            break
                        try:
                            if isinstance(message, (bytes, bytearray)):
                                await websocket.send_bytes(message)
                            else:
                                await websocket.send_text(message)
                        except Exception:
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

    # TEMPORARILY DISABLE HTTP PROXY to test WebSocket routing
    # @app.api_route("/jupyter/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    # async def jupyter_proxy(request: Request, path: str):
    
    # Jupyter proxy routes - HTTP routes defined after WebSocket routes to avoid conflicts
    # Each route needs its own function in FastAPI
    
    @app.api_route("/jupyter/api/contents/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def jupyter_contents_proxy(request: Request, path: str):
        return await jupyter_specific_proxy_impl(request, f"api/contents/{path}")
    
    @app.api_route("/jupyter/api/sessions/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def jupyter_sessions_proxy(request: Request, path: str):
        return await jupyter_specific_proxy_impl(request, f"api/sessions/{path}")
    
    # Kernel HTTP routes - MUST be more specific to avoid WebSocket conflicts
    @app.api_route("/jupyter/api/kernels", methods=["GET", "POST"])
    async def jupyter_kernels_list_proxy(request: Request):
        return await jupyter_specific_proxy_impl(request, "api/kernels")
    
    @app.api_route("/jupyter/api/kernels/{kernel_id}", methods=["GET", "DELETE", "PATCH"])
    async def jupyter_kernel_proxy(request: Request, kernel_id: str):
        return await jupyter_specific_proxy_impl(request, f"api/kernels/{kernel_id}")
    
    @app.api_route("/jupyter/api/kernels/{kernel_id}/interrupt", methods=["POST"])
    async def jupyter_kernel_interrupt_proxy(request: Request, kernel_id: str):
        return await jupyter_specific_proxy_impl(request, f"api/kernels/{kernel_id}/interrupt")
    
    @app.api_route("/jupyter/api/kernels/{kernel_id}/restart", methods=["POST"])
    async def jupyter_kernel_restart_proxy(request: Request, kernel_id: str):
        return await jupyter_specific_proxy_impl(request, f"api/kernels/{kernel_id}/restart")
    
    @app.api_route("/jupyter/api/settings/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def jupyter_settings_proxy(request: Request, path: str):
        return await jupyter_specific_proxy_impl(request, f"api/settings/{path}")
    
    @app.api_route("/jupyter/api/translations/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def jupyter_translations_proxy(request: Request, path: str):
        return await jupyter_specific_proxy_impl(request, f"api/translations/{path}")
    
    @app.api_route("/jupyter/api/nbconvert/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def jupyter_nbconvert_proxy(request: Request, path: str):
        return await jupyter_specific_proxy_impl(request, f"api/nbconvert/{path}")
    
    @app.api_route("/jupyter/lsp/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def jupyter_lsp_proxy(request: Request, path: str):
        return await jupyter_specific_proxy_impl(request, f"lsp/{path}")
    
    @app.api_route("/jupyter/lab/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def jupyter_lab_proxy(request: Request, path: str):
        return await jupyter_specific_proxy_impl(request, f"lab/{path}")
    
    @app.api_route("/jupyter/notebooks/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def jupyter_notebooks_proxy(request: Request, path: str):
        return await jupyter_specific_proxy_impl(request, f"notebooks/{path}")
    
    @app.api_route("/jupyter/custom/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def jupyter_custom_proxy(request: Request, path: str):
        return await jupyter_specific_proxy_impl(request, f"custom/{path}")
    
    # Catch-all for remaining Jupyter requests not handled by specific routes above
    @app.api_route("/jupyter/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def jupyter_catchall_proxy(request: Request, path: str):
        return await jupyter_specific_proxy_impl(request, path)
    
    async def jupyter_specific_proxy_impl(request: Request, path: str):
        """Proxy specific /jupyter/* requests to the Jupyter Notebook server (WebSocket-safe)"""
        
        # WebSocket upgrade requests should be handled by WebSocket routes defined before this
        if (request.method == "GET" and 
            request.headers.get("upgrade", "").lower() == "websocket"):
            logging.warning(f"🔄 WEBSOCKET UPGRADE REQUEST INTERCEPTED: {path} - This should not happen!")
            logging.warning(f"🔄 Headers: {dict(request.headers)}")
            # If we get here, it means our WebSocket routes aren't working
        
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
        
        # Kernel protection: Check for blocked kernels; avoid dedupe on GETs
        if "api/kernels" in path:
            # Extract kernel ID from path if present
            kernel_id = None
            path_parts = path.split('/')
            if len(path_parts) >= 3 and path_parts[1] == "api" and path_parts[2] == "kernels":
                if len(path_parts) >= 4:
                    kernel_id = path_parts[3]
            
            # Only dedupe non-idempotent methods; allow GET status polling
            if request.method not in ["GET", "HEAD", "OPTIONS"]:
                request_key = f"{request.method}:{path}:{kernel_id}"
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
                
                # Kernel protection: Handle failures (treat stale kernel 404s as noise)
                if "api/kernels" in path:
                    if response.status_code == 404:
                        # Stale kernel IDs are common after reload; log quietly and do not record
                        path_parts = path.split('/')
                        if len(path_parts) >= 4 and path_parts[1] == "api" and path_parts[2] == "kernels":
                            kernel_id = path_parts[3]
                            logging.info(f"KERNEL 404 (stale): {kernel_id}")
                    elif response.status_code in (401, 403) or 500 <= response.status_code < 600:
                        # Only record auth or server errors
                        path_parts = path.split('/')
                        if len(path_parts) >= 4 and path_parts[1] == "api" and path_parts[2] == "kernels":
                            kernel_id = path_parts[3]
                            kernel_protection.record_kernel_failure(kernel_id)
                            logging.warning(f"KERNEL ERROR {response.status_code}: Recorded failure for kernel {kernel_id}")
                
                # Debug logging for kernel API responses
                if "api/kernels" in path:
                    logging.info(f"KERNEL API RESPONSE: {response.status_code} for {request.method} {path}")
                    if response.status_code >= 400 and response.status_code != 404:
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

    async def seed_workspace_layout(workspace_id: str, notebook_manager) -> None:
        """Ensure a workspace exists with left/right areas collapsed and single-document mode.

        Strategy:
        - GET existing workspace; if not found, start from an empty layout.
        - Recursively enforce `collapsed=True` on any left/right area sections and set any `mode` fields to
          "single-document".
        - PUT the updated workspace back so JupyterLab restores it deterministically on load.
        """
        try:
            if not notebook_manager.is_running():
                return

            base_url = f"http://localhost:{notebook_manager.port}/jupyter/lab/api/workspaces/{workspace_id}"
            params = {"token": notebook_manager.token}

            async with httpx.AsyncClient(timeout=10.0) as client:
                # Try to fetch existing workspace
                workspace = None
                try:
                    resp = await client.get(base_url, params=params)
                    logging.info(f"WORKSPACE GET {workspace_id} status={resp.status_code}")
                    if resp.status_code == 200:
                        try:
                            workspace = resp.json()
                            logging.info(f"WORKSPACE GET {workspace_id} data.shell: {workspace.get('data', {}).get('shell', {})}")
                        except Exception as je:
                            logging.warning(f"WORKSPACE GET parse failed: {je}")
                except Exception:
                    workspace = None

                if not isinstance(workspace, dict):
                    workspace = {"id": workspace_id, "data": {}, "metadata": {}}

                data = workspace.get("data") or {}

                # Heuristically enforce collapsed sidebars and single-document mode across data
                def enforce_layout(obj):
                    if isinstance(obj, dict):
                        new_obj = {}
                        for k, v in obj.items():
                            # Normalize likely layout sections
                            if k.lower() in ("left", "leftarea", "right", "rightarea"):
                                # Ensure an object
                                if not isinstance(v, dict):
                                    v = {}
                                v["collapsed"] = True
                                new_obj[k] = enforce_layout(v)
                                continue
                            if k.lower() == "mode" and isinstance(v, str):
                                new_obj[k] = "single-document"
                                continue
                            new_obj[k] = enforce_layout(v)
                        return new_obj
                    if isinstance(obj, list):
                        return [enforce_layout(x) for x in obj]
                    return obj

                data = enforce_layout(data)

                # Also set top-level shell mode and collapsed sidebars explicitly
                shell = data.get("shell")
                if not isinstance(shell, dict):
                    shell = {}
                shell["mode"] = "single-document"
                # Ensure side areas exist and are collapsed
                left_area = shell.get("leftArea")
                if not isinstance(left_area, dict):
                    left_area = {}
                left_area["collapsed"] = True
                shell["leftArea"] = left_area

                right_area = shell.get("rightArea")
                if not isinstance(right_area, dict):
                    right_area = {}
                right_area["collapsed"] = True
                shell["rightArea"] = right_area

                data["shell"] = shell

                # Persist back
                workspace["data"] = data
                logging.info(f"WORKSPACE PUT {workspace_id} data.shell: {data.get('shell', {})}")
                put_resp = await client.put(base_url, params=params, json=workspace)
                logging.info(f"WORKSPACE PUT {workspace_id} status={put_resp.status_code}")

        except Exception as e:
            # Non-fatal; UI can still attempt runtime collapse
            logging.warning(f"Failed to seed workspace layout for {workspace_id}: {e}")
    
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
        """Apply an approved tool directive to the notebook and add state update to conversation"""
        try:
            if not notebook_manager.is_running():
                raise HTTPException(status_code=503, detail="Jupyter server not running")
            
            # Parse the tool directive and execute the corresponding notebook operation
            result_message = ""
            cell_data = None
            operation_type = ""
            
            inserted_index: Optional[int] = None
            inserted_cell_obj = None
            inserted_notebook_path = None

            if directive.tool == "insert_cell":
                # Insert a new cell at the specified position
                result_message = await insert_notebook_cell(
                    directive,
                    notebook_manager
                )
                operation_type = "inserted"
                
                # Get the inserted cell data for conversation update
                notebook_path = extract_notebook_path_from_session(directive.session_id)
                inserted_notebook_path = notebook_path
                logging.info(f"POST-MOD DEBUG - Result message: {result_message}")
                if notebook_path and "position" in result_message:
                    try:
                        notebook_content = await get_notebook_content(notebook_path, notebook_manager)
                        if notebook_content:
                            # Find the cell that was just inserted
                            cells = notebook_content.get('cells', [])
                            logging.info(f"POST-MOD DEBUG - Found {len(cells)} cells after insert")
                            if cells:
                                # Get position from result message
                                import re
                                pos_match = re.search(r'position (\d+)', result_message)
                                logging.info(f"POST-MOD DEBUG - Position regex match: {pos_match}")
                                if pos_match:
                                    position = int(pos_match.group(1))
                                    inserted_index = position
                                    logging.info(f"POST-MOD DEBUG - Extracted position: {position}")
                                    if position < len(cells):
                                        cell_data = cells[position]
                                        inserted_cell_obj = cell_data
                                        logging.info(f"POST-MOD DEBUG - Found cell data: {cell_data.get('id', 'no-id')}")
                                    else:
                                        logging.warning(f"POST-MOD DEBUG - Position {position} >= cell count {len(cells)}")
                                else:
                                    logging.warning(f"POST-MOD DEBUG - No position found in message: {result_message}")
                    except Exception as e:
                        logging.error(f"Error getting inserted cell data: {e}")
                
            elif directive.tool == "edit_cell":
                # Edit an existing cell
                result_message = await edit_notebook_cell(
                    directive.cell_id,
                    directive.code,
                    notebook_manager,
                    directive.session_id
                )
                operation_type = "updated"
                
                # Get the updated cell data for conversation update
                notebook_path = extract_notebook_path_from_session(directive.session_id)
                if notebook_path and directive.cell_id:
                    try:
                        notebook_content = await get_notebook_content(notebook_path, notebook_manager)
                        if notebook_content:
                            # Find the updated cell by ID
                            for cell in notebook_content.get('cells', []):
                                # Check both ID locations for compatibility
                                if (cell.get('id') == directive.cell_id or 
                                    cell.get('metadata', {}).get('id') == directive.cell_id):
                                    cell_data = cell
                                    break
                    except Exception as e:
                        logging.error(f"Error getting updated cell data: {e}")
                
            elif directive.tool == "delete_cell":
                # Delete a cell
                result_message = await delete_notebook_cell(
                    directive.cell_id,
                    notebook_manager,
                    directive.session_id
                )
                operation_type = "deleted"
                # For deletes, we don't need cell_data, just the ID
                
            else:
                raise HTTPException(status_code=400, detail=f"Unknown tool: {directive.tool}")
            
            # Add post-modification message to conversation history
            try:
                from src.conversation_manager import get_conversation_manager
                from src.conversation_logger import ConversationLogger

                conv_manager = get_conversation_manager()

                # Create the appropriate update message
                if operation_type == "deleted":
                    update_message_content = conv_manager.format_cell_deletion(directive.cell_id)
                elif cell_data:
                    update_message_content = conv_manager.format_cell_update(cell_data, operation_type)
                else:
                    # Fallback if we couldn't get cell data
                    update_message_content = conv_manager.format_operation_failure(
                        directive.tool,
                        "Success but cell data unavailable for conversation",
                        directive.cell_id,
                    )

                # Add the update message to conversation history
                if directive.session_id:
                    session_data = load_session_data(directive.session_id)
                    update_message = {
                        'role': 'user',
                        'content': update_message_content,
                        'timestamp': datetime.now().isoformat(),
                    }
                    session_data['messages'].append(update_message)

                    # Save updated conversation
                    logger = ConversationLogger()
                    logger.save_conversation_state(session_data['log_file'], session_data['messages'])

                    logging.info(
                        f"Added {operation_type} cell message to conversation: {directive.session_id}"
                    )
                    _log_messages_debug("post-approve", session_data['messages'], limit=4)

            except Exception as e:
                logging.error(
                    f"Error adding post-modification message to conversation: {e}"
                )
                # Don't fail the whole operation if conversation update fails
            
            return JSONResponse({
                "success": True,
                "message": result_message,
                "directive_id": directive.id,
                **({
                    "inserted_index": inserted_index,
                    "cell": inserted_cell_obj,
                    "notebook_path": inserted_notebook_path,
                } if operation_type == "inserted" else {})
            })
            
        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"Error applying directive: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    
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