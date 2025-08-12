"""
Notebook Routes Module for htmx-based nbscribe

FastAPI routes for notebook operations with htmx integration.
Supports cell editing, execution, and SSE streaming.
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from . import nbio
from . import kernel

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Pydantic models for request bodies
class CellSaveRequest(BaseModel):
    content: str


def create_notebook_router(templates: Jinja2Templates) -> APIRouter:
    """
    Create and configure the notebook router with template access.
    
    Args:
        templates: Jinja2Templates instance for rendering
        
    Returns:
        Configured APIRouter
    """
    # Create a new router instance for this function
    local_router = APIRouter()
    
    # IMPORTANT: More specific routes must be defined BEFORE catch-all routes
    # Define all cell-specific routes first, then the general notebook route
    
    @local_router.get("/debug/test")
    async def debug_test():
        """Simple debug test route"""
        logger.info("DEBUG TEST ROUTE CALLED")
        return {"status": "debug test works"}
    
    @local_router.get("/nb/{path:path}/cells/{cell_id}/stream")
    async def stream_cell_execution(path: str, cell_id: str):
        """
        Stream cell execution output via Server-Sent Events.
        
        Args:
            path: Notebook file path
            cell_id: Cell ID to execute
            
        Returns:
            StreamingResponse with SSE events
        """
        try:
            # Read notebook to get cell content (back to simple approach)
            nb = nbio.read_nb(path)
            cell = nbio.get_cell_by_id(nb, cell_id)
            
            if not cell:
                raise HTTPException(status_code=404, detail=f"Cell {cell_id} not found")
            
            if cell.get("cell_type") != "code":
                raise HTTPException(status_code=400, detail="Only code cells can be executed")
            
            # Get cell source code
            source = cell.get("source", [])
            if isinstance(source, list):
                code = ''.join(source)
            else:
                code = str(source)
            
            print(f"DEBUG SSE: About to execute code: {repr(code)}")
            
            # Stream execution
            async def event_stream():
                try:
                    async for sse_event in kernel.execute_stream(path, code):
                        yield sse_event
                except Exception as e:
                    logger.error(f"Error in execution stream: {e}")
                    error_event = kernel.format_sse_event('error', {'message': str(e)})
                    yield error_event
            
            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Cache-Control"
                }
            )
            
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            logger.error(f"Error streaming execution for cell {cell_id} in {path}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return an error SSE stream instead of HTTP 500 to stop retries
            async def error_stream():
                error_html = f'<div class="output error"><pre>Error: {str(e)}</pre></div>'
                yield f"event: message\ndata: {error_html}\n\n"
                yield f"event: close\ndata: \n\n"
            
            return StreamingResponse(error_stream(), media_type="text/plain")

    @local_router.get("/nb/{path:path}/cells/{cell_id}/execute")
    async def execute_cell_simple(
        request: Request,
        path: str,
        cell_id: str
    ):
        """
        Simple synchronous cell execution - no SSE complexity.
        
        Args:
            request: FastAPI request object
            path: Notebook file path
            cell_id: Cell ID to execute
            
        Returns:
            HTML fragment with execution results
        """
        try:
            # Read notebook to get cell content
            nb = nbio.read_nb(path)
            cell = nbio.get_cell_by_id(nb, cell_id)
            
            if not cell:
                from fastapi.responses import HTMLResponse
                return HTMLResponse(content='<div class="output error"><pre>Cell not found</pre></div>')
            
            if cell.get("cell_type") != "code":
                from fastapi.responses import HTMLResponse
                return HTMLResponse(content='<div class="output error"><pre>Only code cells can be executed</pre></div>')
            
            # Get cell source code
            source = cell.get("source", [])
            if isinstance(source, list):
                code = ''.join(source)
            else:
                code = str(source)
            
            print(f"DEBUG SIMPLE EXECUTE: About to execute code: {repr(code)}")
            
            # Execute synchronously using existing function
            from src.kernel import execute_code_sync
            result = execute_code_sync(path, code)
            
            if result.get("error"):
                from fastapi.responses import HTMLResponse
                return HTMLResponse(content=f'<div class="output error"><pre>Error: {result["error"]}</pre></div>')
            
            # Convert outputs to HTML
            output_html = ""
            for output in result.get("outputs", []):
                output_html += output
            
            if not output_html:
                output_html = '<div class="output"><em>No output</em></div>'
                
            # Return as HTML response so browser renders it properly
            from fastapi.responses import HTMLResponse
            return HTMLResponse(content=output_html)
            
        except Exception as e:
            import traceback
            logger.error(f"Error executing cell {cell_id} in {path}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            from fastapi.responses import HTMLResponse
            return HTMLResponse(content=f'<div class="output error"><pre>Error: {str(e)}</pre></div>')

    @local_router.post("/nb/{path:path}/cells/{cell_id}/save")
    async def save_cell_content(
        request: Request,
        path: str,
        cell_id: str,
        content: str = Form(...)
    ):
        """
        Save cell content to notebook file.
        
        Args:
            request: FastAPI request object
            path: Notebook file path
            cell_id: Cell ID to update
            content: New cell content
            
        Returns:
            HTML fragment with updated cell
        """
        try:
            # Read current notebook
            nb = nbio.read_nb(path)
            
            # Update cell content
            success = nbio.update_cell_source(nb, cell_id, content)
            if not success:
                raise HTTPException(status_code=404, detail=f"Cell {cell_id} not found")
            
            # Write notebook back to file
            nbio.write_nb(path, nb)
            
            # Get updated cell for response
            cell = nbio.get_cell_by_id(nb, cell_id)
            if not cell:
                raise HTTPException(status_code=404, detail=f"Cell {cell_id} not found after update")
            
            # Return updated cell fragment
            return templates.TemplateResponse("_cell.html", {
                "request": request,
                "cell": cell,
                "notebook_path": path,
                "save_success": True
            })
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error saving cell {cell_id} in {path}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @local_router.post("/nb/{path:path}/cells/{cell_id}/run")
    async def trigger_cell_execution(
        request: Request,
        path: str,
        cell_id: str
    ):
        """
        Trigger cell execution and return htmx response to start SSE.
        
        Args:
            request: FastAPI request object
            path: Notebook file path
            cell_id: Cell ID to execute
            
        Returns:
            HTML fragment with SSE connection directive
        """
        try:
            # Get form data manually to avoid FastAPI Form parameter issues
            form_data = await request.form()
            content = form_data.get("content", "")
            print(f"DEBUG: content={content}")
            
            # If content is provided, save it first
            if content and content.strip():
                logger.info(f"Auto-saving cell content before execution: {cell_id}")
                nb = nbio.read_nb(path)
                if nbio.update_cell_source(nb, cell_id, content):
                    nbio.write_nb(path, nb)  # Fixed parameter order!
                    logger.info(f"Cell {cell_id} auto-saved before execution")
                else:
                    logger.error(f"Failed to update cell {cell_id} source")
            
            # Read notebook to get cell content (now updated)
            nb = nbio.read_nb(path)
            cell = nbio.get_cell_by_id(nb, cell_id)
            
            print(f"DEBUG: Retrieved cell source: {cell.get('source', 'NO SOURCE')}")
            logger.info(f"Retrieved cell: {type(cell)} -> {cell}")
            
            if not cell:
                raise HTTPException(status_code=404, detail=f"Cell {cell_id} not found")
            
            if cell.get("cell_type") != "code":
                raise HTTPException(status_code=400, detail="Only code cells can be executed")
            
            # Simple approach: just trigger SSE connection
            # The SSE endpoint will read the cell content directly
            
            # Return SSE-enabled output div
            return templates.TemplateResponse("_cell_output.html", {
                "request": request,
                "cell": cell,
                "notebook_path": path,
                "run_now": True  # Enable SSE connection
            })
            
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            logger.error(f"Error triggering execution for cell {cell_id} in {path}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))

    @local_router.post("/nb/{path:path}/cells/add")
    async def add_cell(
        request: Request,
        path: str,
        after: Optional[str] = None,
        cell_type: str = "code"
    ):
        """
        Add a new cell to the notebook.
        
        Args:
            request: FastAPI request object
            path: Notebook file path
            after: Cell ID to insert after (optional, defaults to end)
            cell_type: Type of cell to create (code or markdown)
            
        Returns:
            HTML fragment with new cell
        """
        try:
            # Read current notebook
            nb = nbio.read_nb(path)
            
            # Determine insertion position
            if after:
                after_index = nbio.find_cell_index(nb, after)
                if after_index is None:
                    raise HTTPException(status_code=404, detail=f"Cell {after} not found")
                insert_index = after_index + 1
            else:
                insert_index = len(nb.get("cells", []))
            
            # Create new cell
            if cell_type == "markdown":
                new_cell = nbio.create_markdown_cell("# New markdown cell")
            else:
                new_cell = nbio.create_code_cell("# New code cell")
            
            # Insert cell
            nbio.insert_cell(nb, new_cell, insert_index)
            
            # Write notebook back to file
            nbio.write_nb(path, nb)
            
            # Return new cell fragment
            return templates.TemplateResponse("_cell.html", {
                "request": request,
                "cell": new_cell,
                "notebook_path": path
            })
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error adding cell to {path}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @local_router.post("/nb/{path:path}/cells/{cell_id}/delete")
    async def delete_cell(
        request: Request,
        path: str,
        cell_id: str
    ):
        """
        Delete a cell from the notebook.
        
        Args:
            request: FastAPI request object
            path: Notebook file path
            cell_id: Cell ID to delete
            
        Returns:
            Empty response (cell removed from DOM)
        """
        try:
            # Read current notebook
            nb = nbio.read_nb(path)
            
            # Delete cell
            success = nbio.delete_cell(nb, cell_id)
            if not success:
                raise HTTPException(status_code=404, detail=f"Cell {cell_id} not found")
            
            # Write notebook back to file
            nbio.write_nb(path, nb)
            
            # Return empty response (htmx will remove the cell from DOM)
            return HTMLResponse(content="", status_code=200)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting cell {cell_id} from {path}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @local_router.post("/nb/{path:path}/save")
    async def save_notebook(
        request: Request,
        path: str
    ):
        """
        Save entire notebook (placeholder for future autosave).
        
        Args:
            request: FastAPI request object
            path: Notebook file path
            
        Returns:
            JSON response with save status
        """
        try:
            # For now, this is a no-op since cells save individually
            # Future: implement autosave or manual save-all functionality
            
            return {"success": True, "message": f"Notebook {path} saved"}
            
        except Exception as e:
            logger.error(f"Error saving notebook {path}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # IMPORTANT: The catch-all route must be LAST
    @local_router.get("/nb/{path:path}", response_class=HTMLResponse)
    async def serve_notebook_page(request: Request, path: str):
        """
        Serve the main notebook page with htmx interface.
        
        Args:
            request: FastAPI request object
            path: Notebook file path
            
        Returns:
            HTML response with notebook interface
        """
        try:
            # Verify notebook exists
            notebook_path = Path(path)
            if not notebook_path.exists():
                raise HTTPException(status_code=404, detail=f"Notebook not found: {path}")
            
            # Read notebook content
            nb = nbio.read_nb(path)
            
            # Render notebook page
            return templates.TemplateResponse("notebook.html", {
                "request": request,
                "notebook_path": path,
                "notebook": nb,
                "title": f"nbscribe - {notebook_path.name}"
            })
            
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Notebook not found: {path}")
        except Exception as e:
            logger.error(f"Error serving notebook page {path}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return local_router
