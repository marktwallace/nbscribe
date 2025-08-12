"""
Kernel Management Module for htmx-based nbscribe

Simple synchronous kernel management using jupyter_client.
Supports SSE streaming for cell execution via threading.
"""

import json
import queue
import threading
import time
from typing import AsyncIterator, Dict, Optional, Any
import logging

from jupyter_client import KernelManager
from jupyter_client.blocking import BlockingKernelClient  
from jupyter_client.session import Session
import bleach

logger = logging.getLogger(__name__)


class SingleKernelManager:
    """
    Manages a single kernel instance for the entire application.
    Simple design for single-notebook AI assistant focus.
    """
    
    def __init__(self):
        self.kernel: Optional[KernelManager] = None
        self.session: Optional[Session] = None
        self.last_used: float = 0
        self._lock = threading.Lock()
    
    def get_kernel(self) -> KernelManager:
        """
        Get or create the single application kernel.
        
        Returns:
            KernelManager instance (creates new one if needed)
        """
        with self._lock:
            # Update last used time
            self.last_used = time.time()
            
            if self.kernel is not None:
                if self.kernel.is_alive():
                    logger.debug("Reusing existing kernel")
                    return self.kernel
                else:
                    # Kernel died - log and create new one
                    logger.error("Kernel died! Creating new kernel instance")
                    self.shutdown_kernel()
            
            # Create new kernel
            logger.info("Starting new kernel")
            self.kernel = KernelManager(kernel_name='python3')
            self.kernel.start_kernel()
            
            # Wait for kernel to be ready
            client = self.kernel.blocking_client()
            for i in range(10):  # Wait up to 10 seconds
                if client.is_alive():
                    break
                time.sleep(1)
            else:
                logger.error("Kernel failed to start!")
                self.kernel.shutdown_kernel()
                self.kernel = None
                return None
            
            # Create session for message handling
            self.session = Session()
            
            logger.info(f"Started kernel {self.kernel.kernel_id}")
            return self.kernel
    
    def shutdown_kernel(self) -> None:
        """
        Shutdown the single kernel instance.
        """
        if self.kernel is not None:
            try:
                if self.kernel.is_alive():
                    self.kernel.shutdown_kernel()
                    logger.info("Shut down kernel")
            except Exception as e:
                logger.error(f"Error shutting down kernel: {e}")
            finally:
                # Clean up references
                self.kernel = None
                self.session = None
    
    def shutdown_all_kernels(self) -> None:
        """Shutdown the single kernel."""
        self.shutdown_kernel()
        logger.info("Kernel shut down")


# Global single kernel instance
_kernel_manager: Optional[SingleKernelManager] = None


def get_kernel_manager() -> SingleKernelManager:
    """Get or create the global single kernel manager."""
    global _kernel_manager
    if _kernel_manager is None:
        _kernel_manager = SingleKernelManager()
    return _kernel_manager


def execute_code_sync(nb_path: str, code: str) -> Dict[str, Any]:
    """
    Execute code synchronously and return results.
    
    Args:
        nb_path: Notebook path (used as kernel identifier)  
        code: Python code to execute
        
    Returns:
        Dictionary with execution results and outputs
    """
    kernel_manager = get_kernel_manager()
    
    try:
        # Get the single application kernel
        kernel = kernel_manager.get_kernel()
        
        if kernel is None:
            logger.error(f"Failed to get kernel for {nb_path}")
            return {"error": "Failed to start kernel"}
        
        # Get the blocking kernel client (forces synchronous operations)
        client = kernel.blocking_client()
        
        # Check if client is connected
        if not client.is_alive():
            logger.error(f"Kernel client for {nb_path} is not connected")
            return {"error": "Kernel client is not connected"}
        
        # Send execution request
        msg_id = client.execute(code, silent=False, store_history=True)
        logger.info(f"Sent execute request with msg_id: {msg_id}")
        
        # Collect outputs
        outputs = []
        execution_complete = False
        start_time = time.time()
        timeout_seconds = 30.0
        message_count = 0
        
        while not execution_complete:
            # Check for timeout
            if time.time() - start_time > timeout_seconds:
                logger.warning(f"Execution timeout for {nb_path}")
                return {"error": "Execution timeout"}
            
            # Get message from iopub channel
            try:
                msg = client.get_iopub_msg(timeout=1.0)
                message_count += 1
                logger.info(f"Received message #{message_count}: {type(msg)}")
            except queue.Empty:
                # No message available, continue loop
                continue
            except Exception as e:
                logger.error(f"Error getting iopub message: {e}")
                continue
            
            if not msg:
                continue
            
            logger.info(f"Message content: {msg.get('header', {}).get('msg_type')}")
                
            # Check if this message is for our execution
            parent_header = msg.get('parent_header', {})
            if parent_header.get('msg_id') != msg_id:
                # Skip messages from other executions
                logger.debug(f"Skipping message from different execution: {parent_header.get('msg_id')}")
                continue
                
            msg_type = msg['header']['msg_type']
            logger.info(f"Processing message type: {msg_type} for our execution")
            
            if msg_type == 'status':
                execution_state = msg['content']['execution_state']
                if execution_state == 'idle':
                    execution_complete = True
                    break
            
            elif msg_type in ('stream', 'error', 'execute_result', 'display_data'):
                # Collect output
                output = render_output_message(msg)
                if output:
                    outputs.append(output)
        
        return {"success": True, "outputs": outputs}
        
    except Exception as e:
        logger.error(f"Error executing code in {nb_path}: {e}")
        return {"error": f"Execution failed: {e}"}


async def execute_stream(nb_path: str, code: str) -> AsyncIterator[str]:
    """
    Execute code and stream output as SSE events using threading.
    
    Args:
        nb_path: Notebook path (used as kernel identifier)
        code: Python code to execute
        
    Yields:
        SSE-formatted strings for streaming to client
    """
    # Create a queue to communicate between threads
    result_queue = queue.Queue()
    
    def run_execution():
        """Run the synchronous execution in a thread."""
        try:
            # Use the global single kernel manager
            kernel_manager = get_kernel_manager()
            
            # Get the single kernel (no nb_path needed anymore!)
            kernel = kernel_manager.get_kernel()
            if kernel is None:
                result_queue.put(("error", "Failed to start kernel"))
                return
            
            client = kernel.blocking_client()
            if not client.is_alive():
                result_queue.put(("error", "Kernel client not alive"))
                return
            
            # Execute code
            msg_id = client.execute(code, silent=False, store_history=True)
            print(f"DEBUG KERNEL: Execution started with msg_id: {msg_id}")
            
            # Collect outputs
            outputs = []
            execution_complete = False
            start_time = time.time()
            timeout_seconds = 30.0  # Increased from 10 to 30 seconds
            
            while not execution_complete and (time.time() - start_time < timeout_seconds):
                try:
                    msg = client.get_iopub_msg(timeout=1.0)
                    
                    # Check if this message is for our execution
                    parent_header = msg.get('parent_header', {})
                    if parent_header.get('msg_id') != msg_id:
                        continue
                        
                    msg_type = msg['header']['msg_type']
                    
                    if msg_type == 'status':
                        execution_state = msg['content']['execution_state']
                        if execution_state == 'idle':
                            execution_complete = True
                            break
                    
                    elif msg_type in ('stream', 'error', 'execute_result', 'display_data'):
                        output = render_output_message(msg)
                        if output:
                            outputs.append(output)
                            print(f"DEBUG KERNEL: Collected {msg_type} output")
                            
                except Exception as e:
                    # Only log non-timeout exceptions to reduce noise
                    if "Empty" not in str(e):
                        print(f"DEBUG KERNEL: Exception in message loop: {type(e).__name__}: {e}")
                    continue  # Timeout or other error, keep trying
            
            if time.time() - start_time >= timeout_seconds:
                print(f"DEBUG KERNEL: Execution timed out after {timeout_seconds} seconds")
                result_queue.put(("error", f"Execution timed out after {timeout_seconds} seconds"))
            else:
                print(f"DEBUG KERNEL: Execution complete, collected {len(outputs)} outputs")
                result_queue.put(("result", {"success": True, "outputs": outputs}))
            
        except Exception as e:
            print(f"DEBUG KERNEL: Exception in execution thread: {e}")
            result_queue.put(("error", str(e)))
    
    # Start execution in background thread
    thread = threading.Thread(target=run_execution)
    thread.daemon = True
    thread.start()
    
    # Stream initial status
    yield format_sse_event('status', {'status': 'busy'})
    yield format_sse_event('output', {'html': '<div class="output">Starting execution...</div>'})
    
    # Wait for execution to complete and stream results
    execution_complete = False
    start_time = time.time()
    timeout_seconds = 15.0
    
    while not execution_complete:
        try:
            # Check for timeout
            if time.time() - start_time > timeout_seconds:
                yield format_sse_event('error', {'message': 'Execution timeout'})
                break
            
            # Check for results
            try:
                event_type, data = result_queue.get(timeout=0.1)
                
                if event_type == "result":
                    if "error" in data:
                        yield format_sse_event('error', {'message': data["error"]})
                    else:
                        # Stream all outputs
                        outputs_list = data.get("outputs", [])
                        print(f"DEBUG SSE: Streaming {len(outputs_list)} outputs")
                        for output in outputs_list:
                            yield format_sse_event('output', {'html': output})
                        yield format_sse_event('status', {'status': 'idle'})
                    execution_complete = True
                    
                elif event_type == "error":
                    yield format_sse_event('error', {'message': data})
                    execution_complete = True
                    
            except queue.Empty:
                # No result yet, continue waiting
                continue
                
        except Exception as e:
            logger.error(f"Error in execution stream: {e}")
            yield format_sse_event('error', {'message': str(e)})
            break
    
    # Send completion event
    print("DEBUG SSE: Sending close event")
    yield format_sse_event('close', {})


def format_sse_event(event_type: str, data: Dict[str, Any]) -> str:
    """
    Format data as Server-Sent Event for htmx SSE.
    
    Args:
        event_type: Type of event (output, error, status, close)
        data: Event data dictionary
        
    Returns:
        SSE-formatted string
    """
    if event_type == 'output':
        # Send HTML content with message event for sse-swap="message"
        return f"event: message\ndata: {data.get('html', '')}\n\n"
    elif event_type == 'close':
        # Send close event to terminate SSE connection
        return f"event: close\ndata: \n\n"
    elif event_type == 'error':
        # Send error as HTML
        error_html = f'<div class="output error"><pre>{data.get("message", "Unknown error")}</pre></div>'
        return f"data: {error_html}\n\n"
    else:
        # Skip status events for simplicity
        return ""


def render_output_message(msg: Dict[str, Any]) -> str:
    """
    Render a kernel output message to HTML.
    
    Args:
        msg: Kernel message dictionary
        
    Returns:
        HTML string for the output
    """
    msg_type = msg['header']['msg_type']
    content = msg['content']
    
    if msg_type == 'stream':
        # Standard output/error stream
        stream_name = content['name']  # stdout or stderr
        text = content['text']
        
        # Escape HTML and preserve whitespace
        escaped_text = bleach.clean(text, strip=True)
        return f'<div class="output stream-{stream_name}"><pre>{escaped_text}</pre></div>'
    
    elif msg_type == 'execute_result':
        # Execution result (e.g., expression evaluation)
        data = content['data']
        
        if 'text/html' in data:
            # HTML output - sanitize but allow basic formatting
            html = data['text/html']
            clean_html = bleach.clean(
                html,
                tags=['div', 'span', 'p', 'br', 'b', 'i', 'strong', 'em', 'pre', 'code'],
                attributes={'*': ['class', 'style']},
                strip=True
            )
            return f'<div class="output output-html">{clean_html}</div>'
        
        elif 'text/plain' in data:
            # Plain text output
            text = data['text/plain']
            escaped_text = bleach.clean(text, strip=True)
            return f'<div class="output output-text"><pre>{escaped_text}</pre></div>'
    
    elif msg_type == 'display_data':
        # Display data (e.g., plots, images)
        data = content['data']
        
        if 'text/html' in data:
            html = data['text/html']
            clean_html = bleach.clean(
                html,
                tags=['div', 'span', 'p', 'br', 'b', 'i', 'strong', 'em', 'pre', 'code', 'img'],
                attributes={'*': ['class', 'style', 'src', 'alt', 'width', 'height']},
                strip=True
            )
            return f'<div class="output output-display">{clean_html}</div>'
        
        elif 'image/png' in data:
            # Base64 encoded image
            img_data = data['image/png']
            return f'<div class="output output-image"><img src="data:image/png;base64,{img_data}" /></div>'
        
        elif 'text/plain' in data:
            text = data['text/plain']
            escaped_text = bleach.clean(text, strip=True)
            return f'<div class="output output-text"><pre>{escaped_text}</pre></div>'
    
    elif msg_type == 'error':
        # Execution error
        ename = content['ename']
        evalue = content['evalue']
        traceback = content['traceback']
        
        # Clean and format traceback
        tb_text = '\n'.join(traceback)
        escaped_tb = bleach.clean(tb_text, strip=True)
        
        return f'''
        <div class="output output-error">
            <div class="error-name">{ename}</div>
            <div class="error-value">{evalue}</div>
            <pre class="error-traceback">{escaped_tb}</pre>
        </div>
        '''
    
    return ""