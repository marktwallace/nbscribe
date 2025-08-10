"""
Jupyter Notebook 7 Server Management

Manages a subprocess running Jupyter Notebook 7 with:
- Fixed token authentication
- Auto-discovery of free ports
- Lifecycle management (start/stop)
- Health checking
"""

import subprocess
import socket
import time
import logging
import signal
import os
import threading
import queue
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class NotebookServerManager:
    """Manages a Jupyter Notebook 7 subprocess for nbscribe integration"""
    
    def __init__(self, token: str = "nbscribe-token", start_port: int = 8889):
        self.token = token
        self.start_port = start_port
        self.port: Optional[int] = None
        self.process: Optional[subprocess.Popen] = None
        self.base_url = "/jupyter"
        # Path to our app-specific Jupyter config directory (contains labconfig/page_config.json)
        self.app_jupyter_config_dir = str((Path(__file__).resolve().parent.parent / "jupyter_config").resolve())
        
        # Log capture
        self.stdout_lines = queue.Queue(maxsize=1000)
        self.stderr_lines = queue.Queue(maxsize=1000)
        self._log_threads = []
        
    def _log_reader(self, pipe, log_queue, prefix):
        """Read from subprocess pipe and store in queue"""
        try:
            # Suppress noisy kernel-missing tracebacks while still surfacing the concise warning
            suppress_traceback = False
            kernel_missing_context = False

            for line in iter(pipe.readline, ''):
                if not line:
                    continue

                raw = line.strip()

                # Detect start of a kernel-missing warning
                if (
                    'Kernel does not exist:' in raw
                    or ('/jupyter/api/kernels/' in raw and ' 404 ' in raw)
                    or (' 404 GET ' in raw and '/jupyter/api/kernels/' in raw)
                ):
                    kernel_missing_context = True

                # If we're currently suppressing traceback lines, stop suppression when a new log record begins
                if suppress_traceback:
                    if raw.startswith('['):
                        suppress_traceback = False
                        kernel_missing_context = False
                    else:
                        # Skip stack-frame lines for expected kernel-missing 404s
                        continue

                # If a traceback follows a known kernel-missing warning, suppress the traceback body
                if kernel_missing_context and raw.startswith('Traceback (most recent call last):'):
                    suppress_traceback = True
                    # Do not print the traceback header; keep logs concise
                    continue

                # Log to console as well with immediate flush
                print(f"[Jupyter {prefix}] {raw}", flush=True)
                logger.info(f"[Jupyter {prefix}] {raw}")

                # Store in queue (remove oldest if full)
                try:
                    log_queue.put_nowait(raw)
                except queue.Full:
                    # Remove oldest entry and add new one
                    try:
                        log_queue.get_nowait()
                        log_queue.put_nowait(raw)
                    except queue.Empty:
                        pass
        except Exception as e:
            logger.error(f"Error reading {prefix} logs: {e}")
        finally:
            pipe.close()
            
    def get_recent_logs(self, lines: int = 50) -> dict:
        """Get recent stdout and stderr logs"""
        stdout_logs = []
        stderr_logs = []
        
        # Get stdout logs
        temp_queue = queue.Queue()
        while not self.stdout_lines.empty():
            try:
                line = self.stdout_lines.get_nowait()
                stdout_logs.append(line)
                temp_queue.put(line)
            except queue.Empty:
                break
        
        # Put them back
        while not temp_queue.empty():
            try:
                self.stdout_lines.put_nowait(temp_queue.get_nowait())
            except queue.Full:
                break
                
        # Get stderr logs  
        temp_queue = queue.Queue()
        while not self.stderr_lines.empty():
            try:
                line = self.stderr_lines.get_nowait()
                stderr_logs.append(line)
                temp_queue.put(line)
            except queue.Empty:
                break
                
        # Put them back
        while not temp_queue.empty():
            try:
                self.stderr_lines.put_nowait(temp_queue.get_nowait())
            except queue.Full:
                break
        
        return {
            "stdout": stdout_logs[-lines:] if stdout_logs else [],
            "stderr": stderr_logs[-lines:] if stderr_logs else []
        }

    def find_free_port(self, start_port: int) -> int:
        """Find a free port starting from start_port"""
        port = start_port
        while port < start_port + 100:  # Try 100 ports max
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                port += 1
        raise RuntimeError(f"No free ports found starting from {start_port}")
    
    def start(self, notebook_dir: Optional[Path] = None) -> None:
        """Start Jupyter Notebook 7 server"""
        if self.process and self.process.poll() is None:
            logger.warning("Notebook server already running")
            return
            
        # Find a free port
        self.port = self.find_free_port(self.start_port)
        
        # Set notebook directory to current working directory if not specified
        if notebook_dir is None:
            notebook_dir = Path.cwd()
        
        # Build command - minimal configuration for AI-assisted editing
        cmd = [
            "jupyter", "notebook",
            f"--ServerApp.token={self.token}",
            "--ServerApp.password=''",
            f"--ServerApp.base_url={self.base_url}",
            "--no-browser",
            f"--port={self.port}",
            f"--notebook-dir={notebook_dir}",
            "--ServerApp.allow_origin='*'",  # Allow CORS for iframe
            "--ServerApp.disable_check_xsrf=True",  # Disable XSRF for API calls
            "--ServerApp.tornado_settings={'headers': {'X-Frame-Options': 'ALLOWALL'}}",  # Allow iframe embedding
            
            # Optimal minimal configuration - eliminates noise while preserving functionality
            "--ServerApp.terminals_enabled=False",  # Eliminates /api/terminals polling + WebSocket errors
        ]
        
        logger.info(f"Starting Jupyter Notebook on port {self.port}")
        logger.debug(f"Command: {' '.join(cmd)}")
        logger.info(f"Using JUPYTER_CONFIG_DIR={self.app_jupyter_config_dir}")
        
        # Start subprocess with output capture
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # Ensure our JupyterLab page configuration is applied (disables file browser, etc.)
            env=dict(os.environ, JUPYTER_CONFIG_DIR=self.app_jupyter_config_dir)
        )
        
        # Start log reading threads
        stdout_thread = threading.Thread(
            target=self._log_reader, 
            args=(self.process.stdout, self.stdout_lines, "stdout"),
            daemon=True
        )
        stderr_thread = threading.Thread(
            target=self._log_reader, 
            args=(self.process.stderr, self.stderr_lines, "stderr"),
            daemon=True
        )
        
        stdout_thread.start()
        stderr_thread.start()
        self._log_threads = [stdout_thread, stderr_thread]
        
        # Wait for server to start
        self._wait_for_startup()
        
    def _wait_for_startup(self, timeout: int = 30) -> None:
        """Wait for Jupyter server to respond to requests"""
        import httpx
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.process and self.process.poll() is not None:
                # Process died - show logs
                logs = self.get_recent_logs()
                error_msg = f"Jupyter server failed to start:\n"
                if logs["stdout"]:
                    error_msg += f"STDOUT:\n" + "\n".join(logs["stdout"]) + "\n"
                if logs["stderr"]:
                    error_msg += f"STDERR:\n" + "\n".join(logs["stderr"]) + "\n"
                raise RuntimeError(error_msg)
                
            try:
                # Try to connect to the server
                with httpx.Client() as client:
                    response = client.get(
                        f"http://localhost:{self.port}{self.base_url}/api/status",
                        params={"token": self.token},
                        timeout=2.0
                    )
                    if response.status_code == 200:
                        logger.info(f"Jupyter server ready on port {self.port}")
                        return
            except (httpx.RequestError, httpx.TimeoutException):
                pass
                
            time.sleep(0.5)
            
        raise RuntimeError(f"Jupyter server did not start within {timeout} seconds")
    
    def stop(self) -> None:
        """Stop the Jupyter Notebook server"""
        if not self.process:
            return
            
        logger.info("Stopping Jupyter Notebook server")
        
        # Send SIGTERM first
        self.process.terminate()
        
        try:
            # Wait for graceful shutdown
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # Force kill if it doesn't stop gracefully
            logger.warning("Force killing Jupyter server")
            self.process.kill()
            self.process.wait()
            
        self.process = None
        self.port = None
        
    def is_running(self) -> bool:
        """Check if the notebook server is running"""
        return self.process is not None and self.process.poll() is None
    
    def get_url(self, path: str = "") -> str:
        """Get the full URL for a given path"""
        if not self.port:
            raise RuntimeError("Server not started")
        return f"http://localhost:{self.port}{self.base_url}{path}"
    
    def get_token_url(self, path: str = "") -> str:
        """Get URL with token authentication"""
        base_url = self.get_url(path)
        return f"{base_url}?token={self.token}"
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop() 