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
        
        # Build command
        cmd = [
            "jupyter", "notebook",
            f"--NotebookApp.token={self.token}",
            "--NotebookApp.password=''",
            f"--NotebookApp.base_url={self.base_url}",
            "--no-browser",
            f"--port={self.port}",
            f"--notebook-dir={notebook_dir}",
            "--NotebookApp.allow_origin='*'",  # Allow CORS for iframe
            "--NotebookApp.disable_check_xsrf=True",  # Disable XSRF for API calls
            "--NotebookApp.tornado_settings={'headers': {'X-Frame-Options': 'ALLOWALL'}}"  # Allow iframe embedding
        ]
        
        logger.info(f"Starting Jupyter Notebook on port {self.port}")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        # Start subprocess with output capture
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=dict(os.environ, JUPYTER_CONFIG_DIR=str(Path.home() / ".jupyter"))
        )
        
        # Wait for server to start
        self._wait_for_startup()
        
    def _wait_for_startup(self, timeout: int = 30) -> None:
        """Wait for Jupyter server to respond to requests"""
        import httpx
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.process and self.process.poll() is not None:
                # Process died
                stdout, stderr = self.process.communicate()
                raise RuntimeError(f"Jupyter server failed to start:\nSTDOUT: {stdout}\nSTDERR: {stderr}")
                
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