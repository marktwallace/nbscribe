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
        
        # Project-scoped Jupyter configuration (do not use ~/.jupyter)
        self.project_config_dir = Path("config/jupyter")
        self.project_lab_settings_dir = self.project_config_dir / "lab" / "user-settings"
        self.project_lab_workspaces_dir = self.project_config_dir / "lab" / "workspaces"
        
        # Log capture
        self.stdout_lines = queue.Queue(maxsize=1000)
        self.stderr_lines = queue.Queue(maxsize=1000)
        self._log_threads = []
        
    def _log_reader(self, pipe, log_queue, prefix):
        """Read from subprocess pipe and store in queue"""
        try:
            for line in iter(pipe.readline, ''):
                if line:
                    line = line.strip()
                    # Log to console as well with immediate flush
                    print(f"[Jupyter {prefix}] {line}", flush=True)
                    logger.info(f"[Jupyter {prefix}] {line}")
                    # Store in queue (remove oldest if full)
                    try:
                        log_queue.put_nowait(line)
                    except queue.Full:
                        # Remove oldest entry and add new one
                        try:
                            log_queue.get_nowait()
                            log_queue.put_nowait(line)
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
        
        # Ensure project-local JupyterLab single-document mode is configured
        self._ensure_lab_single_document_mode()
        
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
        
        # Start subprocess with output capture
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self._build_jupyter_env()
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
    
    def _ensure_lab_single_document_mode(self) -> None:
        """Create project-scoped Lab settings that force single-document mode."""
        try:
            # Ensure directories exist
            settings_pkg_dir = self.project_lab_settings_dir / "@jupyterlab" / "application-extension"
            settings_pkg_dir.mkdir(parents=True, exist_ok=True)
            self.project_lab_workspaces_dir.mkdir(parents=True, exist_ok=True)

            settings_file = settings_pkg_dir / "application.jupyterlab-settings"
            desired = {
                "mode": "single-document"
            }
            # Write only if missing or different
            write_file = True
            if settings_file.exists():
                try:
                    import json
                    current = json.loads(settings_file.read_text(encoding="utf-8") or "{}")
                    if current == desired:
                        write_file = False
                except Exception:
                    write_file = True
            if write_file:
                import json
                settings_file.write_text(json.dumps(desired, indent=2), encoding="utf-8")
                logger.info(f"Wrote Lab single-document settings: {settings_file}")
        except Exception as e:
            logger.error(f"Failed to prepare Lab settings: {e}")
        
        # Seed a dedicated workspace with single-document mode and sidebars collapsed
        try:
            import json
            workspace_id = "nbscribe-embed"
            workspace_file = self.project_lab_workspaces_dir / f"{workspace_id}.jupyterlab-workspace"
            desired_workspace = {
                "data": {
                    "layout-restorer:data": {
                        "main": {
                            "dock": {
                                "mode": "single-document",
                                "main": {"current": None, "widgets": []}
                            },
                            "left": {"collapsed": True, "widgets": []},
                            "right": {"collapsed": True, "widgets": []}
                        }
                    }
                },
                "metadata": {"id": workspace_id},
                "schemaVersion": 1
            }
            write_ws = True
            if workspace_file.exists():
                try:
                    current_ws = json.loads(workspace_file.read_text(encoding="utf-8") or "{}")
                    if current_ws == desired_workspace:
                        write_ws = False
                except Exception:
                    write_ws = True
            if write_ws:
                workspace_file.write_text(json.dumps(desired_workspace, indent=2), encoding="utf-8")
                logger.info(f"Seeded Lab workspace: {workspace_file}")
        except Exception as e:
            logger.error(f"Failed to seed Lab workspace: {e}")
    
    def _build_jupyter_env(self) -> dict:
        """Build environment for Jupyter process, isolating from user ~/.jupyter."""
        env = dict(os.environ)
        # Use project-local config to avoid picking up user settings
        env["JUPYTER_CONFIG_DIR"] = str(self.project_config_dir)
        # Force Lab to use our project-scoped user settings (single-document mode)
        env["JUPYTERLAB_SETTINGS_DIR"] = str(self.project_lab_settings_dir)
        # Keep workspaces local as well to avoid persisting user-side layout
        env["JUPYTERLAB_WORKSPACES_DIR"] = str(self.project_lab_workspaces_dir)
        return env
    
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