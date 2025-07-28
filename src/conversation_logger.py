#!/usr/bin/env python3
"""
Conversation Logger for nbscribe
Handles saving and loading HTML conversation logs with <chat-msg> elements
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)


class ConversationLogger:
    """
    Manages HTML conversation log files.
    
    Handles saving DOM state to HTML files and loading existing conversations.
    """
    
    def __init__(self, logs_dir: str = "logs/conversations"):
        """
        Initialize the conversation logger.
        
        Args:
            logs_dir: Directory to store conversation logs
        """
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up Jinja2 environment for templates
        self.jinja_env = Environment(loader=FileSystemLoader('templates'))
    
    def generate_session_filename(self, session_id: str) -> str:
        """
        Generate a filename for a conversation session.
        
        Args:
            session_id: Unique session identifier (already includes timestamp)
            
        Returns:
            Filename for the conversation log
        """
        return f"{session_id}.html"
    
    def create_conversation_log(self, session_id: str, messages: list = None) -> Path:
        """
        Create a new HTML conversation log file.
        
        Args:
            session_id: Unique session identifier
            messages: Optional list of initial messages
            
        Returns:
            Path to the created log file
        """
        filename = self.generate_session_filename(session_id)
        log_file = self.logs_dir / filename
        
        # Create initial HTML structure
        html_content = self._generate_html_template(session_id, messages or [])
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Created conversation log: {log_file}")
        return log_file
    
    def save_conversation_state(self, log_file: Path, messages: list) -> bool:
        """
        Save current conversation state to HTML file.
        
        Args:
            log_file: Path to the conversation log file
            messages: List of message dictionaries with role, content, timestamp
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract session info from existing file if it exists
            session_id = "default"
            created_at = datetime.now().isoformat()
            
            if log_file.exists():
                # Try to preserve session metadata
                from src.conversation_parser import ConversationParser
                parser = ConversationParser()
                
                with open(log_file, 'r', encoding='utf-8') as f:
                    existing_content = f.read()
                
                # Extract session ID from meta tag if present
                import re
                session_match = re.search(r'<meta name="conversation-id" content="([^"]+)"', existing_content)
                if session_match:
                    session_id = session_match.group(1)
                
                created_match = re.search(r'<meta name="created" content="([^"]+)"', existing_content)
                if created_match:
                    created_at = created_match.group(1)
            
            # Generate new HTML with current messages
            html_content = self._generate_html_template(session_id, messages, created_at)
            
            # Write to file
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Saved conversation state to {log_file} ({len(messages)} messages)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save conversation state: {e}")
            return False
    
    def load_conversation_log(self, log_file: Path) -> Optional[Dict[str, Any]]:
        """
        Load an existing conversation log.
        
        Args:
            log_file: Path to the conversation log file
            
        Returns:
            Dictionary with conversation data or None if failed
        """
        try:
            if not log_file.exists():
                logger.warning(f"Conversation log not found: {log_file}")
                return None
            
            with open(log_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Parse the HTML to extract conversation data
            from src.conversation_parser import ConversationParser
            parser = ConversationParser()
            
            messages = parser.parse_html_log(html_content)
            
            # Extract metadata
            import re
            session_id_match = re.search(r'<meta name="conversation-id" content="([^"]+)"', html_content)
            session_id = session_id_match.group(1) if session_id_match else "unknown"
            
            created_match = re.search(r'<meta name="created" content="([^"]+)"', html_content)
            created_at = created_match.group(1) if created_match else ""
            
            logger.info(f"Loaded conversation log: {log_file} ({len(messages)} messages)")
            
            return {
                'session_id': session_id,
                'created_at': created_at,
                'messages': messages,
                'log_file': log_file,
                'html_content': html_content
            }
            
        except Exception as e:
            logger.error(f"Failed to load conversation log: {e}")
            return None
    
    def list_conversation_logs(self) -> list:
        """
        List all conversation log files.
        
        Returns:
            List of conversation log file paths
        """
        try:
            log_files = list(self.logs_dir.glob("*.html"))
            log_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)  # Most recent first
            return log_files
        except Exception as e:
            logger.error(f"Failed to list conversation logs: {e}")
            return []
    
    def _generate_html_template(self, session_id: str, messages: list, created_at: str = None) -> str:
        """
        Generate HTML content for a conversation log using template file.
        
        Args:
            session_id: Session identifier
            messages: List of message dictionaries
            created_at: Creation timestamp (ISO format)
            
        Returns:
            Complete HTML document as string
        """
        if created_at is None:
            created_at = datetime.now().isoformat()
        
        try:
            template = self.jinja_env.get_template('conversation_log.html')
            return template.render(
                session_id=session_id,
                created_at=created_at,
                version="0.1.0",
                messages=messages
            )
        except Exception as e:
            logger.error(f"Failed to render conversation template: {e}")
            # Fallback to minimal template
            return f"""<!DOCTYPE html>
<html><head><title>Conversation Log</title></head>
<body><h1>Error rendering conversation</h1><p>{e}</p></body>
</html>"""


def create_conversation_log(session_id: str, logs_dir: str = "logs/conversations") -> Path:
    """
    Convenience function to create a new conversation log.
    
    Args:
        session_id: Unique session identifier
        logs_dir: Directory to store logs
        
    Returns:
        Path to the created log file
    """
    logger = ConversationLogger(logs_dir)
    return logger.create_conversation_log(session_id) 