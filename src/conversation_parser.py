#!/usr/bin/env python3
"""
Conversation Parser for nbscribe
Extracts conversation context from HTML chat logs using <chat-msg> elements
"""

import logging
from typing import List, Dict, Optional
from pathlib import Path
from bs4 import BeautifulSoup
from datetime import datetime

logger = logging.getLogger(__name__)


class ConversationParser:
    """
    Parser for extracting conversation context from HTML chat logs.
    
    Handles <chat-msg> elements with role and timestamp attributes.
    """
    
    def __init__(self):
        """Initialize the conversation parser."""
        pass
    
    def parse_html_log(self, html_content: str) -> List[Dict[str, str]]:
        """
        Parse HTML conversation log and extract messages.
        
        Args:
            html_content: Raw HTML content containing <chat-msg> elements
            
        Returns:
            List of message dictionaries with role, content, and timestamp
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            messages = []
            
            # Find all chat-msg elements
            chat_msgs = soup.find_all('chat-msg')
            
            for msg_elem in chat_msgs:
                role = msg_elem.get('role', 'unknown')
                timestamp = msg_elem.get('timestamp', '')
                content = msg_elem.get_text().strip()
                
                if content:  # Only include non-empty messages
                    messages.append({
                        'role': role,
                        'content': content,
                        'timestamp': timestamp
                    })
            
            logger.info(f"Parsed {len(messages)} messages from HTML log")
            return messages
            
        except Exception as e:
            logger.error(f"Failed to parse HTML log: {e}")
            return []
    
    def extract_conversation_context(self, html_file: Path, max_messages: Optional[int] = None) -> str:
        """
        Extract conversation context for LLM from HTML file.
        
        Args:
            html_file: Path to HTML conversation log file
            max_messages: Optional limit on number of messages to include
            
        Returns:
            Formatted conversation context string for LLM
        """
        try:
            if not html_file.exists():
                logger.warning(f"HTML log file not found: {html_file}")
                return ""
            
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            messages = self.parse_html_log(html_content)
            
            # Limit messages if specified
            if max_messages and len(messages) > max_messages:
                messages = messages[-max_messages:]  # Keep most recent messages
            
            # Format for LLM context
            context_lines = []
            for msg in messages:
                role = msg['role'].title()
                content = msg['content']
                context_lines.append(f"{role}: {content}")
            
            context = "\n\n".join(context_lines)
            logger.info(f"Extracted context with {len(messages)} messages")
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to extract conversation context: {e}")
            return ""
    
    def validate_html_log(self, html_content: str) -> bool:
        """
        Validate that HTML content contains proper chat-msg structure.
        
        Args:
            html_content: Raw HTML content to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Check for basic structure
            if not soup.find('html'):
                return False
            
            # Check for chat-msg elements with required attributes
            chat_msgs = soup.find_all('chat-msg')
            
            for msg in chat_msgs:
                if not msg.get('role') or not msg.get('timestamp'):
                    logger.warning(f"Invalid chat-msg element: missing role or timestamp")
                    return False
                
                # Validate role values
                valid_roles = {'user', 'assistant', 'system'}
                if msg.get('role') not in valid_roles:
                    logger.warning(f"Invalid role: {msg.get('role')}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"HTML validation failed: {e}")
            return False
    
    def get_message_count(self, html_file: Path) -> int:
        """
        Get the number of messages in an HTML conversation log.
        
        Args:
            html_file: Path to HTML conversation log file
            
        Returns:
            Number of messages in the log
        """
        try:
            if not html_file.exists():
                return 0
            
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            messages = self.parse_html_log(html_content)
            return len(messages)
            
        except Exception as e:
            logger.error(f"Failed to count messages: {e}")
            return 0


def parse_conversation_context(html_file: Path, max_messages: Optional[int] = None) -> str:
    """
    Convenience function to extract conversation context from HTML file.
    
    Args:
        html_file: Path to HTML conversation log file
        max_messages: Optional limit on number of messages
        
    Returns:
        Formatted conversation context string
    """
    parser = ConversationParser()
    return parser.extract_conversation_context(html_file, max_messages) 