"""
Conversation Manager for Linear JSON-First Architecture

This module handles conversation state, notebook formatting, and message management
for the linear conversation architecture where the AI sees complete notebook JSON
as User messages rather than system message injection.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages linear conversation flow with JSON-first notebook representation.
    
    Key responsibilities:
    1. Format complete notebooks as JSON for initial User messages
    2. Format cell updates as JSON fragments for post-modification User messages
    3. Track conversation length for restart triggers
    4. Build linear message history for LLM calls
    """
    
    def __init__(self, max_tokens: int = 100000):
        """
        Initialize conversation manager.
        
        Args:
            max_tokens: Maximum conversation length before restart trigger
        """
        self.max_tokens = max_tokens
        
    def format_notebook_for_ai(self, notebook_content: dict) -> str:
        """
        Convert complete .ipynb JSON to formatted User message.
        
        This creates the initial User message that gives the AI complete context
        of the notebook structure, including all cells, outputs, metadata, etc.
        
        Args:
            notebook_content: Complete .ipynb notebook content
            
        Returns:
            Formatted string for User message containing complete notebook JSON
        """
        try:
            # Pretty-print the notebook JSON for readability
            formatted_json = json.dumps(notebook_content, indent=2, ensure_ascii=False)
            
            # Create the User message with clear labeling
            message = f"Complete notebook JSON:\n{formatted_json}"
            
            cell_count = len(notebook_content.get('cells', []))
            logger.info(f"Formatted notebook with {cell_count} cells for AI context")
            return message
            
        except Exception as e:
            logger.error(f"Error formatting notebook for AI: {e}")
            return "Error: Could not format notebook JSON"
    
    def format_cell_update(self, cell_data: dict, operation: str) -> str:
        """
        Format cell update as complete JSON cell object for User message.
        
        This creates User messages that inform the AI about cell changes after
        human approval of tool directives.
        
        Args:
            cell_data: Complete cell JSON object from notebook
            operation: Type of operation ("inserted", "updated", "failed")
            
        Returns:
            Formatted string for User message containing cell JSON
        """
        try:
            # Pretty-print the cell JSON
            formatted_cell = json.dumps(cell_data, indent=2, ensure_ascii=False)
            
            # Create the User message with operation context
            message = f"Cell {operation}:\n{formatted_cell}"
            
            cell_id = cell_data.get('id', 'unknown')
            cell_type = cell_data.get('cell_type', 'unknown')
            logger.info(f"Formatted cell {operation}: {cell_id} ({cell_type})")
            return message
            
        except Exception as e:
            logger.error(f"Error formatting cell update: {e}")
            return f"Error: Could not format cell {operation}"
    
    def format_cell_deletion(self, cell_id: str) -> str:
        """
        Format cell deletion notification for User message.
        
        This creates a clear User message that informs the AI a cell was deleted,
        which is critical for maintaining referential integrity.
        
        Args:
            cell_id: ID of the deleted cell
            
        Returns:
            Formatted string for User message about cell deletion
        """
        message = f"Cell deleted: {cell_id}"
        logger.info(f"Formatted cell deletion: {cell_id}")
        return message
    
    def format_operation_failure(self, operation: str, error_message: str, cell_id: Optional[str] = None) -> str:
        """
        Format operation failure notification for User message.
        
        This ensures the AI knows when operations failed and why, maintaining
        transparency in the conversation history.
        
        Args:
            operation: Type of operation that failed ("insert", "edit", "delete")
            error_message: Description of what went wrong
            cell_id: ID of cell involved (if applicable)
            
        Returns:
            Formatted string for User message about operation failure
        """
        if cell_id:
            message = f"Operation failed - {operation} cell {cell_id}: {error_message}"
        else:
            message = f"Operation failed - {operation}: {error_message}"
            
        logger.info(f"Formatted operation failure: {operation} - {error_message}")
        return message
    
    def build_linear_conversation(self, messages: List[Dict[str, Any]], system_prompt: str) -> List[Dict[str, str]]:
        """
        Build linear conversation history for LLM calls.
        
        This replaces the system message injection approach with a linear conversation
        where notebook JSON appears as User messages and all state updates are explicit.
        
        Args:
            messages: List of conversation messages from session data
            system_prompt: System prompt content
            
        Returns:
            List of messages formatted for LLM API (role, content)
        """
        try:
            linear_messages = []
            
            # Always start with system prompt
            linear_messages.append({
                'role': 'system',
                'content': system_prompt
            })
            
            # Add all conversation messages in linear order
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                # Map our role names to LLM API format
                if role == 'assistant':
                    api_role = 'assistant'
                else:  # user, system, or any other role maps to user
                    api_role = 'user'
                
                linear_messages.append({
                    'role': api_role,
                    'content': content
                })
            
            logger.info(f"Built linear conversation with {len(linear_messages)} messages")
            return linear_messages
            
        except Exception as e:
            logger.error(f"Error building linear conversation: {e}")
            # Fallback to just system prompt and last user message
            last_user_msg = next((msg for msg in reversed(messages) if msg.get('role') == 'user'), None)
            if last_user_msg:
                return [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': last_user_msg.get('content', '')}
                ]
            else:
                return [{'role': 'system', 'content': system_prompt}]
    
    def should_restart_conversation(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Determine if conversation should be restarted based on length.
        
        This is a simple token estimation based on message count and content length.
        A more sophisticated implementation would use actual token counting.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            True if conversation should be restarted
        """
        try:
            # Simple heuristic: estimate ~1000 characters per ~250 tokens
            total_chars = sum(len(msg.get('content', '')) for msg in messages)
            estimated_tokens = total_chars // 4  # Rough estimation
            
            should_restart = estimated_tokens > self.max_tokens
            
            if should_restart:
                logger.info(f"Conversation restart recommended - estimated {estimated_tokens} tokens > {self.max_tokens}")
            else:
                logger.debug(f"Conversation length OK - estimated {estimated_tokens} tokens")
                
            return should_restart
            
        except Exception as e:
            logger.error(f"Error checking conversation length: {e}")
            # Conservative fallback - restart if we have a lot of messages
            return len(messages) > 50
    
    def create_conversation_summary(self, messages: List[Dict[str, Any]]) -> str:
        """
        Create a summary of the conversation for restart scenarios.
        
        This will be used as part of the system prompt when restarting conversations
        to maintain context while reducing token usage.
        
        Args:
            messages: List of conversation messages to summarize
            
        Returns:
            Summary text for system prompt
        """
        try:
            # For now, create a simple summary
            # TODO: In the future, use LLM to create intelligent summaries
            
            user_messages = [msg for msg in messages if msg.get('role') == 'user']
            assistant_messages = [msg for msg in messages if msg.get('role') == 'assistant']
            
            summary_parts = [
                f"Previous conversation summary:",
                f"- Total messages: {len(messages)}",
                f"- User requests: {len(user_messages)}",
                f"- Assistant responses: {len(assistant_messages)}"
            ]
            
            # Add key topics (simple keyword extraction)
            all_content = ' '.join(msg.get('content', '') for msg in messages)
            if 'pandas' in all_content.lower():
                summary_parts.append("- Discussed pandas data analysis")
            if 'matplotlib' in all_content.lower() or 'plot' in all_content.lower():
                summary_parts.append("- Worked on data visualization")
            if 'insert_cell' in all_content:
                summary_parts.append("- Created new notebook cells")
            if 'edit_cell' in all_content:
                summary_parts.append("- Modified existing cells")
            if 'delete_cell' in all_content:
                summary_parts.append("- Deleted notebook cells")
            
            summary = '\n'.join(summary_parts)
            logger.info(f"Created conversation summary: {len(summary)} characters")
            return summary
            
        except Exception as e:
            logger.error(f"Error creating conversation summary: {e}")
            return "Previous conversation summary: (summary generation failed)"


def get_conversation_manager() -> ConversationManager:
    """
    Get a shared ConversationManager instance.
    
    Returns:
        ConversationManager instance
    """
    # For now, return a new instance each time
    # In the future, we might want to cache this
    return ConversationManager()