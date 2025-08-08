#!/usr/bin/env python3
"""
LLM Interface for nbscribe
Uses LangChain for OpenAI connection only - no hidden state or memory
"""

import os
import logging
import traceback
from pathlib import Path
from typing import Optional, List, Dict

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMInterface:
    """
    Simple LLM interface using LangChain for OpenAI connection.
    No hidden state - all context must be provided explicitly.
    """
    
    def __init__(self, model_name: str = "gpt-4-turbo-preview"):
        """
        Initialize LLM interface.
        
        Args:
            model_name: OpenAI model to use (default: gpt-4-turbo-preview)
        """
        self.model_name = model_name
        self.client = None
        self.system_prompt = None
        
        # Load system prompt
        self._load_system_prompt()
        
        # Initialize OpenAI client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the OpenAI client via LangChain."""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not found in environment variables. "
                    "Please set it or create a .env file."
                )
            
            self.client = ChatOpenAI(
                model_name=self.model_name,
                openai_api_key=api_key,
                temperature=0.7,
                max_tokens=1000,
                timeout=30
            )
            
            logger.info(f"Initialized LLM client with model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _load_system_prompt(self, prompt_file: str = "system_prompt.txt"):
        """
        Load system prompt from prompts directory.
        
        Args:
            prompt_file: Name of prompt file to load
        """
        try:
            prompt_path = Path("prompts") / prompt_file
            
            if not prompt_path.exists():
                logger.warning(f"Prompt file not found: {prompt_path}")
                self.system_prompt = "You are a helpful AI assistant for Jupyter notebooks."
                return
            
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read().strip()
            
            logger.info(f"Loaded system prompt from: {prompt_path}")
            
        except Exception as e:
            logger.error(f"Failed to load system prompt: {e}")
            logger.error(traceback.format_exc())
            # Use fallback prompt
            self.system_prompt = "You are a helpful AI assistant for Jupyter notebooks."
    
    def generate_response(self, messages: List[Dict[str, str]], user_message: Optional[str] = None) -> str:
        """
        Generate a response using the LLM with linear conversation history.
        No hidden state - all context must be provided explicitly via messages.
        
        Args:
            messages: Complete linear conversation history (role, content pairs)
            user_message: Optional - if provided, added as final user message
            
        Returns:
            Generated response text
        
        Raises:
            Exception: If the LLM call fails (fail early principle)
        """
        try:
            if not self.client:
                raise RuntimeError("LLM client not initialized")
            
            # Add final user message if provided
            conversation_messages = messages.copy()
            if user_message:
                conversation_messages.append({'role': 'user', 'content': user_message})
            
            logger.info(f"Generating response with {len(conversation_messages)} messages in conversation")
            
            # Convert to LangChain message objects
            langchain_messages = self._convert_to_langchain_messages(conversation_messages)
            
            logger.info(f"Sending request to {self.model_name}")
            logger.debug(f"Total messages: {len(langchain_messages)}")
            
            # Make the API call
            response = self.client(langchain_messages)
            
            response_text = response.content.strip()
            logger.info(f"Received response ({len(response_text)} chars)")
            logger.debug(f"Response: {response_text}")
            
            return response_text
            
        except Exception as e:
            # Fail early and often - complete error logging with stack traces
            logger.error(f"LLM generation failed: {e}")
            logger.error(f"Messages count: {len(messages)}")
            logger.error(f"Model: {self.model_name}")
            logger.error(traceback.format_exc())
            raise

    def generate_response_stream(self, messages: List[Dict[str, str]], user_message: Optional[str] = None):
        """
        Generate a streaming response using the LLM with linear conversation history.
        Yields chunks as they arrive from the LLM.
        
        Args:
            messages: Complete linear conversation history (role, content pairs)
            user_message: Optional - if provided, added as final user message
            
        Yields:
            String chunks of the response as they arrive
            
        Raises:
            Exception: If the LLM call fails (fail early principle)
        """
        try:
            if not self.client:
                raise RuntimeError("LLM client not initialized")
            
            # Add final user message if provided
            conversation_messages = messages.copy()
            if user_message:
                conversation_messages.append({'role': 'user', 'content': user_message})
            
            logger.info(f"Generating streaming response with {len(conversation_messages)} messages in conversation")
            
            # Convert to LangChain message objects
            langchain_messages = self._convert_to_langchain_messages(conversation_messages)
            
            logger.info(f"Sending streaming request to {self.model_name}")
            logger.debug(f"Total messages: {len(langchain_messages)}")
            
            # Make the streaming API call
            for chunk in self.client.stream(langchain_messages):
                if chunk.content:
                    yield chunk.content
            
            logger.info(f"Completed streaming response")
            
        except Exception as e:
            # Fail early and often - complete error logging with stack traces
            logger.error(f"LLM streaming failed: {e}")
            logger.error(f"Messages count: {len(messages)}")
            logger.error(f"Model: {self.model_name}")
            logger.error(traceback.format_exc())
            raise

    def _convert_to_langchain_messages(self, messages: List[Dict[str, str]]):
        """
        Convert linear conversation messages to LangChain message objects.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            
        Returns:
            List of LangChain message objects
        """
        langchain_messages = []
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                langchain_messages.append(SystemMessage(content=content))
            elif role == 'assistant':
                langchain_messages.append(AIMessage(content=content))
            else:  # 'user' or any other role
                langchain_messages.append(HumanMessage(content=content))
        
        return langchain_messages
    
    def reload_prompt(self, prompt_file: str = "system_prompt.txt"):
        """
        Reload system prompt from file.
        Useful for development and prompt iteration.
        
        Args:
            prompt_file: Name of prompt file to reload
        """
        logger.info(f"Reloading prompt: {prompt_file}")
        self._load_system_prompt(prompt_file)


# Global instance - can be reused across requests
_llm_interface = None


def get_llm_interface() -> LLMInterface:
    """
    Get the global LLM interface instance.
    Creates it if it doesn't exist.
    """
    global _llm_interface
    
    if _llm_interface is None:
        # Allow model override via environment variable
        model_name = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        _llm_interface = LLMInterface(model_name=model_name)
    
    return _llm_interface


def generate_response(user_message: str, conversation_context: Optional[str] = None, notebook_context: Optional[str] = None) -> str:
    """
    LEGACY: Convenience function for generating responses with old system injection approach.
    
    This is maintained for backward compatibility but will be replaced by linear conversation architecture.
    
    Args:
        user_message: The user's message
        conversation_context: Optional conversation history
        notebook_context: Optional current notebook structure
        
    Returns:
        The LLM's response
    """
    llm = get_llm_interface()
    
    # Build legacy-style messages with system injection for backward compatibility
    messages = []
    
    # Add system prompt
    if llm.system_prompt:
        messages.append({'role': 'system', 'content': llm.system_prompt})
    
    # Add notebook context as system message if provided
    if notebook_context:
        notebook_message = f"Current notebook context:\n{notebook_context}\n\n"
        messages.append({'role': 'system', 'content': notebook_message})
    
    # Add conversation context as system message if provided
    if conversation_context:
        context_message = f"Previous conversation context:\n{conversation_context}\n\n"
        messages.append({'role': 'system', 'content': context_message})
    
    # Use new interface with legacy message structure
    return llm.generate_response(messages, user_message) 