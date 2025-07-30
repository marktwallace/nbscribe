#!/usr/bin/env python3
"""
LLM Interface for nbscribe
Uses LangChain for OpenAI connection only - no hidden state or memory
"""

import os
import logging
import traceback
from pathlib import Path
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
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
    
    def generate_response(self, user_message: str, conversation_context: Optional[str] = None, notebook_context: Optional[str] = None) -> str:
        """
        Generate a response using the LLM.
        No hidden state - all context must be provided explicitly.
        
        Args:
            user_message: The user's message
            conversation_context: Optional conversation history (from HTML log)
            notebook_context: Optional current notebook structure
            
        Returns:
            The LLM's response as a string
            
        Raises:
            Exception: If the LLM call fails (fail early principle)
        """
        try:
            if not self.client:
                raise RuntimeError("LLM client not initialized")
            
            # Build messages - no hidden state, everything explicit
            messages = self._build_messages(user_message, conversation_context, notebook_context)
            
            logger.info(f"Sending request to {self.model_name}")
            logger.debug(f"User message: {user_message}")
            
            # Make the API call
            response = self.client(messages)
            
            response_text = response.content.strip()
            logger.info(f"Received response ({len(response_text)} chars)")
            logger.debug(f"Response: {response_text}")
            
            return response_text
            
        except Exception as e:
            # Fail early and often - complete error logging with stack traces
            logger.error(f"LLM generation failed: {e}")
            logger.error(f"User message: {user_message}")
            logger.error(f"Model: {self.model_name}")
            logger.error(traceback.format_exc())
            raise

    def generate_response_stream(self, user_message: str, conversation_context: Optional[str] = None, notebook_context: Optional[str] = None):
        """
        Generate a streaming response using the LLM.
        Yields chunks as they arrive from the LLM.
        
        Args:
            user_message: The user's message
            conversation_context: Optional conversation history (from HTML log)
            notebook_context: Optional current notebook structure
            
        Yields:
            String chunks of the response as they arrive
            
        Raises:
            Exception: If the LLM call fails (fail early principle)
        """
        try:
            if not self.client:
                raise RuntimeError("LLM client not initialized")
            
            # Build messages - no hidden state, everything explicit
            messages = self._build_messages(user_message, conversation_context, notebook_context)
            
            logger.info(f"Sending streaming request to {self.model_name}")
            logger.debug(f"User message: {user_message}")
            
            # Make the streaming API call
            for chunk in self.client.stream(messages):
                if chunk.content:
                    yield chunk.content
            
            logger.info(f"Completed streaming response")
            
        except Exception as e:
            # Fail early and often - complete error logging with stack traces
            logger.error(f"LLM streaming failed: {e}")
            logger.error(f"User message: {user_message}")
            logger.error(f"Model: {self.model_name}")
            logger.error(traceback.format_exc())
            raise

    def _build_messages(self, user_message: str, conversation_context: Optional[str] = None, notebook_context: Optional[str] = None):
        """
        Build message list for LLM calls.
        Extracted into helper method to avoid duplication between streaming and regular calls.
        """
        messages = []
        
        # Add system prompt
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        
        # Add notebook structure context if provided
        if notebook_context:
            notebook_message = f"Current notebook context:\n{notebook_context}\n\n"
            messages.append(SystemMessage(content=notebook_message))
        
        # Add conversation context if provided
        if conversation_context:
            context_message = f"Previous conversation context:\n{conversation_context}\n\n"
            messages.append(SystemMessage(content=context_message))
        
        # Add current user message
        messages.append(HumanMessage(content=user_message))
        
        return messages
    
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
    Convenience function for generating responses.
    
    Args:
        user_message: The user's message
        conversation_context: Optional conversation history
        notebook_context: Optional current notebook structure
        
    Returns:
        The LLM's response
    """
    llm = get_llm_interface()
    return llm.generate_response(user_message, conversation_context, notebook_context) 