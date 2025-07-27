# nbscribe - Project Plan & Tasks

## Overview
AI-powered Jupyter Notebook assistant with a lightweight FastAPI server and XHTML chat interface.

## Phase 1: Proof of Concept
*Goal: Get basic functionality working to start learning and iterating*

### Core Infrastructure
- [x] Set up basic FastAPI server (`server.py`)
  - [x] Static file serving
  - [x] Basic REST endpoints
  - [x] Health check endpoint
- [x] Create minimal XHTML chat interface
  - [x] Basic HTML structure
  - [x] Simple CSS styling
  - [x] JavaScript for form submission
- [x] Set up project dependencies
  - [x] Create `requirements.txt`
  - [x] Add FastAPI, uvicorn, jinja2

### Basic LLM Integration
- [ ] Create `llm_interface.py` stub
  - [ ] Hardcoded response for testing
  - [ ] Basic OpenAI integration (when ready)
  - [ ] Simple prompt loading
- [ ] Wire chat interface to backend
  - [ ] Form submission to API
  - [ ] Display responses in chat
  - [ ] Basic error handling

### Notebook Integration
- [ ] Create `notebook_editor.py`
  - [ ] Read notebook with `nbformat`
  - [ ] Basic cell insertion
  - [ ] Save notebook changes
- [ ] Test with a sample notebook
  - [ ] Create test notebook
  - [ ] Verify read/write operations

### Chat Log System
- [ ] Design `chat_log.xhtml` format
  - [ ] User/assistant message structure
  - [ ] Timestamps
  - [ ] Code block formatting
- [ ] Implement log persistence
  - [ ] Append new messages
  - [ ] Load existing log on startup

## Phase 2: Enhanced Features
*After proof of concept is working*

- [ ] Real OpenAI integration
- [ ] Prompt system (`prompts/` folder)
- [ ] Better UI/UX
- [ ] Log rolling/summarization
- [ ] Jupyter integration improvements
- [ ] Error handling & validation

## Getting Started Commands

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate nbscribe

# OR install with pip
pip install fastapi uvicorn jinja2 nbformat

# Run server
python main.py

# Open browser
open http://localhost:5317
```

## Success Criteria for Phase 1
1. Server starts without errors
2. Chat interface loads in browser
3. Can send a message and get a response
4. Can create/modify a simple notebook
5. Conversation history persists

---

*Keep it simple, get it working, then iterate!* 