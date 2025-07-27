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
- [x] Create `llm_interface.py` stub
  - [x] ~~Hardcoded response for testing~~ (skipped - went straight to LLM)
  - [x] Basic OpenAI integration (LangChain connection)
  - [x] Simple prompt loading
- [x] Wire chat interface to backend
  - [x] Form submission to API
  - [x] Display responses in chat
  - [x] Basic error handling

### Chat Log System (Priority)
- [x] Design live XHTML conversation log format
  - [x] Self-contained document with embedded CSS/JS
  - [x] Conversation metadata (session ID, timestamps)
  - [x] Clean message structure for tight logging
- [ ] Implement XHTML log persistence
  - [ ] Save conversation as standalone XHTML file
  - [ ] Load existing conversation log on startup
  - [ ] Parse XHTML to extract conversation context
- [ ] Enhance log format
  - [ ] Timestamps for each message
  - [ ] Code block syntax highlighting
  - [ ] Message threading/context

### Notebook Integration (After Chat Log)
- [ ] Create `notebook_editor.py`
  - [ ] Read notebook with `nbformat`
  - [ ] Basic cell insertion
  - [ ] Save notebook changes
- [ ] Test with a sample notebook
  - [ ] Create test notebook
  - [ ] Verify read/write operations

## Phase 2: Enhanced Features
*After proof of concept is working*

- [x] Real OpenAI integration
- [x] Prompt system (`prompts/` folder)
- [ ] Better UI/UX
- [ ] **Conversation Memory & Rolling**
  - [ ] XHTML conversation log as context source
  - [ ] Log rolling to archive directory with summarization
  - [ ] Project memories (explicit, user-visible files)
- [ ] Jupyter integration improvements
- [ ] Error handling & validation
- [ ] **Future Memory Features**
  - [ ] Conversation summarization when rolling logs
  - [ ] Project-specific memory files (user-editable)
  - [ ] Memory management UI

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