# nbscribe - Project Plan & Tasks

## Overview
AI-powered Jupyter Notebook assistant with a lightweight FastAPI server and HTML chat interface.

## Phase 1: Proof of Concept
*Goal: Get basic functionality working to start learning and iterating*

### Core Infrastructure
- [x] Set up basic FastAPI server (`server.py`)
  - [x] Static file serving
  - [x] Basic REST endpoints
  - [x] Health check endpoint
- [x] Create minimal HTML chat interface
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
- [x] Design live HTML conversation log format
  - [x] Self-contained document with embedded CSS/JS
  - [x] Conversation metadata (session ID, timestamps)  
  - [x] Clean message structure for tight logging
  - [x] Implement `<chat-msg>` custom elements with semantic CSS
- [x] Implement HTML parsing for conversation context
  - [x] Create conversation parser module
  - [x] Extract messages from `<chat-msg>` elements
  - [x] Format context for LLM input
- [x] **File-First Architecture Implementation**
  - [x] HTML file as canonical source of truth
  - [x] Page refresh always works smoothly
  - [x] Session-based file management
- [x] **Session Routing (Hybrid Option D)**
  - [x] `http://localhost:5317/` → Latest session or new
  - [x] `http://localhost:5317/session/ID` → Specific session
  - [x] `http://localhost:5317/new` → Force new session
  - [x] Auto-redirect logic for session continuity
- [x] Implement HTML log persistence
  - [x] Save conversation as standalone HTML file after each message
  - [x] Load existing conversation log on startup/session load
  - [x] Session file naming: `YYYYMMDD_HHMMSS_mmm` format
  - [x] External CSS to eliminate duplication
  - [x] Initial greeting message persistence
- [x] **Progressive Enhancement (Future UX)**
  - [x] JavaScript optimizations for latency (streaming responses)
  - [x] DOM updates without page refresh (when beneficial)
  - [x] Fallback: Page refresh always works
  - [x] Streaming with Server-Sent Events (SSE)
  - [x] Auto-fallback to regular endpoint if streaming fails
  - [x] **Professional Document-Style Interface**
    - [x] ChatGPT-style message layout (user indented, assistant full-width)
    - [x] Natural browser scrolling (no embedded scroll areas)
    - [x] Full-width design for technical content review
    - [x] Responsive design for screen sharing with Jupyter
    - [x] Blinking dots loading indicator
    - [x] Auto-resizing textarea input
  - [x] **Markdown Rendering**
    - [x] Client-side markdown parsing with marked.js
    - [x] Lightweight custom CSS (GitHub-inspired, proper list spacing)
    - [x] Raw markdown storage in HTML files (human readable)
    - [x] Streaming-compatible rendering (render on completion)
    - [x] Code syntax highlighting and formatting
    - [x] Consistent rendering in live and archived conversations

**Architecture Principle: File-First + Progressive Enhancement**
- HTML files in `logs/conversations/` are the source of truth
- URLs map directly to conversation files
- JavaScript enhances UX but page refresh is always reliable
- Local app performance makes this approach snappy

**CSS Organization Convention:**
- `/static/conversation.css`: All markdown and shared styles (self-contained, no CDN)
- Live chat interface (`chat.html`): Layout CSS inline for self-contained document
- Archived logs (`conversation_log.html`): Minimal template, external CSS only
- Keep templates clean and avoid CSS duplication
- Lightweight approach: custom CSS instead of heavy framework dependencies

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
  - [ ] HTML conversation log as context source
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