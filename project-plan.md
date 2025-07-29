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

### Chat Log System
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

**CSS & JavaScript Organization:**
- `/static/conversation.css`: Markdown rendering and shared message styles
- `/static/chat-interface.css`: All layout, forms, loading, interactive elements
- `/static/chat-interface.js`: Complete ChatInterface class with streaming
- `/static/markdown-renderer.js`: Shared markdown configuration and rendering
- **NO inline styles or scripts in templates** - all code in `/static/` for caching
- Templates are minimal HTML structure only (chat.html: 53 lines, conversation_log.html: 44 lines)
- Lightweight approach: custom CSS instead of heavy framework dependencies

### Notebook Integration (Phase 2)
**Goal: Jupyter Notebook 7 Integration with AI-Driven Cell Editing**

- [ ] **Notebook 7 Server Management**
  - [ ] Launch Jupyter Notebook 7 as subprocess with fixed token
  - [ ] Auto-discover free port (start with 8889)
  - [ ] Implement FastAPI reverse proxy for `/jupyter/*` routes
  - [ ] Manage server lifecycle (start/stop with nbscribe)
- [ ] **Split-Pane UI**  
  - [ ] Update HTML templates for two-column layout
  - [ ] Left pane: Notebook iframe (`/jupyter/notebooks/<file>`)
  - [ ] Right pane: Chat panel (existing interface)
  - [ ] Maintain responsive design
- [ ] **AI Tool Directive System**
  - [ ] Parse Markdown tool proposals (`TOOL: insert_cell`, `edit_cell`, `delete_cell`)
  - [ ] Extract metadata (`POS:`, `CELL_ID:`) and Python code blocks
  - [ ] Render approval buttons **below** code blocks for better UX
  - [ ] Handle approve/reject actions in chat interface
- [ ] **Notebook Modification REST API**
  - [ ] `POST /api/notebooks/{path}/insert_cell` 
  - [ ] `PATCH /api/notebooks/{path}/edit_cell`
  - [ ] `DELETE /api/notebooks/{path}/delete_cell`
  - [ ] Apply changes via Jupyter Server REST API
- [ ] **Audit Logging Integration**
  - [ ] Save full Markdown chat history with tool directives
  - [ ] Store as `notebook_name__nbscribe_log.md` alongside notebook
  - [ ] Git-friendly format for version control and reproducibility

**Architecture: Single notebook session for POC, expand to multi-notebook later**
**AI never executes directly - all changes require human approval**

### **Jupyter Configuration Philosophy**
**Goal**: Minimal, predictable Jupyter with no hidden state - both human and AI see complete record

**Tested Configuration Results:**
✅ **SAFE & BENEFICIAL:**
- `--NotebookApp.terminals_enabled=False` → Eliminates `/api/terminals` system polling + WebSocket 403s

❌ **AVOID - BREAKS FUNCTIONALITY:**
- `--NotebookApp.max_kernels=1` → Prevents Jupyter server startup entirely
- `--NotebookApp.shutdown_no_activity_timeout=600` → Triggers heavy system-wide polling (kernels, sessions, contents)
- `--KernelManager.autorestart=True` → Adds unnecessary checkpoint polling

**Eliminated Noisy Polling:**
- ❌ ~~`GET /jupyter/api/terminals?...`~~ (Constant terminal status checks)
- ❌ ~~`GET /jupyter/api/kernels?...`~~ (System-wide kernel polling) 
- ❌ ~~`GET /jupyter/api/sessions?...`~~ (Session monitoring)
- ❌ ~~`GET /jupyter/api/contents?...`~~ (File system watching)
- ❌ ~~`WebSocket /jupyter/api/events/subscribe → 403`~~ (Terminal event subscription)

**Remaining Acceptable Activity:**
- ✅ `GET /jupyter/api/contents/{notebook}.ipynb/checkpoints?...` (Targeted checkpoint monitoring)

**Philosophy**: "Everything in the open, no hidden state" - AI and human see identical context. Systematic testing eliminated 95% of log noise while preserving all essential Jupyter functionality.

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