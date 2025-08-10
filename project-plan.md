# nbscribe - Project Plan & Tasks

## Overview
AI-powered Jupyter Notebook assistant with a lightweight FastAPI server and HTML chat interface.

## Current State Snapshot (Aug 2025)
- Jupyter Notebook 7 is embedded via an iframe; the assistant proxies `/jupyter/*` as a minimal pass-through.
- Kernel WebSocket proxy is a dumb tunnel: forwards text and binary frames, allows rapid reconnects, and does not throttle kernel GET polling.
- After applying a directive, we refresh the open document in-place using JupyterLab commands when available. Classic full-page reloads are no longer acceptable.
- JSON-first linear conversation is in place: initial User message carries full notebook JSON; post-approval messages include precise cell JSON updates/deletions. System prompt documents delete semantics.
- Outstanding: token counting and conversation summarization. Optional future: point the iframe directly to the Jupyter server URL (no proxy) given intranet/VPN deployment and no CORS concerns.

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
  - [x] **Fix system prompt loading** (`system_prompt.txt` vs `main.txt`)
  - [x] **System prompt transparency** (expandable UI display)
- [x] Wire chat interface to backend
  - [x] Form submission to API
  - [x] Display responses in chat
  - [x] Basic error handling
  - [x] **Streaming responses with SSE**
  - [x] **Tool directive parsing and UI**

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

- [x] **Notebook 7 Server Management**
  - [x] Launch Jupyter Notebook 7 as subprocess with fixed token
  - [x] Auto-discover free port (start with 8889)
  - [x] Implement FastAPI reverse proxy for `/jupyter/*` routes
  - [x] Manage server lifecycle (start/stop with nbscribe)
  - [x] Kernel WebSocket proxy simplified to pass-through (text/bytes, no dedupe; allow kernel GET polling)
  - [x] Clean console logging (filter repetitive noise)
- [x] **Split-Pane UI**  
  - [x] Update HTML templates for two-column layout
  - [x] Left pane: Notebook iframe (`/jupyter/notebooks/<file>`)
  - [x] Right pane: Chat panel (existing interface)
  - [x] Maintain responsive design
  - [x] Chat-only mode (no notebook pane for `/chat` route)
  - [x] Notebook picker with proper scrolling
- [x] **AI Tool Directive System**
  - [x] Parse Markdown tool proposals (`TOOL: insert_cell`, `edit_cell`, `delete_cell`)
  - [x] Extract metadata (`POS:`, `CELL_ID:`) and Python code blocks
  - [x] **BEFORE/AFTER cell ID positioning** for inserts (with POS fallback)
  - [x] Render approval buttons **below** code blocks for better UX
  - [x] Handle approve/reject actions in chat interface
  - [x] System prompt transparency (expandable view)
  - [x] Fix system prompt loading (`system_prompt.txt` vs `main.txt`)
  - [x] **Updated validation and parsing** for relative positioning
- [x] **Directive rendering hardening**
  - [x] Tolerant regex for tool blocks (optional fence languages, CRLF, whitespace)
  - [x] Support `json` metadata fences and case-insensitive keys
  - [x] Clear inline error when malformed
- [x] **Notebook Modification REST API** ✅ **BASIC IMPLEMENTATION COMPLETE**
  - [x] Basic endpoint structure and validation
  - [x] Tool directive request/response models  
  - [x] Error handling and logging
  - [x] BEFORE/AFTER/POS parameter support in models
  - [x] **Core Implementation: Jupyter API Integration**
    - [x] **Session Context Management**
      - [x] Extract notebook path from session/URL context
      - [x] Validate notebook exists and is accessible
      - [x] Handle concurrent modification detection
    - [x] **Cell ID Management Strategy** 
      - [x] Ensure all cells have stable UUIDs in metadata
      - [x] Add cell IDs to existing notebooks on first access
      - [x] Use `cell.metadata.id` field (Jupyter standard)
    - [x] **Read Notebook Operation**
      - [x] `GET /jupyter/api/contents/{notebook_path}` with full content
      - [x] Parse nbformat JSON structure
      - [x] Extract cell list with IDs and content
    - [x] **Position Resolution Logic**
      - [x] BEFORE: Find target cell by ID, insert at `target_index`
      - [x] AFTER: Find target cell by ID, insert at `target_index + 1`
      - [x] POS: Direct index insertion (fallback for empty notebooks)
      - [x] Handle edge cases: cell not found, index out of bounds
    - [x] **Cell Modification Operations**
      - [x] INSERT: Create new cell with UUID, insert at resolved position
      - [x] EDIT: Find cell by ID, update `source` field, preserve metadata/outputs
      - [x] DELETE: Find cell by ID, remove from cells array
    - [x] **Write Notebook Operation**  
      - [x] Validate modified notebook structure (nbformat compliance)
      - [x] `PUT /jupyter/api/contents/{notebook_path}` with updated content
      - [x] Handle write conflicts and error recovery
  - [x] **Error Handling & Edge Cases**
    - [x] Cell ID not found → helpful error message
    - [x] Notebook file conflicts → retry logic or user notification
    - [x] Invalid notebook format → validation and recovery
    - [x] Permission errors → clear user feedback

- [ ] **Linear Conversation Architecture** ⭐ **CURRENT FOCUS**
  - [ ] **Conversation Flow Design**
    - [x] Replace system message injection with linear conversation history
    - [x] First User message contains complete notebook source at conversation start
    - [x] Assistant sees notebook before offering help (natural flow)
    - [x] Subsequent conversation maintains linear message history
    - [x] Post-modification User messages report cell changes to AI
  - [ ] **Message Type Architecture**
    - [ ] System: Prompt instructions + conversation summary (on restart)
    - [x] User: Initial notebook source, human requests, cell state updates
    - [x] Assistant: Help offers, code proposals with TOOL directives
  - [ ] **Cell State Tracking**
    - [x] Format complete notebook as first User message
    - [x] Post-approval: "Cell {id} {operation}: {content}" as User messages
    - [x] Include successful operations, failures, and deletions
    - [ ] Wrap state updates in identifiers for UI compacting/expanding
  - [ ] **Conversation Management**
    - [ ] Token count-based conversation restart triggers
    - [ ] User-controlled manual conversation restart
    - [ ] Conversation summarization before restart
    - [ ] Preserve transparency while optimizing for LLM performance
  - [ ] **Delete Operation Handling**
    - [ ] Clear "Cell {id} deleted" indicators in conversation
    - [ ] System prompt explains significance of deletion to AI
    - [ ] Maintain referential integrity in conversation context

**Architecture Decisions Made:**
- **Cell ID Strategy**: Use Jupyter's standard `cell.metadata.id` field with UUIDs
- **Positioning Approach**: BEFORE/AFTER cell IDs (primary), POS index (fallback)
- **Modification Method**: File-based via Jupyter Contents API (not live kernel)
- **Concurrency**: Simple conflict detection, no complex locking
- **AI Integration**: LINEAR conversation with notebook source as User messages (NOT system injection)
- **State Management**: AI never tracks state - all updates explicit via User messages
- **Conversation Lifecycle**: Token-based restart with summarization for performance

**Implementation Priority (Linear Conversation Architecture):**
1. **Conversation Message Builder** - Format notebook source for initial User message
2. **LLM Interface Refactor** - Replace system injection with linear conversation history
3. **Post-Modification Messaging** - Add cell update User messages after approvals
4. **Conversation Length Tracking** - Token counting and restart trigger logic
5. **Conversation Summarization** - Summarize and restart conversations
6. **System Prompt Updates** - Include delete operation significance and examples
7. **UI State Message Compacting** - Expandable/collapsible state update messages

## Focused Status: No-Prompt In-Place Refresh (Aug 2025)

- Problem: Clicking Accept sometimes triggers a browser "Leave site?" dialog. Full navigations (Classic `/jupyter/notebooks/...`) cause prompts and kernel churn.
- Target behavior: Always refresh the notebook in-place without navigation using JupyterLab commands.
- Plan:
  - Serve the iframe at the Lab document route: `/jupyter/lab/tree/<notebook_path>` so `window.jupyterapp.commands` is available.
  - On Accept, execute `docmanager:revert` (or `docmanager:reload`) inside the iframe to refresh the model without page navigation.
  - On iframe load, minimize the Lab chrome (hide left area, single-document mode) so the UI remains notebook-only.
- Current code:
  - Frontend (`static/chat-interface.js`): initializes the iframe, hides left panel, tries Lab commands on Accept, falls back only if Lab app missing.
  - Frontend (`static/markdown-renderer.js`): tolerant tool parsing; renders approve buttons reliably.
  - Prompt (`prompts/system_prompt.txt`): explicitly requires ```python code fence and an immediate second fenced metadata block (or ```json), no prose between.
- Remaining action:
  - Ensure the template uses `/jupyter/lab/tree/<notebook_path>` for `notebook_iframe_url` so the Lab command registry is always present. This removes Classic reloads and the browser prompt.

- [ ] **Audit Logging Integration**
  - [x] Save full Markdown chat history with tool directives
  - [x] Conversation persistence as HTML files
  - [ ] Store as `notebook_name__nbscribe_log.md` alongside notebook
  - [ ] Git-friendly format for version control and reproducibility

**Architecture: Single notebook session for POC, expand to multi-notebook later**
**AI never executes directly - all changes require human approval**

**Example Linear Conversation Flow (JSON-First):**
```
1. SYSTEM: [Full system prompt with tool instructions and delete significance]

2. USER: Complete notebook JSON:
{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 1,
      "id": "abc123",
      "metadata": {},
      "outputs": [],
      "source": ["import pandas as pd\nimport numpy as np"]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "id": "def456", 
      "metadata": {},
      "outputs": [{"name": "stdout", "output_type": "stream", "text": ["   A  B\n0  1  4\n1  2  5\n2  3  6"]}],
      "source": ["df = pd.read_csv('data.csv')\nprint(df.head())"]
    }
  ],
  "metadata": {"kernelspec": {"name": "python3"}},
  "nbformat": 4,
  "nbformat_minor": 4
}

3. ASSISTANT: I can see your notebook with 2 cells for data loading. The CSV loaded successfully with 3 rows. How can I help you analyze this data?

4. USER: Add a histogram of the data

5. ASSISTANT: Here's a histogram visualization you can add:

```python
import matplotlib.pyplot as plt
df.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()
```

```
TOOL: insert_cell
AFTER: def456
```

6. USER: Cell inserted:
{
  "cell_type": "code",
  "execution_count": null,
  "id": "ghi789",
  "metadata": {},
  "outputs": [],
  "source": ["import matplotlib.pyplot as plt\ndf.hist(figsize=(10, 8))\nplt.tight_layout()\nplt.show()"]
}

7. USER: Now add error handling to the data loading

8. ASSISTANT: I'll update the data loading cell with error handling:

```python
try:
    df = pd.read_csv('data.csv')
    print(f"Loaded {len(df)} rows")
    print(df.head())
except FileNotFoundError:
    print("CSV file not found")
    df = pd.DataFrame()
```

```
TOOL: edit_cell
CELL_ID: def456
```

9. USER: Cell updated:
{
  "cell_type": "code",
  "execution_count": 3,
  "id": "def456",
  "metadata": {},
  "outputs": [],
  "source": ["try:\n    df = pd.read_csv('data.csv')\n    print(f\"Loaded {len(df)} rows\")\n    print(df.head())\nexcept FileNotFoundError:\n    print(\"CSV file not found\")\n    df = pd.DataFrame()"]
}
```

**Success Criteria for Notebook Modification:**
1. ✅ AI suggests: `AFTER: data-456` → Cell inserted after data loading
2. ✅ AI suggests: `BEFORE: analysis-789` → Cell inserted before analysis  
3. ✅ AI suggests: `CELL_ID: setup-123` → Existing cell edited/deleted
4. ✅ Empty notebook: `POS: 0` → First cell created
5. ✅ Cell not found → Clear error message, no notebook corruption
6. ✅ Notebook structure preserved → Metadata, outputs, execution counts maintained

---

## 🎯 **Next Implementation Guide - Linear Conversation Architecture**

**Immediate Focus:** Implement linear conversation architecture to replace system message injection:

### **Phase 1: Core Linear Conversation (Essential)**
1. **JSON Notebook Formatter** (`src/conversation_manager.py`)
   - `format_notebook_for_ai()` - Convert .ipynb JSON to formatted User message
   - `format_cell_update()` - Create cell JSON fragments for post-modification User messages
   - Handle cell insertions, edits, deletions, and failures as complete JSON cell objects

2. **LLM Interface Refactor** (`src/llm_interface.py`)
   - Replace `_build_messages()` to use linear conversation history
   - Remove system message injection of notebook context
   - Load conversation history from HTML logs for context

3. **Post-Modification Integration** (`src/server.py`)
   - Add cell update messages after successful directive approvals
   - Include failure messages as User messages for transparency
   - Integrate with existing `approve_directive()` endpoint

### **Phase 2: Conversation Management (Performance)**
4. **Token Counting & Restart Logic** (`src/conversation_manager.py`)
   - Implement token counting for conversation length
   - Add restart trigger logic (token threshold + user control)
   - Create conversation summarization before restart

5. **UI Message Compacting** (`static/markdown-renderer.js`)
   - Add collapsible/expandable state update messages
   - Preserve transparency while reducing visual clutter
   - Maintain full message content for training/debugging

### **Phase 3: Enhanced Prompting (Quality)**
6. **System Prompt Updates** (`prompts/system_prompt.txt`)
   - Add delete operation significance and examples
   - Include linear conversation flow examples
   - Clarify AI's role in state tracking (AI never tracks, only responds)

**Key Files to Create/Modify:**
- `src/conversation_manager.py` - **NEW** - Core conversation logic
- `src/llm_interface.py` - **MAJOR REFACTOR** - Linear conversation support
- `src/server.py` - **MODERATE** - Post-modification messaging
- `prompts/system_prompt.txt` - **UPDATE** - Delete handling + examples
- `static/markdown-renderer.js` - **MINOR** - UI message compacting

**JSON Implementation Examples:**

```python
# src/conversation_manager.py

def format_notebook_for_ai(notebook_content: dict) -> str:
    """Convert complete .ipynb JSON to User message format"""
    return f"Complete notebook JSON:\n{json.dumps(notebook_content, indent=2)}"

def format_cell_update(cell_data: dict, operation: str) -> str:
    """Format cell update as complete JSON cell object"""
    return f"Cell {operation}:\n{json.dumps(cell_data, indent=2)}"

def format_cell_deletion(cell_id: str) -> str:
    """Format cell deletion notification"""
    return f"Cell deleted: {cell_id}"
```

**Success Criteria:**
1. ✅ AI sees complete notebook JSON before first interaction
2. ✅ AI can reference specific cell IDs from JSON structure in conversation history
3. ✅ Cell modifications are tracked via complete JSON cell objects in User messages
4. ✅ Token-based conversation restart works smoothly with JSON format
5. ✅ Delete operations are clearly communicated to AI with cell ID significance
6. ✅ AI leverages native .ipynb JSON comprehension for better understanding

### **Jupyter Configuration Philosophy**
**Goal**: Minimal, predictable Jupyter; assistant only tunnels traffic necessary for kernels to function.

**Effective settings:**
- ✅ `--ServerApp.base_url=/jupyter` and fixed token
- ✅ `--NotebookApp.terminals_enabled=False`

**Do not interfere with kernel operation:**
- Allow kernel channel WebSockets to reconnect freely (no dedupe throttling on WS).
- Allow kernel status GET polling (no throttling on GETs).
- Forward both text and binary WS frames.

**Optional future:** Direct iframe to Jupyter URL (no proxy) given intranet/VPN deployment.

---

## 🧠 **Linear Conversation Architecture Philosophy**

### **Why Linear Conversation vs System Message Injection?**

**Previous Approach (System Message Injection):**
```
SYSTEM: [Prompt instructions]
SYSTEM: [Notebook structure - regenerated each call]  ← Expensive & non-cacheable
SYSTEM: [Conversation history]
USER: [Human request]
```

**New Approach (Linear Conversation with JSON):**
```
SYSTEM: [Prompt instructions + summary on restart]
USER: [Complete .ipynb JSON at start]               ← AI sees full notebook structure
ASSISTANT: [Offers help after seeing notebook]
USER: [Human request]
ASSISTANT: [Code proposal with TOOL directive]
USER: [Cell JSON fragment with updates]             ← Complete cell data as JSON
USER: [Next human request]
... continues linearly
```

### **Benefits for AI Understanding:**

1. **Natural Code Review Flow**: AI sees complete .ipynb JSON → understands structure → offers help (vs blind suggestions)
2. **Native Format Comprehension**: LLMs trained on .ipynb files understand JSON format natively
3. **Complete Context**: Sees outputs, execution counts, metadata, cell types - not just source code
4. **Clear State Tracking**: Updates come as complete cell JSON objects, not inference from system injection
5. **No State Management Burden**: AI doesn't track what changed, it's told directly via JSON fragments
6. **Conversation Caching**: Linear messages enable LLM caching, reducing costs
7. **Referential Integrity**: Cell IDs in JSON directly match TOOL directive CELL_ID parameters

### **Benefits for Performance:**

1. **LLM Caching**: Linear conversation history enables token-level caching
2. **Incremental Updates**: Only new messages added, not full notebook regeneration
3. **Cost Optimization**: Fewer redundant tokens in system messages
4. **Token Management**: Clear conversation restart triggers based on length

### **Benefits for Transparency:**

1. **All State Visible**: Every change appears as complete cell JSON in conversation log
2. **Training Data Ready**: Linear conversations with native .ipynb JSON perfect for fine-tuning datasets
3. **Human Readable**: JSON format familiar to developers, conversation flow mirrors natural code review
4. **Debugging Clarity**: Easy to trace AI decision-making through message history with complete cell context
5. **Format Preservation**: No information loss from custom syntax - outputs, metadata, execution counts all preserved
6. **Jupyter Ecosystem Alignment**: Uses standard .ipynb format that all Jupyter tools understand

## Phase 2: Enhanced Features
*After linear conversation architecture is working*

- [x] Real OpenAI integration
- [x] Prompt system (`prompts/` folder)
- [x] **Core Conversation Management** - *Moved to Phase 1 (Essential)*
  - [x] HTML conversation log as context source
  - [ ] Linear conversation architecture implementation
  - [ ] Token-based conversation restart with summarization
- [ ] **Advanced Conversation Features**
  - [ ] Project memories (explicit, user-visible files)
  - [ ] Cross-session memory persistence
  - [ ] Advanced conversation summarization strategies
- [ ] **UI/UX Enhancements**
  - [ ] Better notebook integration visual design
  - [ ] Conversation message compacting/expanding
  - [ ] Performance optimizations for large conversations
-- [ ] **Jupyter Integration Improvements**
  - [x] Kernel WebSocket pass-through stabilized
  - [x] Immediate cell visibility via iframe reload
  - [ ] Optional: direct iframe to Jupyter (no proxy) for even simpler architecture
  - [ ] Multi-notebook session support
- [ ] **Advanced Error Handling & Validation**
  - [ ] Conversation integrity validation
  - [ ] Advanced conflict resolution for concurrent modifications
  - [ ] Robust recovery from conversation corruption

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

## Success Criteria for Phase 1 (Linear Conversation Architecture)
1. ✅ Server starts without errors
2. ✅ Chat interface loads in browser  
3. ✅ Can send a message and get a response
4. ✅ Can create/modify a simple notebook (basic implementation complete)
5. ✅ Conversation history persists
6. ✅ AI sees complete notebook source before offering help
7. ✅ AI can reference specific cells from conversation history
8. ✅ Cell modifications generate automatic state update User messages
9. [ ] Token counting triggers conversation restart appropriately
10. ✅ Delete operations are clearly communicated to AI with significance

---

*Keep it simple, get it working, then iterate!* 