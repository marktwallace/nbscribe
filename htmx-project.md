# htmx Migration Project Checklist

## Overview
Complete migration from iframe-embedded Jupyter to htmx-based notebook interface with server-side rendering, SSE streaming, and direct nbformat I/O.

**Backward compatibility is a non-goal** - we will proactively remove all old code that is no longer needed.

## Phase 1: Core Infrastructure (M0)

### 1. Dependencies & Requirements
- [ ] **Update environment.yml** with new dependencies:
  - `nbformat>=5.9,<6` (for direct notebook I/O)
  - `jupyter_client>=8.6,<9` (for kernel management)
  - `ipykernel>=6.29,<7` (kernel runtime)
  - `bleach>=6.1,<7` (HTML output sanitization)
- [ ] **Remove unused dependencies**:
  - Review if `websockets` is still needed
  - Check if `beautifulsoup4` is only used by old code

### 2. New Core Modules

#### 2.1 Notebook I/O Module (`src/nbio.py`)
- [ ] **Create `src/nbio.py`** with functions:
  - `read_nb(path: str) -> dict` - load notebook with cell ID generation
  - `write_nb(path: str, nb: dict) -> None` - save notebook to disk
  - `ensure_cell_ids(nb: dict) -> None` - add stable UUIDs to cells
  - `find_cell_index(nb: dict, cell_id: str) -> int | None` - locate cell by ID

#### 2.2 Kernel Management Module (`src/kernel.py`)
- [ ] **Create `src/kernel.py`** with:
  - `KernelPool` class for per-notebook kernel management
  - `execute_stream(nb_path: str, code: str) -> AsyncIterator[str]` - SSE event generator
  - Output rendering functions for stream/error/display_data/execute_result
  - HTML sanitization for safe output display

#### 2.3 Notebook Routes (`src/notebook_routes.py`)
- [ ] **Create `src/notebook_routes.py`** with endpoints:
  - `GET /nb/{path:path}` - serve notebook page
  - `POST /nb/{path:path}/cells/{cell_id}/save` - save cell content
  - `POST /nb/{path:path}/cells/{cell_id}/run` - trigger cell execution
  - `GET /nb/{path:path}/cells/{cell_id}/stream` - SSE output stream

### 3. Template System

#### 3.1 Base Template (`templates/base.html`)
- [ ] **Create `templates/base.html`** with:
  - htmx 1.9.12 + SSE extension includes
  - CodeMirror 5 includes (CSS + JS + Python mode)
  - Base styling for cells, toolbars, and outputs
  - JavaScript for CodeMirror initialization and keyboard shortcuts (Ctrl/Cmd+Enter)

#### 3.2 Notebook Template (`templates/notebook.html`)
- [ ] **Create `templates/notebook.html`** that:
  - Extends base template
  - Shows notebook path in header
  - Loops through cells and includes `_cell.html` fragment

#### 3.3 Cell Fragment Template (`templates/_cell.html`)
- [ ] **Create `templates/_cell.html`** with:
  - Container div with cell ID
  - Code cells: form with textarea, Save/Run buttons, output div
  - Markdown cells: form with textarea, Save button
  - htmx directives for Save (`hx-post`, `hx-target`, `hx-swap`)
  - Conditional SSE connection for Run operations (`sse-connect` when `run_now=True`)

### 4. Server Integration

#### 4.1 FastAPI App Modifications (`src/server.py`)
- [ ] **Expose templates** on `app.state.templates` for route access
- [ ] **Include notebook router** in app with error handling
- [ ] **Keep chat endpoints** (`/api/chat`, `/api/chat/stream`) unchanged
- [ ] **Keep session/picker endpoints** (`/`, `/new`, `/notebook/{path}` redirect)

#### 4.2 Notebook Picker Updates
- [ ] **Update `templates/notebook_picker.html`** links:
  - Change `/notebook/{path}` links to `/nb/{path}`
  - Maintain existing create notebook functionality

## Phase 2: Legacy Code Removal

### 5. Remove Jupyter Proxy System
- [ ] **Delete/comment Jupyter proxy routes** in `src/server.py`:
  - All `/jupyter/*` HTTP proxy routes (`jupyter_contents_proxy`, etc.)
  - `jupyter_specific_proxy_impl` function
  - Jupyter WebSocket proxy routes (`jupyter_events_websocket`, `jupyter_kernel_websocket`)
- [ ] **Remove Jupyter server management**:
  - `notebook_manager` global variable
  - `/api/notebook/start`, `/api/notebook/stop`, `/api/notebook/status` endpoints
  - `startup_event`/`shutdown_event` notebook server lifecycle
- [ ] **Remove filtering and protection**:
  - `JupyterRepetitiveFilter` class
  - `KernelProtection` class
  - Associated logging filters

### 6. Remove React/SPA Components
- [ ] **Delete React notebook interface**:
  - `static/notebook/NotebookApp.js`
  - `static/notebook/CodeCellEditor.js` (if exists)
  - Associated React-specific static files
- [ ] **Remove old notebook route**:
  - `GET /notebook/{notebook_path}` endpoint that serves iframe interface
  - Template rendering for iframe-based interface

### 7. Clean Up Utility Functions
- [ ] **Remove Jupyter API helpers**:
  - `get_notebook_content` function (replace with `nbio.read_nb`)
  - `write_notebook_content` function (replace with `nbio.write_nb`)
  - `trigger_notebook_refresh` function (no longer needed)
- [ ] **Keep and adapt session helpers**:
  - `generate_session_id`, `extract_notebook_path_from_session`
  - `get_latest_session`, `load_session_data`
  - `get_recent_notebooks` (but update for new routes)

## Phase 3: Directive System Integration

### 8. Adapt Directive System
- [ ] **Update `/api/directives/approve` endpoint**:
  - Replace `insert_notebook_cell`/`edit_notebook_cell`/`delete_notebook_cell` to use `nbio` functions
  - Return success/failure without triggering notebook refresh
  - Optional: return htmx fragments for OOB updates
- [ ] **Keep directive parsing**:
  - `src/tool_directive_parser.py` unchanged
  - `static/markdown-renderer.js` directive creation unchanged
  - Directive approval flow in frontend unchanged
- [ ] **Update conversation logging**:
  - Ensure directive results are still logged to conversation
  - Maintain audit trail functionality

### 9. Cell Operations via nbio
- [ ] **Implement cell insertion**:
  - Use `nbio.read_nb`/`nbio.write_nb` instead of Jupyter Contents API
  - Handle `pos`, `before`, `after` positioning logic
  - Generate stable cell IDs for new cells
- [ ] **Implement cell editing**:
  - Locate cell by ID, update source, clear outputs
  - Preserve cell metadata and execution count
- [ ] **Implement cell deletion**:
  - Remove cell from notebook structure
  - Handle position adjustments for remaining cells

## Phase 4: Enhanced Features (M1+)

### 10. Additional Cell Operations
- [ ] **Add cell manipulation endpoints**:
  - `POST /nb/{path}/cells/add?after={idx}&type={code|markdown}` - add new cell
  - `POST /nb/{path}/cells/{cell_id}/delete` - delete cell
  - `POST /nb/{path}/cells/{cell_id}/move?direction=up|down` - reorder cells
- [ ] **Add notebook-level operations**:
  - `POST /nb/{path}/save` - save entire notebook
  - Autosave functionality with debouncing

### 11. UI/UX Improvements
- [ ] **Add execution indicators**:
  - Spinners during cell execution (`hx-indicator`)
  - Execution count display
  - Last-run duration tracking
- [ ] **Enhance output display**:
  - Error traceback formatting with folding
  - Image display for matplotlib plots
  - HTML output with safe rendering
- [ ] **Keyboard shortcuts**:
  - Ctrl/Cmd+Enter to run cell
  - Cell navigation shortcuts
  - Add cell shortcuts

### 12. Security & Robustness
- [ ] **Add security measures**:
  - CSRF protection for POST requests
  - Request timeouts and limits per execution
  - Kill runaway kernels functionality
- [ ] **Output sanitization**:
  - Use `bleach` for HTML output cleaning
  - Prevent XSS in cell outputs
- [ ] **Error handling**:
  - Graceful degradation for kernel failures
  - User-friendly error messages
  - Recovery from corrupted notebooks

## Phase 5: Testing & Validation

### 13. Testing Strategy
- [ ] **Manual testing checklist**:
  - Load existing notebook → edit cell → save → run → see output
  - Create new notebook → add cells → run → save to disk
  - Test directive system with new backend
  - Test SSE streaming with various output types
- [ ] **Integration testing**:
  - Chat + directive → notebook edit → verification
  - Multiple cell types (code/markdown)
  - Error handling and recovery scenarios
- [ ] **Performance validation**:
  - SSE streaming performance vs old system
  - Cell load times for large notebooks
  - Memory usage with kernel pools

### 14. Documentation Updates
- [ ] **Update README.md**:
  - Remove references to Jupyter server management
  - Document new htmx-based architecture
  - Update installation and usage instructions
- [ ] **Update project plans**:
  - Mark completed migration milestones
  - Document architectural decisions
  - Update feature roadmap

## Migration Notes

### Compatibility Breaks (Intentional)
- **No more iframe embedding** - completely different UI approach
- **No WebSocket proxying** - SSE replaces WebSocket for output streaming
- **No Jupyter server dependency** - direct kernel management via `jupyter_client`
- **Different URL structure** - `/nb/{path}` instead of `/notebook/{path}`

### Preserved Functionality
- **Chat interface** - unchanged endpoints and behavior
- **Directive system** - same parsing and approval flow
- **Session management** - same session ID and conversation logging
- **Notebook file format** - still standard `.ipynb` files
- **AI integration** - same LLM interface and prompting

### Success Criteria
- [ ] **M0 Complete**: Single cell edit → save → run → stream output
- [ ] **M1 Complete**: Multi-cell notebook with add/delete/move operations
- [ ] **Legacy Removal**: All Jupyter proxy and React code removed
- [ ] **Directive Integration**: AI directives work with new backend
- [ ] **Feature Parity**: All essential notebook operations working

---

## Implementation Order

1. **Start with M0 core**: New modules + templates + basic routes
2. **Test single cell flow**: Edit → Save → Run → Output streaming
3. **Remove legacy code**: Clean up old proxy and React systems
4. **Integrate directives**: Adapt approval system to new backend
5. **Add enhanced features**: Cell manipulation and UI improvements
6. **Security hardening**: CSRF, sanitization, timeouts
7. **Final testing**: Comprehensive validation and performance testing

This checklist provides a systematic approach to migrating from the iframe-based architecture to the modern htmx approach while maintaining all essential functionality and improving AI assistant integration.
