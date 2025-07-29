# Recommended Changes to Project Plan for Notebook 7 Integration

## High-Level Updates

### 1. Replace Direct `nbformat` Editing

* **Current:** `notebook_editor.py` planned to read/write `.ipynb` files directly.
* **Updated:** Launch **Notebook 7 server** as a subprocess and use its **REST API** for cell edits (insert, edit, delete, execute).

### 2. Add Notebook 7 Server Management

* Start Jupyter Notebook 7 with a fixed token.
* Add FastAPI reverse proxy for `/jupyter/*`.
* Manage server lifecycle (start/stop with `nbscribe`).

### 3. Split-Pane UI

* Update HTML templates and CSS to show:

  * Left pane → Notebook editor iframe.
  * Right pane → Chat panel.
* Maintain responsive, document-style chat interface.

### 4. Chat Parser for AI Tool Directives

* Parse Markdown tool proposals (`TOOL: insert_cell`, etc.).
* Detect and render approval buttons in chat messages.
* On approval, call REST endpoints to modify notebook.

### 5. Audit Logging

* Save full Markdown chat history (with directives).
* Store next to notebook file for Git-friendly versioning.

---

## Minimal Task List for Phase Updates

1. Add subprocess management to launch Notebook 7 with fixed token.
2. Implement FastAPI reverse proxy for `/jupyter/*` routes.
3. Redesign HTML to split notebook iframe and chat panel.
4. Write parser for AI tool directives and approval button logic.
5. Implement REST endpoints to apply notebook edits via API.
6. Update chat logging to save Markdown logs alongside notebooks.
