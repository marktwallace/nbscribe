# Feature: Notebook 7 Integration with AI-Driven Cell Editing

## Summary

This feature integrates a Jupyter Notebook 7 server into `nbscribe`, presented alongside a GPT‑4-powered chat panel.
Users can view, approve, and apply AI-suggested notebook edits via a safe, human-readable Markdown chat log.
All interactions are auditable and saved alongside the notebook.

---

## Goals

* Run Notebook 7 server embedded in `nbscribe`.
* GPT‑4 can **suggest edits (not execute them)**.
* Human-readable, Markdown-based chat history.
* Structured tool directives parsed from chat.
* HTML renderer shows **approve/reject buttons**.
* Audit log saved as `.md` or `.html` for reproducibility.

---

## Components

### 1. FastAPI Server (`nbscribe`)

* **Responsibilities**

  * Launch and manage Notebook 7 server with known token.
  * Proxy `/jupyter/*` requests to Notebook server.
  * Serve chat UI and Markdown renderer.
  * Parse AI tool directives from chat messages.
  * Expose REST endpoints for:

    * Insert, edit, delete cells.
    * Execute a specific cell.
  * Save full chat history to disk.

* **Endpoints**

  * `POST /api/chat` → send user message, get GPT‑4 response.
  * `POST /api/notebooks/{path}/insert_cell`
  * `PATCH /api/notebooks/{path}/edit_cell`
  * `DELETE /api/notebooks/{path}/delete_cell`
  * `POST /api/notebooks/{path}/execute_cell`
  * `GET /api/logs/{session_id}` → download saved Markdown chat log.

---

### 2. Notebook 7 Server

* Runs as a subprocess with:

  ```bash
  jupyter notebook \
    --NotebookApp.token=nbscribe-token \
    --NotebookApp.password='' \
    --NotebookApp.base_url=/jupyter \
    --no-browser \
    --port=8889
  ```
* Accessible under `/jupyter/` via FastAPI proxy.
* Provides minimal UI for notebook editing and execution.

---

### 3. Chat Panel and Renderer

* **Frontend Layout**

  * Left: Notebook iframe (`/jupyter/tree` or `/jupyter/notebooks/<file>`).
  * Right: Chat panel with GPT‑4 conversation.

* **AI Output Parsing**

  * Each AI message is Markdown.
  * Tool proposals detected via regex:

    ````regex
    >\s*```.*TOOL:\s*(?P<tool>\w+)
    ````
  * Metadata extracted:

    * `TOOL:` (`insert_cell`, `edit_cell`, `delete_cell`)
    * `POS:` (for insertion position)
    * `CELL_ID:` (for edits)
  * Associated Python code extracted from fenced blocks.

* **HTML Rendering**

  * Messages displayed as chat bubbles.
  * Tool directives shown as **approval cards**:

    * Original AI text preserved (for log).
    * “✅ Approve” → calls the matching REST endpoint.
    * “❌ Reject” → marks proposal as ignored.
  * Multiple tool proposals handled sequentially.

* **Audit Log**

  * Full Markdown conversation (including directives and code) saved as:

    * `notebook_name__nbscribe_log.md`
    * Optionally rendered as `.html`.
  * Git-friendly for versioning and reproducibility.

---

## AI Output Format

Example AI message:

````markdown
Here’s how you could plot a histogram of `df['X']`.

> **Cell Edit Proposal**
> ```
> TOOL: insert_cell
> POS: 3
> ```
>
> ```python
> import matplotlib.pyplot as plt
> df['X'].hist()
> plt.show()
> ```
````

---

## Workflow

1. **User message → GPT‑4**

   * FastAPI sends user prompt to GPT‑4 with system prompt specifying tool syntax.
2. **Model response**

   * Chat message stored as Markdown.
   * Tool proposals parsed but **not executed**.
3. **Frontend renders**

   * HTML shows normal text + approval cards.
   * Buttons linked to `/insert_cell` or `/edit_cell`.
4. **User approves**

   * FastAPI applies changes via Notebook REST API.
   * Jupyter server updates live.
5. **Logging**

   * Every message + action appended to Markdown log.
   * Saved automatically with notebook.

---

## Benefits

* AI never executes code directly → safer collaboration.
* Logs are permanent, auditable, and readable.
* Structured directives allow deterministic cell edits.
* Future-proof: Tools can expand without breaking log readability.
* Minimal UI complexity: uses Markdown + HTML buttons.
