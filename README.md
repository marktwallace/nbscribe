# Project Plan: nbscribe - A Jupyter AI Scribe

## Overview

**nbscribe** is an AI-powered assistant designed to enhance Jupyter Notebook workflows using a lightweight FastAPI server. It delivers a minimalist yet powerful XHTML interface for maintaining a human-readable and inspectable conversation log between the user and an LLM (via LangChain). The assistant can suggest or insert notebook edits in real time, track dialogue context for high-quality completions, and eventually support log summarization and context rolling for longer work sessions.

This tool is targeted toward computational researchers who already use Jupyter in their workflows and want an embedded assistant without heavy IDEs, browser plugins, or hidden state. Its closest inspiration is List-Pet, but adapted to the habits of scientific researchers who prefer Python.

---

## Core Components

### 1. `server.py` (FastAPI backend)

* Serves:

  * XHTML chat log (`chat_log.xhtml`)
  * Static files (JS, CSS)
  * REST endpoints for submitting user messages
  * Rolling and summarizing log
* Calls:

  * LangChain agent
  * `nbformat` to read and write notebooks
  * Jupyter Contents API (optional for live reload)

### 2. `llm_interface.py`

* Manages all calls to OpenAI (or other providers) via LangChain.
* Loads prompt sets from the `prompts/` folder.
* Emits structured completions (e.g. `<cell type="code">...</cell>`) for easy parsing.

### 3. `notebook_editor.py`

* Wraps `nbformat` for:

  * Inserting code or markdown cells
  * Editing or replacing existing cells by ID
  * Saving notebook changes

### 4. `templates/`

* Jinja2 templates for XHTML rendering:

  * `index.xhtml.j2`: Live conversation log UI
  * Supports expandable sections, syntax highlighting, streamed completions

### 5. `static/`

* `script.js`: Handles form input, sends messages to FastAPI, receives completions
* `style.css`: Clean, readable chat UI theme

### 6. `chat_log.xhtml`

* Grows with each user/LLM interaction
* Readable in browser (via CSS)
* Parsed to build context for new completions

### 7. `prompts/`

* Modular prompt files:

  * `main.txt`: Primary behavior definition
  * `notebook-edit.txt`: Style and structure for cell edits
  * `log-roll.txt`: Summarization and history compression
* Could be several thousand tokens total

---

## Key Features

* 🚀 Minimal install: Jupyter + `nbscribe`
* 📜 Inspectable conversation logs (append-only XHTML)
* 🧠 LLM-powered notebook edits (via LangChain + OpenAI)
* 🔁 Log rolling: summarize & reset context without losing history
* ✏️ Works on user-authored notebooks
* 🔐 User must explicitly authorize edits
* 🧬 Python-first for researcher adoption

---

## Operational Flow

1. **User starts server**:

   ```bash
   python server.py
   ```

2. **User opens `localhost:5317`**:

   * XHTML log displays
   * Input box sends text to FastAPI

3. **Server builds LLM context**:

   * XHTML log parsed and trimmed
   * Optional notebook summary included
   * Prompt sent to LangChain

4. **LLM reply parsed**:

   * Reply shown in chat
   * Cell changes saved using `nbformat`
   * (Optional) notebook reload triggered

5. **User may request a log roll**:

   * Summarizes history
   * Adds notebook state
   * Clears prior log content

---

## Near-Term Tasks (Cursor Dev Notes)

* [ ] Scaffold `server.py` with FastAPI + static routing
* [ ] Implement first XHTML UI + JS submit handler
* [ ] Add dummy LangChain call returning hardcoded completion
* [ ] Write `notebook_editor.py` with `nbformat` insert
* [ ] Design `chat_log.xhtml` format (roles, time, structure)
* [ ] Add prompt file loader (`prompts/`)
* [ ] Wire it all together end-to-end

---

## Long-Term Considerations

* 🔐 OpenAI key management (from `config.json` or env var)
* 🌐 Deployability via Docker
* 🧩 LangChain agent memory (optional)
* 🧪 Test notebooks with rich output (plots, widgets)
* 🔄 Reversible edits
* 🧬 Later support for Julia via alt prompt sets

---

## License & Repo Goals

* MIT License
* Hosted on GitHub
* Modular design for contributions
* Clear README with usage & architecture

---

## Final Notes

This project is positioned to offer a new type of interaction model for scientists: a trustworthy, local-first assistant that supports visible, auditable workflows rather than black-box automation. It should resonate with both hacker types and educators, and be useful with surprisingly little code.
