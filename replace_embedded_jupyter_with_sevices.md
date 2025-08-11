## Mini-project plan: Replace embedded Jupyter UI with a thin React client over Jupyter Services

Goal: Remove any embedded Jupyter UI and add a minimal, agent-friendly notebook page that talks to a real CPython kernel via our existing FastAPI `/jupyter/*` proxies using `@jupyterlab/services`. Keep all styling in `static/`, avoid user home `~/.jupyter` settings, and ensure cache-busting by bumping version strings on JS edits.

Important preferences to honor:
- Keep the UI simple; no JupyterLab, no Files tab, no widgets/comms.
- Use Notebook 7 kernel protocol via `@jupyterlab/services`; rely on our proxy routes.
- Place all CSS in `static/`; no inline styles beyond tiny, contextual bits.
- Do not auto-run the server from tools; ask the user to test manually.
- After any JS change, bump a `?v=...` query string to avoid cache issues.

### Phase 0 — Baseline verification (no code changes)
- Confirm the backend already exposes the proxies (these should exist per current design):
  - `GET /jupyter/api/kernels` (HTTP)
  - `WS /jupyter/api/kernels/{kernel_id}/channels` (WebSocket)
- Open the app and verify a Jupyter server is launched in logs (from `NotebookServerManager`).

Test as you go
- With the server running as you normally do, hit `/jupyter/api/kernels` in the browser; you should see JSON (likely empty list if no sessions yet).

### Phase 1 — New notebook route and template
Files to add/modify
- Add `templates/notebook.html`
- Modify `src/server.py` to add a route `GET /notebook` that renders `templates/notebook.html`

Template contents (high level)
- A minimal HTML skeleton that loads the notebook UI assets from `static/notebook/` with explicit version query strings, e.g. `/static/notebook/notebook.js?v=0.1.0` and `/static/notebook/notebook.css?v=0.1.0`.
- No embedded Jupyter UI. Provide a single root div for the React app.

Test as you go
- Start the server, open `/notebook`. You should see a blank page with a placeholder header.

### Phase 2 — Thin client scaffolding (no-build, ES modules)
Files to add
- Create a new folder: `static/notebook/`
  - `static/notebook/notebook.css` — styles for the notebook page
  - `static/notebook/notebook.js` — bootstraps the app and mounts the React component
  - `static/notebook/NotebookApp.js` — the main React component (cells UI + controls)
  - `static/notebook/kernelClient.js` — session and execution utilities using `@jupyterlab/services`
  - `static/notebook/mime.js` — render IOPub messages (`stream`, `error`, `text/plain`, `text/html`, `image/png`)
  - `static/notebook/notebookModel.js` — in-memory model and `toIpynb`/`fromIpynb` helpers for `.ipynb`

Notes
- Use a lightweight, build-less setup: import React and ReactDOM from ESM CDNs (e.g., `esm.sh` or `jspm.io`), and import `@jupyterlab/services` as an ESM module the same way. This avoids a bundler in v1 and keeps assets simple in `static/`.
- If CDN reliability is a concern later, we can pin and vendor the ESM bundles into `static/notebook/vendor/` without changing the API.

Test as you go
- Refresh `/notebook` and confirm the React app mounts and shows a simple UI shell (buttons: Run, Run All, Interrupt, Restart, Open, Save).
- If assets don’t update, bump the `?v=` in `templates/notebook.html`.

### Phase 3 — Kernel session management
`static/notebook/kernelClient.js`
- Implement `startSession({ baseUrl, wsUrl, token, kernelName, notebookPath })` using `@jupyterlab/services` `ServerConnection` + `SessionManager`.
- Default `baseUrl` to `${window.location.origin}/jupyter/` and set `wsUrl` to `baseUrl` with `http→ws`.
- Implement `execute(session, code, onIOPub)` wiring `future.onIOPub = onIOPub` and awaiting `future.done`.
- Implement `interrupt`, `restart`, `shutdown` proxies to kernel.

Test as you go
- Add a minimal text area and a Run button; on click, start a session, run `print("Hello")`, and render IOPub to a debug panel (plain text) to confirm connectivity.

### Phase 4 — IOPub rendering for basic MIME types
`static/notebook/mime.js`
- Implement `renderIOPub(msg, container)` handling:
  - status: ignore
  - stream: show `content.text` (pre-wrap)
  - error: show `ename`, `evalue`, and `traceback` (pre-wrap, red)
  - execute_result / display_data: handle `image/png`, `text/html`, `text/plain` in that order

Integrate in `NotebookApp.js`
- For a code cell’s latest run, clear an outputs container and call `renderIOPub` for each incoming message.

Test as you go
- Run three cells manually with:
  1) `print("Hello")`
  2) `import duckdb, pandas as pd; con = duckdb.connect(':memory:'); con.execute('select 1 as a').df()`
  3) `%matplotlib inline` + a simple `matplotlib` plot producing `image/png`
- Confirm outputs render correctly.

### Phase 5 — Minimal notebook model and UI
`static/notebook/notebookModel.js`
- Define `{ id, kind: 'code'|'markdown', source, outputs?, execution_count? }`.
- Implement `toIpynb(cells)` and `fromIpynb(nb)` for v4 notebooks.

`static/notebook/NotebookApp.js`
- Maintain a list of cells in component state.
- Render cells with a simple editor (a `<textarea>` is acceptable for v1). No line numbers.
- Buttons: Run (current cell), Run All, Interrupt, Restart, Open, Save.
- Run/Run All:
  - Ensure session exists (start one if needed)
  - On run, clear outputs for the cell and stream IOPub via `renderIOPub` into the cell’s output container.
  - Track `execution_count` if present.
- Open/Save:
  - Open: file picker, parse JSON, `fromIpynb` → set state.
  - Save: serialize via `toIpynb` and trigger a Blob download as `.ipynb`.

Test as you go
- Create two code cells and one markdown cell; run each code cell and verify outputs are isolated per cell.
- Save to `.ipynb`, then reload the page and open the saved file; ensure the same structure appears.

### Phase 6 — Interrupt and Restart controls
`static/notebook/NotebookApp.js`
- Wire Interrupt and Restart buttons to `kernelClient.interrupt` and `kernelClient.restart`.
- Disable buttons if no active session.

Test as you go
- Run a long-running cell (e.g., `import time; time.sleep(30)`), click Interrupt and confirm the kernel stops execution.
- Click Restart, confirm a new execution count begins at 1 and state resets appropriately.

### Phase 7 — Styling and accessibility
`static/notebook/notebook.css`
- Provide clear typography, simple spacing, and responsive layout.
- No inline styles except tiny bits where strictly necessary.
- Ensure focus rings and keyboard navigation for buttons and editors.

Test as you go
- Verify the layout on narrow and wide screens.

### Phase 8 — Cache-busting and versioning
- In `templates/notebook.html`, append explicit `?v=0.1.0` (or date stamp) to every JS and CSS asset, and bump it after each JS edit as a rule of thumb.
- If serving via a reverse proxy or CDN, confirm no aggressive caching overrides query strings.

### Phase 9 — (Optional) Associate kernel with a path
- If we serve `/notebook/{path}`, pass `notebookPath` down via a small inline script `data-*` attribute and initialize the session with `{ path, type: 'notebook' }` to associate the kernel with a file.

### Phase 10 — Acceptance tests (final pass)
- Thin client connects to kernel through `/jupyter/` proxies using `@jupyterlab/services`.
- Run/Run All renders IOPub:
  - `stream` shows stdout
  - `error` shows tracebacks
  - `text/plain`, `text/html` render
  - `image/png` renders plots
- DuckDB and matplotlib work in-kernel (per Phase 4 tests).
- Interrupt and Restart work.
- Round-trip a real `.ipynb` via Open/Save.

### Concrete edit checklist (in order)
1) Add `templates/notebook.html` with root div and versioned links to:
   - `/static/notebook/notebook.css?v=0.1.0`
   - `/static/notebook/notebook.js?v=0.1.0`
2) Add route in `src/server.py`:
   - `GET /notebook` → render `templates/notebook.html`
   - Optionally `GET /notebook/{path:path}` → pass `path` to the template for session association (Phase 9)
3) Add `static/notebook/` files:
   - `notebook.css`
   - `notebook.js` (bootstraps app, imports React/ReactDOM ESM, and `NotebookApp.js`)
   - `NotebookApp.js`
   - `kernelClient.js`
   - `mime.js`
   - `notebookModel.js`
4) Implement Phase 3 (session) and Phase 4 (IOPub) utilities; smoke-test execution with a single code area.
5) Implement Phase 5 UI for multiple cells with Open/Save; test round-trip.
6) Implement Phase 6 controls for Interrupt/Restart.
7) Polish Phase 7 styling; test responsiveness and a11y basics.
8) Enforce Phase 8 cache-busting discipline; bump `?v=` whenever JS changes.
9) Optional Phase 9 path association and route.
10) Run the acceptance tests (Phase 10).

### Manual test protocol after each step
- Always run the server the same way you do today and test in the browser; do not rely on tool-run servers.
- Check server stdout/stderr logs for kernel and proxy health.
- When JS changes do not appear, bump the version query string in `templates/notebook.html`.

### Out of scope (for clarity)
- No Jupyter UI embedding or Lab/Lumino widgets.
- No ipywidgets/comms or realtime collaboration.
- No heavy bundling step for v1; if we later need a bundler, we will vendor ESM dependencies first and adopt the minimal tool that fits.


