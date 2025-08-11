# Project Plan — Notebook Editor (FastAPI + Jinja + htmx + CodeMirror + SSE)

## 1) Goals & Non‑Goals

**Goals**

- Provide a reliable, minimal web UI to **view, edit, run, and stream outputs** of Jupyter notebooks using a Python‑first stack.
- Favor **server-rendered HTML** with small, local JS (CodeMirror, optional Alpine) over a SPA.
- Make it easy for an AI assistant to produce **working slices** (endpoint + template) on the first try.

**Non‑Goals (v1)**

- Multi-user collaboration in the same notebook.
- Full parity with JupyterLab (extensions, rich outputs beyond basics, advanced debugging).
- Offline editing / sync conflict resolution.

---

## 2) Architecture Overview

**Backend:** FastAPI + Jinja2 templates; `nbformat` for notebook I/O; `jupyter_client` (or `jupyter_server`) for kernel lifecycle; streaming via SSE (Starlette `StreamingResponse`).

**Frontend:** Server-rendered HTML (+ Tailwind optional). Interactivity via **htmx** for actions (run/save/add/move/delete) and **SSE** for streaming outputs. **CodeMirror 6** powers code cell editing.

**Key Idea:** Each cell renders as a **self-contained fragment**. htmx posts back to endpoints that return updated HTML fragments. Outputs stream via SSE and progressively update the cell's output area.

```
Browser <—HTML (templates/fragments)— FastAPI
   |                                 |
   |—(htmx POSTs)———————> endpoints  |——> kernel exec via jupyter_client
   |<—(HTML fragments)———————        |<—— IOPub messages mapped to SSE
   |—(SSE connect per cell)———> /stream
```

---

## 3) Domain Model

**Notebook**

- `id` (path or UUID), `path`, `kernel_name`
- `cells: List[Cell]`, `metadata`

**Cell**

- `id` (stable UUID), `index` (order), `cell_type: "code" | "markdown"`
- `source: str`, `metadata: dict`
- `outputs: List[Output]` (persisted after run)

**Output** (subset of nbformat)

- `type`: `stream` | `execute_result` | `display_data` | `error`
- `data`: mimebundle | text
- `execution_count?: int`

---

## 4) Routes & Endpoints (v1)

**Notebook lifecycle**

- `GET /nb/{nb_id}` → notebook page (list of rendered cell fragments)
- `POST /nb/{nb_id}/save` → persist notebook to disk
- `POST /nb/create` (optional) → create new notebook

**Cell ops**

- `POST /nb/{nb_id}/cells/add?after={idx}&type={code|markdown}` → returns updated notebook or new cell fragment
- `POST /nb/{nb_id}/cells/{cell_id}/delete`
- `POST /nb/{nb_id}/cells/{cell_id}/move?direction=up|down`
- `POST /nb/{nb_id}/cells/{cell_id}/save` (source comes from form)

**Execution & Streaming**

- `POST /nb/{nb_id}/cells/{cell_id}/run` → starts execution, clears old outputs; returns updated cell fragment (with attached SSE block)
- `GET  /nb/{nb_id}/cells/{cell_id}/stream` (SSE) → emits execution events as they arrive

**Conventions**

- Endpoints return **HTML fragments** for drop-in swaps, plus HTTP 422 for validation.
- SSE events use `event:` types like `output`, `clear`, `done`, `error`.

---

## 5) Templates & Fragments

**Templates**

- `templates/base.html` – layout, htmx + CodeMirror includes
- `templates/notebook.html` – renders notebook header + loops `include '_cell.html'`
- `templates/_cell.html` – a single cell (form + buttons + output region)

**\_cell.html** (shape)

- Container: `<div id="cell-{{ cell.id }}" class="cell card">`
- **Editor**: `<textarea name="source" class="code" data-lang="python">{{ cell.source }}</textarea>`
- **Buttons**
  - Run: `hx-post="/nb/{{nb_id}}/cells/{{cell.id}}/run" hx-target="#cell-{{ cell.id }}" hx-swap="outerHTML"`
  - Save: `hx-post="/nb/{{nb_id}}/cells/{{cell.id}}/save" hx-include="closest .cell"`
  - Move Up/Down, Delete: similar POSTs, swap target is notebook container or cell
- **Output**: `<div id="out-{{ cell.id }}" hx-ext="sse" sse-connect="/nb/{{ nb_id }}/cells/{{ cell.id }}/stream"></div>`

---

## 6) htmx Patterns

- **Fragment swaps**: prefer `hx-swap="outerHTML"` for cell, or `afterbegin/afterend` for add.
- **Progress**: `hx-indicator` spinner per cell.
- **Error handling**: server returns fragment with an inline error panel; use HTTP status codes.
- **OOB updates**: use `hx-swap-oob` for notebook-level status (e.g., kernel busy/idle badge).

---

## 7) CodeMirror Integration (minimal)

- Include CM6 bundles.
- On page load and after any cell swap, **hydrate** any `.code` `<textarea>` into a CodeMirror instance; keep `<textarea>` in DOM so form posts send the current value.
- Choose modes based on `data-lang`. Keyboard shortcuts: `Ctrl/Cmd+Enter` → run cell (dispatch a click on Run button).

---

## 8) Kernel Exec & SSE

**Kernel lifecycle**

- `KernelManager` per notebook; lazy-start on first run; shutdown on inactivity.
- Map nb\_id → kernel + IOPub consumer task.

**Execution**

- On `run`, send cell `source` to kernel, clear outputs, emit `event: clear` via SSE, then forward IOPub messages as SSE `event: output` with a serialized minimal mimebundle.
- Terminate with `event: done` and include `execution_count`.

**SSE Event JSON (examples)**

- `output` (stream): `{ "kind": "stream", "name": "stdout", "text": "line..." }`
- `output` (execute\_result/display\_data): `{ "kind": "display", "mimetype": "text/html", "data": "<pre>...</pre>" }`
- `error`: `{ "ename": "NameError", "evalue": "x is not defined", "traceback": ["..."] }`

**Rendering outputs**

- Text: `<pre>`; HTML: inject as-is **after sanitization**; images: `<img src="data:image/png;base64,...">`.

---

## 9) Persistence & Autosave

- Load notebooks from disk using `nbformat.read`.
- On cell `save` and notebook `save`, update in-memory model and write back with `nbformat.write`.
- Optional: autosave debounce (server-side) – e.g., save 2s after last cell save.
- File locking to avoid concurrent writes.

---

## 10) Security & Safety

- **Sanitize HTML outputs** (e.g., `bleach`) to prevent XSS.
- CSRF protection for POSTs (FastAPI middleware + token in forms).
- Request limits & timeout per execution; kill runaway kernels.
- CORS disabled unless needed; if enabled, restrict origins.
- Consider running kernels in a restricted environment (user, cgroups, or containerized) for prod.

---

## 11) Testing Strategy

- **Unit**: notebook I/O, kernel wrapper, SSE generator.
- **Integration (httpx + pytest)**: run a notebook with a few cells; assert HTML fragments and event sequences (`clear`, `output`, `done`).
- **UI checks (Playwright optional)**: load page, edit, run, see streamed text, add/delete/move cell.
- **Self‑test route (dev only)**: `/dev/smoke` creates a temp notebook with 2 cells and runs them.

---

## 12) Observability

- Structured logs: request id, nb\_id, cell\_id, kernel state, exec time, bytes streamed.
- Counters: runs, failures, timeouts, SSE disconnects.
- Traces around exec → IOPub → SSE pipeline.

---

## 13) Migration Plan (from current codebase)

1. **Inventory**: list current modules (server, Jupyter embed attempts, React assets, routing).
2. **Carve out notebook core**: keep nbformat and kernel logic; drop iframe/Jupyter UI embed.
3. **Introduce templates**: `base.html`, `notebook.html`, `_cell.html`.
4. **Replace React endpoints** with the Route set above (adapter handlers until full switch).
5. **Feature parity path**: load → edit → run → stream → save → add/delete/move.
6. **Delete SPA code** once htmx path covers required flows.

---

## 14) Milestones & Acceptance

**M0 – Skeleton (1 feature slice)**

- Routes: `GET /nb/{id}`, `POST /nb/{id}/cells/{cell_id}/save`, `POST /run`, `GET /stream`.
- One code cell: edit + run + stream text output.
- **Done when**: A sample notebook cell prints lines that appear incrementally.

**M1 – Notebook Basics**

- Add/delete/move cells; markdown rendering; notebook save; kernel lazy start/stop.
- **Done when**: You can create a 3‑cell notebook, rearrange, run, and save to disk.

**M2 – Rich Outputs**

- HTML, images (matplotlib), error tracebacks with folding, execution\_count.
- **Done when**: plot appears via base64 image; errors render safely.

**M3 – UX Polish**

- Keyboard shortcuts; spinners; sticky toolbar; autosave badge; last-run duration.

**M4 – Hardening**

- CSRF, limits/timeouts, sanitized HTML, integration tests, logging.

---

## 15) Conventions for AI‑Generated Changes

- Always deliver **one working slice**: endpoint + template fragment + init JS.
- No new deps unless named and version‑pinned. Prefer stdlib or already‑listed packages.
- Keep **one feature per file** when possible; avoid global state.
- Provide **run instructions** and a **smoke test**.

---

## 16) Open Questions (to resolve as we integrate your code)

- Will kernels run under the same user/process or in per‑notebook sandboxes?
- Do we need markdown preview on edit or only on run?
- Any auth/multi‑user constraints for Karius internal use?
- Preferred CSS (Tailwind vs minimal handcrafted)?

---

## 17) Appendix — Minimal Snippets (for reference only)

**SSE** (shape)

```python
from fastapi import APIRouter
from starlette.responses import StreamingResponse

router = APIRouter()

@router.get("/nb/{nb_id}/cells/{cell_id}/stream")
async def stream(nb_id: str, cell_id: str):
    async def event_source():
        yield "event: clear\n\n"
        async for msg in iopub_messages(nb_id, cell_id):
            data = serialize(msg)
            yield f"event: output\ndata: {data}\n\n"
        yield "event: done\n\n"
    return StreamingResponse(event_source(), media_type="text/event-stream")
```

**Cell Run (handler skeleton)**

```python
@router.post("/nb/{nb_id}/cells/{cell_id}/run")
async def run_cell(nb_id: str, cell_id: str):
    # trigger execution and return updated _cell.html
    return templates.TemplateResponse("_cell.html", context)
```

**htmx Run Button**

```html
<button hx-post="/nb/{{nb_id}}/cells/{{cell.id}}/run"
        hx-target="#cell-{{ cell.id }}" hx-swap="outerHTML">Run</button>
```

**CodeMirror Init (minimal)**

```html
<script>
  function initEditors(scope=document){
    scope.querySelectorAll('textarea.code').forEach(t => {
      if (t._cm) return;
      t._cm = new CodeMirror.fromTextArea(t, { lineNumbers: true, mode: t.dataset.lang || 'python' });
    });
  }
  document.addEventListener('DOMContentLoaded', () => initEditors());
  document.body.addEventListener('htmx:afterSwap', (e) => initEditors(e.target));
</script>
```



---

# Integration Plan Mapped to Your Current Codebase (src/server.py)

Below is a concrete, low‑risk refactor path that **removes the embedded Jupyter UI & proxies from the user path**, keeps your chat features, and moves to **FastAPI + Jinja + htmx + CodeMirror + SSE** with \*\*server‑side kernel execution via \*\*\`\` (no browser→kernel WebSockets).

## A) High‑level decisions

- **Stop using the Jupyter UI** (no iframes). Keep the Jupyter server optional for now, but **do not route user traffic to it**.
- **Persist notebooks with ****\`\`**** directly** (read/write `.ipynb` on disk). Retire the Contents API reads/writes used in `get_notebook_content`/`write_notebook_content`.
- \*\*Execute cells via \*\*\`\` inside FastAPI. Stream outputs to the browser with **SSE**.
- Keep your **chat endpoints** and **directive tool** contract; re‑implement the internals to call the new notebook I/O and cell ops.

## B) New modules (server‑side)

\`\` — notebook file I/O & helpers

```python
from pathlib import Path
import nbformat, uuid

def read_nb(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    ensure_cell_ids(nb)
    return nb

def write_nb(path: str, nb: dict) -> None:
    ensure_cell_ids(nb)
    with open(path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

def ensure_cell_ids(nb: dict) -> None:
    for c in nb.get('cells', []):
        cid = c.get('id') or c.get('metadata', {}).get('id')
        if not cid:
            cid = uuid.uuid4().hex[:8]
            c['id'] = cid
        c.setdefault('metadata', {}).setdefault('id', c['id'])

def find_cell_index(nb: dict, cell_id: str) -> int | None:
    for i, c in enumerate(nb.get('cells', [])):
        if c.get('id') == cell_id or c.get('metadata', {}).get('id') == cell_id:
            return i
    return None
```

\`\` — per‑notebook kernel manager + execution stream

```python
import asyncio, json
from typing import AsyncIterator
from jupyter_client import AsyncKernelManager

class KernelPool:
    def __init__(self):
        self._km_by_nb: dict[str, AsyncKernelManager] = {}

    async def get(self, nb_path: str) -> AsyncKernelManager:
        km = self._km_by_nb.get(nb_path)
        if km is None or not await km.is_alive():
            km = AsyncKernelManager(kernel_name='python3')
            await km.start_kernel()
            self._km_by_nb[nb_path] = km
        return km

    async def execute_stream(self, nb_path: str, code: str) -> AsyncIterator[str]:
        km = await self.get(nb_path)
        kc = await km.client()
        await kc.start_channels()
        msg_id = kc.execute(code, store_history=True)
        yield "event: clear

"
        try:
            while True:
                msg = await kc.get_iopub_msg(timeout=60)
                t = msg['header']['msg_type']
                if t == 'status' and msg['content'].get('execution_state') == 'idle':
                    break
                if t in ('stream','error','execute_result','display_data'):
                    yield f"event: output
data: {json.dumps(msg['content'], default=str)}

"
        finally:
            yield "event: done

"

    async def shutdown_idle(self):
        # optional: sweep idle kernels
        pass
```

> Keeps all WebSocket complexity server‑side; the browser only listens to SSE.

## C) Routes to add/replace (FastAPI)

Add a new router file (e.g., `src/notebook_routes.py`) and include it in `create_app()`.

```python
from fastapi import APIRouter, Request, HTTPException
from starlette.responses import StreamingResponse
from .nbio import read_nb, write_nb, find_cell_index
from .kernel import KernelPool

router = APIRouter()
kernels = KernelPool()

@router.get('/nb/{path:path}')
async def view_nb(request: Request, path: str):
    nb = read_nb(path)
    return request.app.state.templates.TemplateResponse('notebook.html', {
        'request': request,
        'nb_path': path,
        'cells': nb.get('cells', [])
    })

@router.post('/nb/{path:path}/cells/{cell_id}/save')
async def save_cell(request: Request, path: str, cell_id: str):
    form = await request.form()
    source = form.get('source','')
    nb = read_nb(path)
    i = find_cell_index(nb, cell_id)
    if i is None: raise HTTPException(404, 'cell not found')
    nb['cells'][i]['source'] = source.split('
')
    if nb['cells'][i]['cell_type'] == 'code':
        nb['cells'][i]['outputs'] = []; nb['cells'][i]['execution_count'] = None
    write_nb(path, nb)
    return request.app.state.templates.TemplateResponse('_cell.html', {
        'request': request, 'nb_path': path, 'cell': nb['cells'][i]
    })

@router.post('/nb/{path:path}/cells/{cell_id}/run')
async def run_cell(request: Request, path: str, cell_id: str):
    nb = read_nb(path)
    i = find_cell_index(nb, cell_id)
    if i is None: raise HTTPException(404, 'cell not found')
    # Return fresh cell fragment with SSE hook attached
    return request.app.state.templates.TemplateResponse('_cell.html', {
        'request': request, 'nb_path': path, 'cell': nb['cells'][i]
    })

@router.get('/nb/{path:path}/cells/{cell_id}/stream')
async def stream_cell(path: str, cell_id: str):
    nb = read_nb(path)
    i = find_cell_index(nb, cell_id)
    if i is None: raise HTTPException(404, 'cell not found')
    code = ''.join(nb['cells'][i]['source']) if nb['cells'][i]['cell_type']=='code' else ''
    async def gen():
        async for chunk in kernels.execute_stream(path, code):
            yield chunk
    return StreamingResponse(gen(), media_type='text/event-stream')
```

Also add simple **add/delete/move** endpoints mirroring the plan’s Section 4.

## D) Templates (Jinja) & htmx wiring

\`\`: load htmx, CodeMirror, and a small init script. Add `hx-boost="true"` if you want boosted links.

\`\`

```html
{% extends 'base.html' %}
{% block content %}
  <h1 class="text-xl">Notebook: {{ nb_path }}</h1>
  <div id="nb">
    {% for cell in cells %}
      {% include '_cell.html' %}
    {% endfor %}
  </div>
{% endblock %}
```

\`\`

```html
<div id="cell-{{ cell.id }}" class="cell card">
  {% if cell.cell_type == 'code' %}
  <form hx-post="/nb/{{ nb_path }}/cells/{{ cell.id }}/save"
        hx-target="#cell-{{ cell.id }}" hx-swap="outerHTML">
    <textarea name="source" class="code" data-lang="python">{{ (cell.source|join('')) | e }}</textarea>
    <div class="toolbar">
      <button type="submit">Save</button>
      <button type="button"
        hx-post="/nb/{{ nb_path }}/cells/{{ cell.id }}/run"
        hx-target="#cell-{{ cell.id }}" hx-swap="outerHTML">Run</button>
    </div>
  </form>
  <div id="out-{{ cell.id }}" hx-ext="sse"
       sse-connect="/nb/{{ nb_path }}/cells/{{ cell.id }}/stream"></div>
  {% else %}
    <textarea name="source" class="md">{{ (cell.source|join('')) | e }}</textarea>
  {% endif %}
</div>
```

**CodeMirror bootstrap** (add to `base.html`)

```html
<script>
  function initEditors(scope=document){
    scope.querySelectorAll('textarea.code').forEach(t=>{
      if(t._cm) return;
      t._cm = CodeMirror.fromTextArea(t, { lineNumbers: true, mode: t.dataset.lang||'python' });
    });
  }
  document.addEventListener('DOMContentLoaded', ()=>initEditors());
  document.body.addEventListener('htmx:afterSwap', e=>initEditors(e.target));
</script>
```

## E) Map from your current `src/server.py` to new approach

- **Keep**: chat endpoints (`/api/chat`, `/api/chat/stream`), session helpers, conversation logging, `/`, `/new`, and notebook picker.
- **Replace** `/notebook/{notebook_path}`: serve \`\` (no `notebook_iframe_url`).
- **Delete/Disable for v1**: all \`\`\*\* HTTP & WS proxy routes\*\*, `KernelProtection`, and repetitive polling filters. (Optional: keep the manager but stop mounting routes.)
- **Replace** `get_notebook_content`, `write_notebook_content`, `trigger_notebook_refresh` with `nbio.read_nb` / `nbio.write_nb`.
- **Directive tool** (`/api/directives/approve`): call into `nbio` helpers for insert/edit/delete, and return success; (optional) also render an updated cell fragment to push via htmx oob.

## F) M0 (one‑day slice)

- Routes: `GET /nb/{path}`, `POST /nb/{path}/cells/{id}/save`, `POST /nb/{path}/cells/{id}/run`, `GET /nb/{path}/cells/{id}/stream`.
- Templates: `base.html`, `notebook.html`, `_cell.html` with CodeMirror init.
- Kernel: `KernelPool.execute_stream()` supporting `stream`, `error`, `execute_result`, `display_data`.
- **Done when**: edit a single code cell → **Save** → **Run** streams text output; simple matplotlib plot shows as `image/png`.

## G) M1–M3 (fit to your backlog)

- **M1**: add/delete/move cells; Markdown render (server‑side `markdown2` or client‑side); notebook‑level Save.
- **M2**: keyboard shortcuts (Ctrl/Cmd+Enter), spinners (`hx-indicator`), autosave badge, execution\_count.
- **M3**: security hardening (CSRF token in forms), HTML sanitization for outputs, timeout/kill.

## H) Measurement plan (for internal Karius evidence)

Track for 10 “feature slices” built with AI assistance:

- **First‑run pass rate** (renders & runs with zero edits),
- **Iteration count to green** (edits needed),
- **Wall‑clock to green**. Compare **React  + @jupyterlab/services** vs **Jinja + htmx + SSE** slices of equal scope (e.g., “add cell”, “run cell”, “render plot”).

---



---

# Patch Set — FastAPI + Jinja + htmx + SSE (M0)

> This patch set implements the **M0 slice** from the plan: open a notebook, edit a single code cell, click **Run**, and stream outputs (text, PNG, HTML) via **SSE**. It avoids the Jupyter UI and does not require client→kernel websockets.

## 0) Requirements (new/confirmed)

Add (or pin) these to `requirements.txt` / `pyproject.toml`:

```
nbformat>=5.9,<6
jupyter_client>=8.6,<9
ipykernel>=6.29,<7
bleach>=6.1,<7   # for safe HTML output sanitization
```

---

## 1) New file: `src/nbio.py`

```python
from pathlib import Path
import uuid
import nbformat

__all__ = [
    "read_nb",
    "write_nb",
    "ensure_cell_ids",
    "find_cell_index",
]


def read_nb(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Notebook not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    ensure_cell_ids(nb)
    return nb


def write_nb(path: str, nb: dict) -> None:
    ensure_cell_ids(nb)
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)


def ensure_cell_ids(nb: dict) -> None:
    for c in nb.get("cells", []):
        cid = c.get("id") or c.get("metadata", {}).get("id")
        if not cid:
            cid = uuid.uuid4().hex[:8]
            c["id"] = cid
        c.setdefault("metadata", {}).setdefault("id", c["id"])


def find_cell_index(nb: dict, cell_id: str) -> int | None:
    for i, c in enumerate(nb.get("cells", [])):
        if c.get("id") == cell_id or c.get("metadata", {}).get("id") == cell_id:
            return i
    return None
```

---

## 2) New file: `src/kernel.py`

```python
import json
from typing import AsyncIterator, Optional

from jupyter_client import AsyncKernelManager

try:
    import bleach  # type: ignore
except Exception:  # pragma: no cover
    bleach = None  # sanitize becomes a no-op


class KernelPool:
    """Simple per-notebook kernel pool.
    One kernel per notebook path, lazy-started on first execution.
    """

    def __init__(self) -> None:
        self._by_nb: dict[str, AsyncKernelManager] = {}

    async def _get_or_start(self, nb_path: str) -> AsyncKernelManager:
        km = self._by_nb.get(nb_path)
        if km is None:
            km = AsyncKernelManager(kernel_name="python3")
            await km.start_kernel()
            self._by_nb[nb_path] = km
        return km

    async def execute_stream(self, nb_path: str, code: str) -> AsyncIterator[str]:
        """Yield SSE-formatted events while executing `code` in the kernel for `nb_path`.
        Produces `event: output` with pre-rendered HTML chunks, then `event: done`.
        """
        km = await self._get_or_start(nb_path)
        kc = await km.client()
        await kc.start_channels()

        # Start execution
        kc.execute(code, store_history=True)

        # helper to emit a single SSE event with possibly multi-line data
        def sse(event: str, data: str) -> str:
            lines = data.splitlines() or [""]
            return "".join([f"event: {event}
", *[f"data: {ln}
" for ln in lines], "
"])  # end of message

        # Render IOPub -> HTML fragments
        def render_html(msg_content: dict) -> Optional[str]:
            # stream
            if "text" in msg_content and "name" in msg_content:
                name = msg_content.get("name") or "stdout"
                text = msg_content.get("text", "")
                cls = "stderr" if name == "stderr" else "stdout"
                return f'<pre class="stream {cls}">{_html(text)}</pre>'
            # error
            if {"ename", "evalue"}.issubset(msg_content.keys()):
                tb = "
".join(msg_content.get("traceback", []) or [])
                e = _html(f"{msg_content.get('ename')}: {msg_content.get('evalue')}
{tb}")
                return f'<pre class="error">{e}</pre>'
            # display/execute_result
            data = msg_content.get("data")
            if isinstance(data, dict):
                if data.get("image/png"):
                    b64 = _join(data["image/png"])  # list or str
                    return f'<img alt="png" src="data:image/png;base64,{b64}" />'
                if data.get("text/html"):
                    html = _join(data["text/html"])  # may be list
                    return _sanitize(html)
                if data.get("text/plain"):
                    txt = _join(data["text/plain"])  # may be list
                    return f"<pre>{_html(txt)}</pre>"
            return None

        # Pull IOPub until we see idle
        while True:
            msg = await kc.get_iopub_msg(timeout=60)
            t = msg["header"]["msg_type"]
            if t == "status" and msg["content"].get("execution_state") == "idle":
                break
            if t in ("stream", "error", "display_data", "execute_result"):
                html = render_html(msg["content"])
                if html:
                    yield sse("output", html)

        yield sse("done", "")


def _join(v: object) -> str:
    if isinstance(v, list):
        return "".join(str(x) for x in v)
    return str(v)


def _html(s: str) -> str:
    # minimal escape (we do stronger sanitize when rendering HTML)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _sanitize(html: str) -> str:
    if not bleach:
        return html
    return bleach.clean(
        html,
        tags=[
            "div",
            "span",
            "pre",
            "code",
            "img",
            "svg",
            "p",
            "em",
            "strong",
            "table",
            "thead",
            "tbody",
            "tr",
            "td",
            "th",
            "ul",
            "ol",
            "li",
            "br",
            "hr",
            "a",
        ],
        attributes={"img": ["src", "alt", "title", "width", "height"], "a": ["href", "title"], "*": ["class"]},
        protocols=["http", "https", "data"],
        strip=True,
    )
```

---

## 3) New file: `src/notebook_routes.py`

```python
from fastapi import APIRouter, Request, HTTPException
from starlette.responses import StreamingResponse

from .nbio import read_nb, write_nb, find_cell_index
from .kernel import KernelPool

router = APIRouter()
kernels = KernelPool()


@router.get("/nb/{path:path}")
async def view_nb(request: Request, path: str):
    nb = read_nb(path)
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "notebook.html",
        {"request": request, "nb_path": path, "cells": nb.get("cells", [])},
    )


@router.post("/nb/{path:path}/cells/{cell_id}/save")
async def save_cell(request: Request, path: str, cell_id: str):
    form = await request.form()
    source = form.get("source", "")
    nb = read_nb(path)
    i = find_cell_index(nb, cell_id)
    if i is None:
        raise HTTPException(404, "cell not found")
    nb["cells"][i]["source"] = source.split("
")
    if nb["cells"][i]["cell_type"] == "code":
        nb["cells"][i]["outputs"] = []
        nb["cells"][i]["execution_count"] = None
    write_nb(path, nb)
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "_cell.html",
        {"request": request, "nb_path": path, "cell": nb["cells"][i], "run_now": False},
    )


@router.post("/nb/{path:path}/cells/{cell_id}/run")
async def run_cell(request: Request, path: str, cell_id: str):
    nb = read_nb(path)
    i = find_cell_index(nb, cell_id)
    if i is None:
        raise HTTPException(404, "cell not found")
    # Return fresh cell fragment with SSE connection activated
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "_cell.html",
        {"request": request, "nb_path": path, "cell": nb["cells"][i], "run_now": True},
    )


@router.get("/nb/{path:path}/cells/{cell_id}/stream")
async def stream_cell(path: str, cell_id: str):
    nb = read_nb(path)
    i = find_cell_index(nb, cell_id)
    if i is None:
        raise HTTPException(404, "cell not found")
    cell = nb["cells"][i]
    if cell.get("cell_type") != "code":
        raise HTTPException(400, "only code cells are executable")
    code = "".join(cell.get("source", []))

    async def gen():
        async for chunk in kernels.execute_stream(path, code):
            yield chunk

    return StreamingResponse(gen(), media_type="text/event-stream")
```

---

## 4) Modify `src/server.py` (minimal, non-destructive)

**Add imports & router include, and stash **``** on **``** so routes can use it.**

```diff
@@ def create_app():
-    templates = Jinja2Templates(directory="templates")
+    templates = Jinja2Templates(directory="templates")
+    # expose for routers
+    app.state.templates = templates
+
+    # Notebook UI (htmx/Jinja) routes
+    try:
+        from src import notebook_routes
+        app.include_router(notebook_routes.router)
+    except Exception as e:
+        logging.error(f"Failed to include notebook routes: {e}")
```

> Optional (recommended soon after M0): comment out or remove the `/jupyter/*` proxy routes and the websocket proxies to reduce complexity/noise. They are no longer needed for the new UI path.

---

## 5) New templates

### `templates/base.html`

```html
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{{ title or "nbscribe" }}</title>
    <!-- htmx + SSE ext -->
    <script src="https://unpkg.com/htmx.org@1.9.12"></script>
    <script src="https://unpkg.com/htmx.org@1.9.12/dist/ext/sse.js"></script>
    <!-- CodeMirror 5 (simple, reliable for M0) -->
    <link rel="stylesheet" href="https://unpkg.com/codemirror@5/lib/codemirror.css" />
    <script src="https://unpkg.com/codemirror@5/lib/codemirror.js"></script>
    <script src="https://unpkg.com/codemirror@5/mode/python/python.js"></script>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 1rem; }
      .cell { border: 1px solid #e5e7eb; border-radius: 0.5rem; padding: 0.75rem; margin: 0.75rem 0; }
      .toolbar { display: flex; gap: 0.5rem; margin: 0.5rem 0; }
      .toolbar button { padding: 0.35rem 0.75rem; border: 1px solid #d1d5db; border-radius: 0.375rem; background: #f9fafb; cursor: pointer; }
      .stream.stdout { color: #111827; white-space: pre-wrap; }
      .stream.stderr { color: #991b1b; white-space: pre-wrap; }
      .error { color: #b91c1c; white-space: pre-wrap; }
      .cm-editor, .CodeMirror { border: 1px solid #e5e7eb; border-radius: 0.375rem; }
    </style>
  </head>
  <body>
    {% block content %}{% endblock %}

    <script>
      function initEditors(scope=document){
        scope.querySelectorAll('textarea.code').forEach(t=>{
          if (t._cm) return;
          t._cm = CodeMirror.fromTextArea(t, { lineNumbers: true, mode: t.dataset.lang||'python' });
          // Ctrl/Cmd+Enter runs the nearest cell
          t._cm.addKeyMap({
            'Cmd-Enter': ()=>runNearest(t),
            'Ctrl-Enter': ()=>runNearest(t),
          });
        });
      }
      function runNearest(textarea){
        const cell = textarea.closest('.cell');
        const runBtn = cell && cell.querySelector('[data-run]');
        if (runBtn) runBtn.click();
      }
      document.addEventListener('DOMContentLoaded', ()=> initEditors());
      document.body.addEventListener('htmx:afterSwap', (e)=> initEditors(e.target));
    </script>
  </body>
</html>
```

### `templates/notebook.html`

```html
{% extends 'base.html' %}
{% block content %}
  <h1 style="margin:0 0 1rem 0;">Notebook: {{ nb_path }}</h1>
  <div id="nb">
    {% for cell in cells %}
      {% include '_cell.html' %}
    {% endfor %}
  </div>
{% endblock %}
```

### `templates/_cell.html`

```html
<div id="cell-{{ cell.id }}" class="cell">
  {% if cell.cell_type == 'code' %}
  <form hx-post="/nb/{{ nb_path }}/cells/{{ cell.id }}/save"
        hx-target="#cell-{{ cell.id }}" hx-swap="outerHTML">
    <textarea name="source" class="code" data-lang="python">{{ (cell.source|join('')) | e }}</textarea>
    <div class="toolbar">
      <button type="submit">Save</button>
      <button type="button" data-run
              hx-post="/nb/{{ nb_path }}/cells/{{ cell.id }}/run"
              hx-target="#cell-{{ cell.id }}" hx-swap="outerHTML">Run</button>
    </div>
  </form>
  <div id="out-{{ cell.id }}"
       {% if run_now %} hx-ext="sse" sse-connect="/nb/{{ nb_path }}/cells/{{ cell.id }}/stream" sse-swap="output" hx-swap="beforeend" {% endif %}
       ></div>
  {% else %}
    <form hx-post="/nb/{{ nb_path }}/cells/{{ cell.id }}/save"
          hx-target="#cell-{{ cell.id }}" hx-swap="outerHTML">
      <textarea name="source" class="md">{{ (cell.source|join('')) | e }}</textarea>
      <div class="toolbar">
        <button type="submit">Save</button>
      </div>
    </form>
  {% endif %}
</div>
```

> Note: The SSE connection is only attached when `run_now` is true (after clicking **Run**). That prevents auto-executing on initial render.

---

## 6) (Optional) Update notebook picker to use the new route

If your `templates/notebook_picker.html` previously linked to `/notebook/{path}`, update links to `/nb/{{ item.path }}`. Example snippet:

```html
<ul>
  {% for n in recent_notebooks %}
    <li><a href="/nb/{{ n.path }}">{{ n.name }}</a> <small>({{ n.modified }})</small></li>
  {% endfor %}
</ul>
```

---

## 7) Run instructions

1. Install new deps (from project root):
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure you have at least one `.ipynb` in the working directory (or create one via your existing `/api/notebooks/create`).
3. Start the app as usual:
   ```bash
   python -m src.server  # or however you start it today
   ```
4. Open a notebook via the new route, e.g.:
   - `http://localhost:5317/nb/YourNotebook.ipynb`
5. Edit the first code cell, click **Save**, then **Run**. You should see output streaming and images for matplotlib.

---

## 8) Next small steps (post‑M0)

- Add **add/delete/move** cell endpoints + buttons.
- Show `execution_count` and last-run duration in the cell header.
- Add a small error panel fragment for exceptions (already streamed as `<pre class="error">…</pre>`).
- Wire the **directive tool** to return updated cell fragments (htmx OOB swap) after operations.

---

