# Cursor Prompt — Replace Embedded Jupyter UI with a Thin Client over the Kernel Protocol

**Purpose:** Refactor the current app to **remove embedded Jupyter UI** and replace it with a **minimal custom notebook UI** that speaks the **Jupyter Kernel Messaging Protocol** via our FastAPI **/jupyter/** proxies. Keep the UI small and agent‑friendly; rely on a real CPython kernel for execution so normal plotting and DuckDB work.

---

## Context (read first)

- Backend: FastAPI app (“nbscribe”) that starts a Jupyter server (`NotebookServerManager`) and proxies Jupyter under **`/jupyter/*`**:
  - HTTP: `/jupyter/api/contents`, `/jupyter/api/sessions`, `/jupyter/api/kernels`, etc.
  - WebSocket: `/jupyter/api/kernels/{kernel_id}/channels` (token gets auto-injected by the proxy).
- **Do not** embed Jupyter’s UI. Use a thin client that talks to the kernel via `@jupyterlab/services` over our proxy.
- Execution must be **real CPython** (plots, DuckDB, Pandas). No widgets/comms initially.

---

## Goals

- Frontend: small React component for **cells (code/markdown)**; **run cell / run all / interrupt / restart**; basic outputs (**text/plain, text/html, image/png**).
- Files: **open/save real `.ipynb` (v4)**. Client-side file picker is acceptable for v1.
- Backend: re‑use existing FastAPI **/jupyter/** proxy; no direct calls to the notebook server from the browser in dev.
- Keep the UI simple and agent‑controllable (insert cell / edit / run).

## Non-goals (explicitly out of scope)

- ipywidgets / comms
- Rich MIME beyond text/html/plain + image/png
- Realtime collaboration (CRDT/Yjs)
- Jupyter Contents API syncing/checkpoints (optional later)
- Embedding JupyterLab UI or Lumino widgets

## Tech constraints

- Frontend: React (or existing framework), **`@jupyterlab/services`** for kernel/session mgmt.
- Backend: already present (FastAPI + proxies). Kernel has `ipykernel`, `pandas`, `matplotlib`, `duckdb` installed.
- Security: dev can be tokenless in client; proxy injects token. In prod, require token + CORS allowlist + HTTPS.

---

## Deliverables (files to create)

### 1) `frontend/src/kernelClient.ts`
```ts
import { ServerConnection, SessionManager, Session } from '@jupyterlab/services';

export type KernelSession = Session.ISessionConnection;

export async function startSession(opts?: {
  baseUrl?: string; wsUrl?: string; token?: string;
  kernelName?: string; notebookPath?: string;
}): Promise<KernelSession> {
  const baseUrl = opts?.baseUrl ?? `${window.location.origin}/jupyter/`; // NOTE: trailing slash
  const wsUrl  = opts?.wsUrl  ?? baseUrl.replace(/^http/, 'ws');

  const serverSettings = ServerConnection.makeSettings({
    baseUrl, wsUrl,
    token: opts?.token ?? '',            // proxy injects token server-side in dev
    init:  { credentials: 'omit' }       // no cookies or creds in dev
  });

  const sessions = new SessionManager({ serverSettings });
  await sessions.ready;
  const session = await sessions.startNew({
    name: 'ai-thin-client',
    path: opts?.notebookPath,            // optional: ties kernel to a file
    type: 'notebook',
    kernel: { name: opts?.kernelName ?? 'python3' }
  });
  return session;
}

export async function execute(
  session: KernelSession,
  code: string,
  onIOPub: (msg: any) => void
): Promise<void> {
  const future = session.kernel!.requestExecute({ code, stop_on_error: true }, true);
  future.onIOPub = onIOPub;
  await future.done;
}

export const interrupt = (s: KernelSession) => s.kernel!.interrupt();
export const restart   = (s: KernelSession) => s.kernel!.restart();
export const shutdown  = (s: KernelSession) => s.shutdown();
```

### 2) `frontend/src/mime.ts`
```ts
export function renderIOPub(msg: any, container: HTMLElement) {
  const t = msg.header.msg_type;
  if (t === 'status') return;

  const div = document.createElement('div');
  if (t === 'stream') {
    div.textContent = msg.content.text;
    div.style.whiteSpace = 'pre-wrap';
  } else if (t === 'error') {
    div.textContent = `${msg.content.ename}: ${msg.content.evalue}\n${(msg.content.traceback || []).join('\n')}`;
    div.style.whiteSpace = 'pre-wrap';
    div.style.color = '#b91c1c';
  } else if (t === 'execute_result' || t === 'display_data') {
    const data = msg.content.data || {};
    if (data['image/png']) {
      const img = document.createElement('img');
      img.src = 'data:image/png;base64,' + (Array.isArray(data['image/png']) ? data['image/png'].join('') : data['image/png']);
      img.style.maxWidth = '100%';
      div.appendChild(img);
    } else if (data['text/html']) {
      div.innerHTML = Array.isArray(data['text/html']) ? data['text/html'].join('') : data['text/html'];
    } else if (data['text/plain']) {
      div.textContent = Array.isArray(data['text/plain']) ? data['text/plain'].join('') : data['text/plain'];
      div.style.whiteSpace = 'pre-wrap';
    } else {
      div.textContent = '[unsupported mime]';
    }
  } else {
    return;
  }
  container.appendChild(div);
}
```

### 3) `frontend/src/notebookModel.ts`
```ts
export type Cell = {
  id: string;
  kind: 'code' | 'markdown';
  source: string;
  outputs?: any[];
  execution_count?: number | null;
};

export function toIpynb(cells: Cell[]) {
  return {
    nbformat: 4,
    nbformat_minor: 5,
    metadata: {
      kernelspec: { name: 'python3', display_name: 'Python 3' },
      language_info: { name: 'python' }
    },
    cells: cells.map(c => ({
      cell_type: c.kind,
      source: c.source.split(/(?<=\n)/),
      metadata: {},
      ...(c.kind === 'code'
        ? { outputs: c.outputs ?? [], execution_count: c.execution_count ?? null }
        : {})
    }))
  };
}

export function fromIpynb(nb: any): Cell[] {
  return (nb.cells || []).map((c: any) => ({
    id: crypto.randomUUID(),
    kind: c.cell_type === 'markdown' ? 'markdown' : 'code',
    source: Array.isArray(c.source) ? c.source.join('') : (c.source || ''),
    outputs: c.outputs || [],
    execution_count: c.execution_count ?? null
  }));
}
```

### 4) `frontend/src/Notebook.tsx` (skeleton)
- Render list of cells (code or markdown).
- Editor of your choice (no line numbers). Markdown uses a renderer (marked/remark).
- Buttons: **Run**, **Run All**, **Interrupt**, **Restart**, **Open**, **Save**.
- Wire Run to:
  ```ts
  import { startSession, execute, interrupt, restart } from './kernelClient';
  import { renderIOPub } from './mime';

  // Example:
  const session = await startSession({ /* optionally notebookPath from server */ });
  outEl.innerHTML = '';
  await execute(session, code, msg => renderIOPub(msg, outEl));
  ```

- Open/Save can be client-side (file picker + Blob download) using `fromIpynb` / `toIpynb`.  
  (Optional later: talk to `/jupyter/api/contents/{path}` via your proxy.)

---

## Getting started (deps & scripts)

**Frontend deps**
```bash
npm i @jupyterlab/services @lumino/signaling
# optional: markdown renderer (e.g., marked) and your editor of choice
```

**Backend is already in place.** In dev, run the FastAPI app as you do today; it starts the Jupyter server and exposes `/jupyter/*` proxies (HTTP + WS). The browser client should use:
- `baseUrl = ${window.location.origin}/jupyter/`
- `wsUrl   = baseUrl with http→ws`

No token in dev (proxy injects it). In prod, configure tokens, HTTPS, and CORS allowlist.

---

## Smoke tests (must pass)

Cell A
```python
print("Hello")
```

Cell B
```python
import duckdb, pandas as pd
con = duckdb.connect(':memory:')
df = con.execute('select 1 as a, 2 as b').df()
df
```

Cell C
```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 6.28, 100)
plt.plot(x, np.sin(x)); plt.title('sine'); plt.show()
```

Expect:
- A: stdout stream
- B: DataFrame repr (text/plain or HTML, either is fine)
- C: `image/png` plot

---

## Guardrails (keep scope tight)

- No widgets/comms. No CRDT. No JupyterLab UI.  
- Only handle these MIME types: `stream`, `error`, `text/plain`, `text/html`, `image/png`.
- No line numbers by default.
- Don’t depend on Jupyter Contents API in v1 (client-side open/save is fine).

---

## Acceptance criteria

- Thin client connects to kernel through **our FastAPI `/jupyter/` proxies** using `@jupyterlab/services`.
- Run/Run All emits IOPub that we render correctly.
- **DuckDB** and **matplotlib** work in-kernel.
- Round-trip real `.ipynb` (client-side).
- **Interrupt** and **Restart** buttons function.

---

## Notes specific to this backend

- WebSocket path **must** be `/jupyter/api/kernels/{kernel_id}/channels` (your proxy handles token injection).
- If you serve a notebook page at `/notebook/{notebook_path}`, pass that path down to the frontend and start the session with `{ path: notebookPath, type: 'notebook' }` to associate the kernel with a file (optional but useful).
- Keep all kernel lifecycle logging on the server; the client stays dumb and tiny.

---

**Done means:** running the app yields a minimal notebook UI that executes code, shows text/HTML/PNG outputs, and can open/save `.ipynb`. No Jupyter UI is embedded; all execution is via the kernel protocol over our proxies.
