const VERSION = (window.NBSCRIBE && window.NBSCRIBE.version) || '0.0.0';

// Use the singleton React injected by the ESM entry to avoid duplicate React instances
const React = window.React;
const EditorLazy = React.lazy(() => import(`/static/notebook/CodeCellEditor.js?v=${VERSION}`));

function Button({ onClick, children, disabled }) {
  return (
    React.createElement('button', {
      onClick,
      disabled,
      style: {
        marginRight: '8px',
        padding: '6px 10px',
        borderRadius: 6,
        border: '1px solid #d1d5db',
        background: disabled ? '#f3f4f6' : '#ffffff',
        cursor: disabled ? 'not-allowed' : 'pointer'
      }
    }, children)
  );
}

export default function NotebookApp() {
  const [session, setSession] = React.useState(null);
  const [cells, setCells] = React.useState([
    { id: crypto.randomUUID(), kind: 'code', source: 'print("Hello")', outputs: [], execution_count: null }
  ]);
  // Registry of live editor views keyed by cell ID; avoids React-triggered focus loss
  const editorViewsRef = React.useRef(new Map());
  const [busy, setBusy] = React.useState(false);
  const [status, setStatus] = React.useState('');

  // Pre-warm the kernel/session to avoid a "first click does nothing" experience
  React.useEffect(() => {
    (async () => {
      try {
        setBusy(true); setStatus('Starting kernel...');
        await ensureSession();
      } catch (e) {
        console.error('Session warmup failed', e);
      } finally {
        setBusy(false); setStatus('');
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function ensureSession() {
    if (session) return session;
    const mod = await import(`/static/notebook/kernelClient.js?v=${VERSION}`);
    const s = await mod.startSession();
    setSession(s);
    return s;
  }

  async function runCell(idx) {
    const containerId = `out-${cells[idx].id}`;
    const outEl = document.getElementById(containerId);
    if (outEl) outEl.innerHTML = '';
    try {
      if (!session) {
        setBusy(true); setStatus('Starting kernel...');
      }
      const s = await ensureSession();
      const { execute } = await import(`/static/notebook/kernelClient.js?v=${VERSION}`);
      const { renderIOPub } = await import(`/static/notebook/mime.js?v=${VERSION}`);
      const liveView = editorViewsRef.current.get(cells[idx].id);
      const codeToRun = liveView ? liveView.state.doc.toString() : cells[idx].source;
      let sawMsg = false;
      setBusy(true); setStatus('Executing...');
      await execute(s, codeToRun, (msg) => {
        try { console.debug('IOPub:', msg.header?.msg_type, msg); } catch (_) {}
        if (outEl) {
          renderIOPub(msg, outEl);
          sawMsg = true;
        }
      });
      if (!sawMsg && outEl) {
        const div = document.createElement('div');
        div.style.color = '#6b7280';
        div.textContent = '(no output)';
        outEl.appendChild(div);
      }
    } catch (e) {
      console.error('Run failed', e);
      if (outEl) {
        const div = document.createElement('div');
        div.style.color = '#b91c1c';
        div.style.whiteSpace = 'pre-wrap';
        div.textContent = String(e?.message || e);
        outEl.appendChild(div);
      }
    } finally {
      setBusy(false); setStatus('');
    }
  }

  async function runAll() {
    for (let i = 0; i < cells.length; i++) {
      // eslint-disable-next-line no-await-in-loop
      await runCell(i);
    }
  }

  async function interruptKernel() {
    if (!session) return;
    const { interrupt } = await import(`/static/notebook/kernelClient.js?v=${VERSION}`);
    await interrupt(session);
  }

  async function restartKernel() {
    if (!session) return;
    const { restart } = await import(`/static/notebook/kernelClient.js?v=${VERSION}`);
    await restart(session);
  }

  function addCell(kind = 'code') {
    setCells((prev) => [...prev, { id: crypto.randomUUID(), kind, source: '', outputs: [], execution_count: null }]);
  }

  function updateCell(idx, value) {
    // Update without replacing the array reference if possible to avoid re-render storms
    setCells((prev) => {
      const next = prev.slice();
      next[idx] = { ...next[idx], source: value };
      return next;
    });
  }

  function CellView({ cell, index }) {
    const isCode = cell.kind === 'code';
    return (
      React.createElement('div', { style: { marginBottom: 16 } },
        React.createElement('div', null,
          React.createElement('select', {
            value: cell.kind,
            onChange: (e) => setCells((prev) => prev.map((c, i) => i === index ? { ...c, kind: e.target.value } : c))
          },
            React.createElement('option', { value: 'code' }, 'Code'),
            React.createElement('option', { value: 'markdown' }, 'Markdown')
          )
        ),
        isCode
          ? React.createElement(React.Suspense, { fallback: React.createElement('div', null, 'Loading editor...') },
              React.createElement(EditorLazy, {
                value: cell.source,
                onReady: (view) => {
                  // Keep the editor in control of its own content; only update state on blur
                  if (view) {
                    editorViewsRef.current.set(cell.id, view);
                    view.dom.addEventListener('blur', () => {
                      try {
                        const text = view.state.doc.toString();
                        updateCell(index, text);
                      } catch (_) {}
                    }, { capture: true });
                  } else {
                    editorViewsRef.current.delete(cell.id);
                  }
                }
              })
            )
          : React.createElement('textarea', {
              defaultValue: cell.source,
              onInput: (e) => updateCell(index, e.target.value),
              rows: 4,
              style: { width: '100%' }
            }),
        isCode && React.createElement('div', { style: { marginTop: 8 } },
          React.createElement(Button, { onClick: () => runCell(index) }, 'Run')
        ),
        React.createElement('div', { id: `out-${cell.id}`, style: { marginTop: 8 } })
      )
    );
  }

  return (
    React.createElement('div', { className: 'nb-container' },
      React.createElement('div', { className: 'nb-header' }, 'Thin Notebook UI'),
      status && React.createElement('div', { style: { margin: '8px 0', color: '#6b7280' } }, status),
      React.createElement('div', { style: { marginBottom: 12 } },
        React.createElement(Button, { onClick: runAll, disabled: busy }, 'Run All'),
        React.createElement(Button, { onClick: interruptKernel, disabled: !session || busy }, 'Interrupt'),
        React.createElement(Button, { onClick: restartKernel, disabled: !session || busy }, 'Restart'),
        React.createElement(Button, { onClick: () => addCell('code'), disabled: busy }, 'Add Code'),
        React.createElement(Button, { onClick: () => addCell('markdown'), disabled: busy }, 'Add Markdown')
      ),
      cells.map((cell, i) => React.createElement(CellView, { key: cell.id, cell, index: i }))
    )
  );
}


