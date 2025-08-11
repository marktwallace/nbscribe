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
    { 
      id: crypto.randomUUID(), 
      type: 'code', 
      source: 'print("Hello")', 
      outputs: [], 
      executionCount: null 
    }
  ]);
  const [busy, setBusy] = React.useState(false);
  const [status, setStatus] = React.useState('');
  
  // Get notebook path from global config if available
  const notebookPath = window.NBSCRIBE?.notebookPath;

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
    const s = await mod.startSession({ notebookPath });
    setSession(s);
    return s;
  }

  async function runCell(idx) {
    console.log('NotebookApp: runCell() called', { idx, cellId: cells[idx]?.id });
    
    try {
      // Only set busy state at start - don't clear outputs yet
      setBusy(true); 
      setStatus('Executing...');
      
      // Ensure we have a session
      const s = await ensureSession();
      console.log('NotebookApp: Session ready', s.id);
      
      const { execute } = await import(`/static/notebook/kernelClient.js?v=${VERSION}`);
      
      // Get current value from textarea (might be newer than React state if user hasn't blurred)
      const textareas = document.querySelectorAll('textarea');
      const currentTextarea = textareas[idx]; // Assuming textareas are in same order as cells
      const codeToRun = currentTextarea ? currentTextarea.value : cells[idx].source;
      
      console.log('NotebookApp: Executing code:', codeToRun);
      
      // Clear outputs and collect new ones
      const newOutputs = [];
      
      await execute(s, codeToRun, (msg) => {
        const msgType = msg?.header?.msg_type;
        console.log('NotebookApp: IOPub message:', msgType);
        
        if (msgType && msgType !== 'status') {
          newOutputs.push(msg);
        }
      });
      
      console.log('NotebookApp: Execution complete, updating cell with', newOutputs.length, 'outputs');
      
      // Standard React immutable update pattern with extensive logging
      setCells(prevCells => {
        console.log('setCells FINAL: Before update, cell outputs:', prevCells[idx]?.outputs?.length || 0);
        const newOutputsToSet = newOutputs.length > 0 ? newOutputs : [{ type: 'no-output', content: '(no output)' }];
        
        const nextCells = [...prevCells];
        const oldCell = nextCells[idx];
        nextCells[idx] = {
          ...nextCells[idx],
          outputs: newOutputsToSet
        };
        
        console.log('setCells FINAL: After update, cell outputs:', nextCells[idx].outputs.length);
        console.log('setCells FINAL: Array changed?', nextCells !== prevCells);
        console.log('setCells FINAL: Cell changed?', nextCells[idx] !== oldCell);
        console.log('setCells FINAL: Outputs changed?', nextCells[idx].outputs !== oldCell.outputs);
        console.log('setCells FINAL: Cell ID:', nextCells[idx].id);
        
        return nextCells;
      });
      
    } catch (e) {
      console.error('NotebookApp: Execution failed', e);
      setCells(prevCells => {
        const nextCells = [...prevCells];
        nextCells[idx] = {
          ...nextCells[idx],
          outputs: [{ type: 'error', content: String(e?.message || e) }]
        };
        return nextCells;
      });
    } finally {
      setBusy(false); 
      setStatus('');
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

  function addCell(type = 'code') {
    setCells(prev => [...prev, { 
      id: crypto.randomUUID(), 
      type, 
      source: '', 
      outputs: [], 
      executionCount: null 
    }]);
  }

  function OutputRenderer({ msg }) {
    const msgType = msg?.header?.msg_type;
    console.log('OutputRenderer: Rendering message', { msgType, msgStructure: msg });
    
    if (msg.type === 'no-output') {
      return React.createElement('div', { 
        style: { color: '#6b7280', fontStyle: 'italic' } 
      }, msg.content);
    }
    
    if (msg.type === 'error') {
      return React.createElement('div', { 
        style: { color: '#b91c1c', whiteSpace: 'pre-wrap' } 
      }, msg.content);
    }
    
    if (msgType === 'stream') {
      const text = msg.content?.text ?? '';
      console.log('OutputRenderer: Rendering stream with text:', text);
      return React.createElement('div', { 
        style: { 
          whiteSpace: 'pre-wrap',
          backgroundColor: '#f0f9ff',
          padding: '4px',
          border: '1px solid #ccc',
          marginBottom: '4px'
        } 
      }, text);
    }
    
    if (msgType === 'execute_input') {
      console.log('OutputRenderer: Ignoring execute_input message');
      return null; // Don't render execute_input messages
    }
    
    if (msgType === 'error') {
      const e = msg.content || {};
      const errorText = `${e.ename || 'Error'}: ${e.evalue || ''}\n${(e.traceback || []).join('\n')}`;
      return React.createElement('div', { 
        style: { 
          whiteSpace: 'pre-wrap',
          color: '#b91c1c',
          backgroundColor: '#fef2f2',
          padding: '4px',
          border: '1px solid #fca5a5',
          marginBottom: '4px'
        } 
      }, errorText);
    }
    
    if (msgType === 'execute_result' || msgType === 'display_data') {
      const data = msg.content?.data || {};
      if (data['text/plain']) {
        const text = Array.isArray(data['text/plain']) ? data['text/plain'].join('') : data['text/plain'];
        return React.createElement('div', { 
          style: { 
            whiteSpace: 'pre-wrap',
            padding: '4px',
            border: '1px solid #ccc',
            marginBottom: '4px'
          } 
        }, text);
      }
      if (data['text/html']) {
        const html = Array.isArray(data['text/html']) ? data['text/html'].join('') : data['text/html'];
        return React.createElement('div', { 
          style: { 
            padding: '4px',
            border: '1px solid #ccc',
            marginBottom: '4px'
          },
          dangerouslySetInnerHTML: { __html: html }
        });
      }
    }
    
    console.log('OutputRenderer: Unknown message type, falling back to debug display');
    return React.createElement('div', { 
      style: { color: '#6b7280', fontStyle: 'italic' } 
    }, `[${msgType || 'unknown'}] ${JSON.stringify(msg)}`);
  }

  function updateCell(idx, source) {
    setCells(prev => {
      const next = [...prev];
      next[idx] = { ...next[idx], source };
      return next;
    });
  }

  function CellView({ cell, index }) {
    const isCode = cell.type === 'code';
    console.log('CellView render:', { cellId: cell.id, outputCount: cell.outputs.length, timestamp: Date.now() });
    console.log('CellView outputs detail:', cell.outputs);
    console.log('CellView: Why is this rendering? Cell reference changed?', Math.random());
    

    
    return (
      React.createElement('div', { style: { marginBottom: 16 } },
        React.createElement('div', null,
          React.createElement('select', {
            value: cell.type,
            onChange: (e) => setCells(prev => prev.map((c, i) => i === index ? { ...c, type: e.target.value } : c))
          },
            React.createElement('option', { value: 'code' }, 'Code'),
            React.createElement('option', { value: 'markdown' }, 'Markdown')
          )
        ),
        isCode
          ? React.createElement('textarea', {
              key: cell.id, // Ensure React doesn't reuse across different cells
              defaultValue: cell.source,
              onBlur: (e) => updateCell(index, e.target.value),
              rows: 3,
              style: { width: '100%', fontFamily: 'monospace', fontSize: '14px' },
              placeholder: 'Enter Python code...'
            })
          : React.createElement('textarea', {
              key: cell.id,
              defaultValue: cell.source,
              onBlur: (e) => updateCell(index, e.target.value),
              rows: 4,
              style: { width: '100%' },
              placeholder: 'Enter markdown...'
            }),
        isCode && React.createElement('div', { style: { marginTop: 8 } },
          React.createElement(Button, { onClick: () => runCell(index) }, 'Run')
        ),
        React.createElement('div', { 
          style: { 
            marginTop: 8, 
            minHeight: '20px',
            border: '1px dashed #ccc',
            padding: '4px',
            backgroundColor: '#f9f9f9'
          }
        }, 
          cell.outputs.length > 0 
            ? cell.outputs.map((msg, i) => 
                React.createElement(OutputRenderer, { 
                  key: `${cell.id}-output-${i}`, 
                  msg 
                })
              )
            : '(output will appear here)'
        )
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


