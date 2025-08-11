// CodeMirror 6 code editor for code cells (no line numbers, minimal setup)
// Uses a singleton React instance exposed on window by the ESM entry
const React = window.React;

// IMPORTANT: Do NOT use ?bundle here; bundling each package can duplicate @codemirror/state
// Pin to the same major line so esm.sh resolves a single shared instance
import { EditorState } from 'https://esm.sh/@codemirror/state@6';
import { EditorView, drawSelection, keymap } from 'https://esm.sh/@codemirror/view@6';
import { history, defaultKeymap, historyKeymap } from 'https://esm.sh/@codemirror/commands@6';
import { python } from 'https://esm.sh/@codemirror/lang-python@6';

export default function CodeCellEditor({ value, onReady, readOnly }) {
  const containerRef = React.useRef(null);
  const viewRef = React.useRef(null);

  React.useEffect(() => {

    const state = EditorState.create({
      doc: value || '',
      extensions: [
        keymap.of([...defaultKeymap, ...historyKeymap]),
        history(),
        drawSelection(),
        python(),
        EditorView.editable.of(!readOnly),
        EditorView.theme({
          '&': { border: '1px solid #e5e7eb', borderRadius: '6px' },
          '.cm-content': { fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace', fontSize: '14px', padding: '8px' },
          '.cm-scroller': { overflow: 'auto', maxHeight: '220px' }
        })
      ]
    });

    viewRef.current = new EditorView({ state, parent: containerRef.current });
    try { onReady && onReady(viewRef.current); } catch (_) {}
    return () => {
      try { onReady && onReady(null); } catch (_) {}
      try { viewRef.current && viewRef.current.destroy(); } catch (_) {}
      viewRef.current = null;
    };
  }, []);

  // Apply external value updates without recreating the view
  React.useEffect(() => {
    if (!viewRef.current) return;
    const current = viewRef.current.state.doc.toString();
    if (value != null && value !== current) {
      viewRef.current.dispatch({ changes: { from: 0, to: current.length, insert: value } });
    }
  }, [value]);

  return React.createElement('div', { ref: containerRef });
}


