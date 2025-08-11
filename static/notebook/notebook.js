// ESM entry: import React/ReactDOM from CDN and mount NotebookApp
const VERSION = (window.NBSCRIBE && window.NBSCRIBE.version) || '0.0.0';

const reactUrl = 'https://esm.sh/react@18?dev';
const reactDomUrl = 'https://esm.sh/react-dom@18/client?dev';

async function bootstrap() {
  const root = document.getElementById('notebook-root');
  if (!root) return;

  const ReactModule = await import(reactUrl);
  const React = ReactModule.default || ReactModule;
  const ReactDOM = await import(reactDomUrl);
  // Expose a single React instance globally to avoid multiple copies
  window.React = React;
  const { default: NotebookApp } = await import(`/static/notebook/NotebookApp.js?v=${VERSION}`);

  const rootEl = ReactDOM.createRoot(root);
  rootEl.render(React.createElement(NotebookApp));
}

bootstrap().catch((err) => {
  console.error('Notebook bootstrap failed:', err);
  const root = document.getElementById('notebook-root');
  if (root) {
    const div = document.createElement('div');
    div.style.color = '#b91c1c';
    div.textContent = 'Failed to load notebook UI. Check console logs.';
    root.appendChild(div);
  }
});


