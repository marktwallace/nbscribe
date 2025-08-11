// Minimal Jupyter services client via ESM import from CDN (bundle all deps)
// Pin to a recent stable v7.x to avoid bitrot while staying current
import * as jls from 'https://esm.sh/@jupyterlab/services@7.4.5?bundle';

export async function startSession(opts = {}) {
  const baseUrl = opts.baseUrl ?? `${window.location.origin}/jupyter/`;
  const wsUrl = opts.wsUrl ?? baseUrl.replace(/^http/, 'ws');
  // Don't pass token - our proxy handles authentication automatically
  const token = opts.token ?? '';

  console.log('KernelClient: Creating server settings', { baseUrl, wsUrl, token });

  const serverSettings = jls.ServerConnection.makeSettings({
    baseUrl,
    wsUrl,
    token,
    init: { credentials: 'omit' }
  });

  console.log('KernelClient: Server settings created', serverSettings);

  console.log('KernelClient: Creating KernelManager');
  const kernelManager = new jls.KernelManager({ serverSettings });
  if (kernelManager.ready) {
    console.log('KernelClient: Waiting for KernelManager ready');
    await kernelManager.ready;
  }
  console.log('KernelClient: KernelManager ready');

  console.log('KernelClient: Creating SessionManager');
  const sessionManager = new jls.SessionManager({ serverSettings, kernelManager });
  if (sessionManager.ready) {
    console.log('KernelClient: Waiting for SessionManager ready');
    await sessionManager.ready;
  }
  console.log('KernelClient: SessionManager ready');

  // Jupyter sessions generally require a path and type; provide a default
  const defaultPath = opts.notebookPath || `thin-client-${Date.now()}.ipynb`;
  console.log('KernelClient: Starting new session', { defaultPath });
  const session = await sessionManager.startNew({
    name: 'ai-thin-client',
    path: defaultPath,
    type: 'notebook',
    kernel: { name: opts.kernelName ?? 'python3' }
  });
  console.log('KernelClient: Session started', session.id);
  // Ensure kernel is ready before returning (avoids 404 on early WS connect)
  console.log('KernelClient: Waiting for kernel idle');
  await waitForKernelIdle(session.kernel);
  console.log('KernelClient: Kernel is idle, session ready');
  return session;
}

async function waitForKernelIdle(kernel, timeoutMs = 15000) {
  if (!kernel) throw new Error('Kernel connection missing');
  if (kernel.status === 'idle') return;
  await new Promise((resolve, reject) => {
    const onStatus = (_sender, status) => {
      if (status === 'idle') {
        cleanup();
        resolve();
      }
    };
    const cleanup = () => {
      try { kernel.statusChanged.disconnect(onStatus); } catch (_) {}
      clearTimeout(timer);
    };
    const timer = setTimeout(() => {
      cleanup();
      reject(new Error('Kernel did not reach idle state in time'));
    }, timeoutMs);
    kernel.statusChanged.connect(onStatus);
  });
}

export async function execute(session, code, onIOPub) {
  console.log('KernelClient: execute() called', { code, sessionId: session.id });
  try {
    const future = session.kernel.requestExecute({ code, stop_on_error: true }, true);
    console.log('KernelClient: requestExecute created future', future);
    future.onIOPub = onIOPub;
    console.log('KernelClient: onIOPub handler set, waiting for future.done');
    await future.done;
    console.log('KernelClient: execution completed');
  } catch (error) {
    console.error('KernelClient: execute() error', error);
    throw error;
  }
}

export const interrupt = (s) => s.kernel.interrupt();
export const restart = (s) => s.kernel.restart();
export const shutdown = (s) => s.shutdown();


