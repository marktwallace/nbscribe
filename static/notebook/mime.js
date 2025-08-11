export function renderIOPub(msg, container) {
  const t = msg?.header?.msg_type;
  if (!t || t === 'status') return;

  const div = document.createElement('div');
  if (t === 'stream') {
    div.textContent = msg.content?.text ?? '';
    div.style.whiteSpace = 'pre-wrap';
  } else if (t === 'error') {
    const e = msg.content || {};
    div.textContent = `${e.ename || 'Error'}: ${e.evalue || ''}\n${(e.traceback || []).join('\n')}`;
    div.style.whiteSpace = 'pre-wrap';
    div.style.color = '#b91c1c';
  } else if (t === 'execute_result' || t === 'display_data') {
    const data = msg.content?.data || {};
    if (data['image/png']) {
      const img = document.createElement('img');
      const b64 = Array.isArray(data['image/png']) ? data['image/png'].join('') : data['image/png'];
      img.src = 'data:image/png;base64,' + b64;
      img.style.maxWidth = '100%';
      div.appendChild(img);
    } else if (data['text/html']) {
      const html = Array.isArray(data['text/html']) ? data['text/html'].join('') : data['text/html'];
      div.innerHTML = html;
    } else if (data['text/plain']) {
      const text = Array.isArray(data['text/plain']) ? data['text/plain'].join('') : data['text/plain'];
      div.textContent = text;
      div.style.whiteSpace = 'pre-wrap';
    } else {
      div.textContent = '[unsupported mime]';
    }
  } else {
    return;
  }
  container.appendChild(div);
}


