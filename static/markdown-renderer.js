/**
 * nbscribe Shared Markdown Renderer
 * Common markdown configuration and rendering used by both live and archived conversations
 */

function initMarkdownRenderer() {
    // Configure marked for consistent rendering - more like ChatGPT
    if (typeof marked !== 'undefined') {
        marked.setOptions({
            breaks: true,
            gfm: true,
            headerIds: false,
            mangle: false,
            pedantic: false,  // Don't be overly strict about markdown
            smartypants: false // Prevent smart quote conversion
        });
        
        // Custom renderer to handle lists more like ChatGPT
        const renderer = new marked.Renderer();
        
        // Override list item rendering to reduce paragraph wrapping
        renderer.listitem = function(text) {
            // If the text is simple (no nested blocks), don't wrap in <p>
            if (!text.includes('<p>') && !text.includes('<blockquote>') && !text.includes('<pre>')) {
                return '<li>' + text.trim() + '</li>\n';
            }
            return '<li>' + text + '</li>\n';
        };
        
        marked.setOptions({ renderer: renderer });
    }
}

function renderMarkdownInElement(messageElement) {
    // Only render markdown for assistant messages
    if (messageElement.getAttribute('role') !== 'assistant') return;
    
    console.log('🎨 RENDERING MARKDOWN for assistant message');
    
    const rawContent = messageElement.textContent;
    const groupId = messageElement.getAttribute('timestamp') || `group_${Date.now()}`;
    console.log('🎨 Raw content:', rawContent);
    
    if (typeof marked !== 'undefined' && rawContent.trim()) {
        try {
            // First, check for tool directives and parse them
            const processedContent = processToolDirectives(rawContent, { groupId });
            
            // Then render the markdown
            const renderedHtml = marked.parse(processedContent);
            messageElement.innerHTML = '<div class="markdown-body">' + renderedHtml + '</div>';
            console.log('🎨 MARKDOWN RENDERED SUCCESSFULLY');
        } catch (error) {
            console.warn('🎨 Markdown rendering failed:', error);
            // Keep original text content if rendering fails
        }
    } else {
        console.log('🎨 SKIPPING MARKDOWN: marked not available or no content');
    }
}

function processToolDirectives(markdownText, { groupId } = {}) {
    console.log('🔧 PROCESSING TOOL DIRECTIVES (tokenizer mode)');
    let seq = 0;

    // Tokenize into text and fenced code blocks
    const fenceRe = /```([\w-]*)\s*\r?\n([\s\S]*?)```/g;
    const blocks = [];
    let lastIndex = 0;
    let m;
    while ((m = fenceRe.exec(markdownText)) !== null) {
        const [all, langRaw, body] = m;
        const start = m.index;
        const end = start + all.length;
        if (start > lastIndex) {
            blocks.push({ type: 'text', content: markdownText.slice(lastIndex, start) });
        }
        blocks.push({ type: 'code', lang: (langRaw || '').trim(), content: (body || '').trim(), raw: all });
        lastIndex = end;
    }
    if (lastIndex < markdownText.length) {
        blocks.push({ type: 'text', content: markdownText.slice(lastIndex) });
    }

    // Walk blocks: look for code fence followed by metadata fence with only TOOL info, allowing only whitespace between
    const out = [];
    for (let i = 0; i < blocks.length; i++) {
        const b = blocks[i];
        if (b.type !== 'code') { out.push(b.content); continue; }

        // Peek ahead to next code fence, ensure only whitespace in-between
        let j = i + 1;
        let between = '';
        while (j < blocks.length && blocks[j].type === 'text' && /^\s*$/.test(blocks[j].content)) {
            between += blocks[j].content;
            j++;
        }
        if (j < blocks.length && blocks[j].type === 'code') {
            const metaBlock = blocks[j];
            const metaText = metaBlock.content.trim();
            const isToolMeta = /^TOOL\s*:/i.test(metaText) || metaText.startsWith('{');
            if (isToolMeta) {
                const directive = parseDirectiveMetadata(metaText, b.content, (b.lang || 'python'), metaBlock.lang || '');
                if (directive && validateDirective(directive)) {
                    directive.group_id = groupId || 'group_default';
                    directive.seq = seq++;
                    out.push(createDirectiveHTML(directive, b.content, b.lang || 'python'));
                    i = j; // skip the meta block
                    continue;
                } else {
                    out.push(createErrorHTML(b.content, b.lang || 'python', `❌ MALFORMED TOOL DIRECTIVE: ${metaText}`));
                    i = j;
                    continue;
                }
            }
        }

        // Not a directive pair; keep original code block
        out.push(b.raw);
    }

    const result = out.join('');
    console.log('🔧 TOKENIZER RESULT length:', result.length);
    return result;
}

function parseDirectiveMetadata(metadata, code, language, metaLang) {
    try {
        // If metadata looks like JSON, prefer JSON parsing
        const metaIsJson = (metaLang && typeof metaLang === 'string' && metaLang.toLowerCase() === 'json') || metadata.trim().startsWith('{');
        if (metaIsJson) {
            try {
                const obj = JSON.parse(metadata);
                const norm = {};
                Object.keys(obj).forEach(k => { norm[k.toLowerCase()] = obj[k]; });
                const sessionIdFromMeta = norm.session_id || null;
                const sessionId = sessionIdFromMeta || (document.querySelector('meta[name="conversation-id"]')?.content || null);
                return {
                    tool: norm.tool,
                    pos: norm.pos ?? null,
                    cell_id: norm.cell_id ?? null,
                    before: norm.before ?? null,
                    after: norm.after ?? null,
                    code: code,
                    language: language,
                    id: norm.id || generateDirectiveId(),
                    session_id: sessionId,
                    group_id: norm.group_id || undefined,
                    seq: norm.seq ?? undefined
                };
            } catch (e) {
                console.warn('Metadata JSON parse failed, falling back to KV parsing:', e);
            }
        }

        const lines = metadata.split('\n').map(line => line.trim()).filter(line => line);
        
        let tool = null;
        let pos = null;
        let cell_id = null;
        let before = null;
        let after = null;
        let id = null;
        let group_id = null;
        let seq = null;
        let sessionIdFromMeta = null;
        
        for (const line of lines) {
            const u = line.toUpperCase();
            const val = line.split(':', 2)[1]?.trim();
            if (u.startsWith('TOOL:')) {
                tool = val;
            } else if (u.startsWith('POS:')) {
                const n = parseInt(val, 10);
                pos = Number.isFinite(n) ? n : null;
            } else if (u.startsWith('CELL_ID:')) {
                cell_id = val;
            } else if (u.startsWith('BEFORE:')) {
                before = val;
            } else if (u.startsWith('AFTER:')) {
                after = val;
            } else if (u.startsWith('ID:')) {
                id = val;
            } else if (u.startsWith('GROUP_ID:')) {
                group_id = val;
            } else if (u.startsWith('SEQ:')) {
                const n = parseInt(val, 10);
                seq = Number.isFinite(n) ? n : null;
            } else if (u.startsWith('SESSION_ID:')) {
                sessionIdFromMeta = val;
            }
        }
        
        if (!tool) return null;
        
        // Get session ID from meta tag
        const sessionId = sessionIdFromMeta || (document.querySelector('meta[name="conversation-id"]')?.content || null);
        
        return {
            tool: tool,
            pos: pos,
            cell_id: cell_id,
            before: before,
            after: after,
            code: code,
            language: language,
            id: id || generateDirectiveId(),
            session_id: sessionId,
            group_id: group_id || undefined,
            seq: seq ?? undefined
        };
        
    } catch (error) {
        console.error('Error parsing directive metadata:', error);
        return null;
    }
}

function validateDirective(directive) {
    if (!['insert_cell', 'edit_cell', 'delete_cell'].includes(directive.tool)) {
        return false;
    }
    
    if (directive.tool === 'insert_cell') {
        // For insert_cell, we need either BEFORE, AFTER, or POS
        const hasRelativePosition = directive.before || directive.after;
        const hasAbsolutePosition = directive.pos !== null && directive.pos !== undefined;
        
        if (!hasRelativePosition && !hasAbsolutePosition) {
            return false;
        }
        
        // Can't have both relative and absolute positioning
        if (hasRelativePosition && hasAbsolutePosition) {
            return false;
        }
        
        // Can't have both BEFORE and AFTER
        if (directive.before && directive.after) {
            return false;
        }
    }
    
    if (['edit_cell', 'delete_cell'].includes(directive.tool) && !directive.cell_id) {
        return false;
    }
    
    return true;
}

function generateDirectiveId() {
    return `directive_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

function createDirectiveHTML(directive, code, language) {
    const escapedCode = escapeHtml(code);
    
    // Generate button text
    let approveText = '✅ Apply';
    if (directive.tool === 'insert_cell') {
        if (directive.before) {
            approveText = `✅ Insert Cell (before ${directive.before})`;
        } else if (directive.after) {
            approveText = `✅ Insert Cell (after ${directive.after})`;
        } else if (directive.pos !== null) {
            approveText = `✅ Insert Cell (pos ${directive.pos})`;
        } else {
            approveText = '✅ Insert Cell';
        }
    } else if (directive.tool === 'edit_cell') {
        approveText = `✅ Edit Cell${directive.cell_id ? ` (${directive.cell_id})` : ''}`;
    } else if (directive.tool === 'delete_cell') {
        approveText = `✅ Delete Cell${directive.cell_id ? ` (${directive.cell_id})` : ''}`;
    }
    
    const directiveJson = escapeHtml(JSON.stringify(directive));
    
    return `
<div class="tool-directive-container" data-directive-id="${directive.id}">
    <pre><code class="language-${language}">${escapedCode}</code></pre>
    <div class="tool-directive-buttons">
        <button type="button" class="approve-btn" 
                onclick="event && event.stopPropagation && event.stopPropagation(); event && event.preventDefault && event.preventDefault(); approveDirective('${directive.id}'); return false;" 
                data-directive='${directiveJson}'>
            ${approveText}
        </button>
        <button type="button" class="reject-btn" 
                onclick="event && event.stopPropagation && event.stopPropagation(); event && event.preventDefault && event.preventDefault(); rejectDirective('${directive.id}'); return false;">
            ❌ Reject
        </button>
    </div>
</div>
`;
}

function createErrorHTML(code, language, errorMsg) {
    const escapedCode = escapeHtml(code);
    
    return `
<div class="tool-directive-error">
    <pre><code class="language-${language}">${escapedCode}</code></pre>
    <div class="error-message">${errorMsg}</div>
</div>
`;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function renderAllMarkdown() {
    // Render markdown in all assistant messages on the page
    console.log('🎨 RENDER ALL MARKDOWN: Starting');
    const assistantMessages = document.querySelectorAll('chat-msg[role="assistant"]');
    console.log(`🎨 RENDER ALL MARKDOWN: Found ${assistantMessages.length} assistant messages`);
    
    assistantMessages.forEach(function(messageEl, index) {
        console.log(`🎨 RENDER ALL MARKDOWN: Processing message ${index + 1}`);
        renderMarkdownInElement(messageEl);
    });
    
    console.log('🎨 RENDER ALL MARKDOWN: Complete');
}

// Auto-initialize for archived logs
function initArchivedLog() {
    console.log('🚀 INITIALIZING ARCHIVED LOG');
    initMarkdownRenderer();
    renderAllMarkdown();
    console.log('🚀 ARCHIVED LOG INITIALIZED');
}

// Tool Directive Functions
async function approveDirective(directiveId) {
    try {
        console.log('🟦 APPROVE CLICK:', { directiveId });
        // Prefer robust lookup via container and its approve button
        const container = document.querySelector(`[data-directive-id="${directiveId}"]`);
        const button = container ? container.querySelector('button.approve-btn[data-directive]') : null;
        if (!button) {
            throw new Error('Approve button not found for directive');
        }
        const directiveData = JSON.parse(button.getAttribute('data-directive'));
        console.log('🟦 APPROVE DATA:', directiveData);
        
        // Disable buttons
        disableDirectiveButtons(directiveId);
        
        // Call backend API to apply the directive
        console.log('🟦 APPROVE FETCH /api/directives/approve start');
        const response = await fetch('/api/directives/approve', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(directiveData)
        });
        console.log('🟦 APPROVE FETCH status:', response.status);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        console.log('🟦 APPROVE RESULT:', result);
        
        if (result.success) {
            // Replace buttons with success status
            replaceDirectiveButtons(directiveId, true, result.message || 'Applied successfully');

            // In-place classic refresh (no navigation, no Lab dependency)
            console.log('🟦 APPROVE REFRESH calling refreshNotebookIframeLabSafe()');
            await refreshNotebookIframeLabSafe();
            console.log('🟦 APPROVE REFRESH done');
        } else {
            throw new Error(result.error || 'Unknown error occurred');
        }
        
    } catch (error) {
        console.error('Error approving directive:', error);
        // Re-enable buttons and show error
        enableDirectiveButtons(directiveId);
        alert(`Error applying directive: ${error.message}`);
    }
}

// Ensure handlers are available for inline onclick and also via delegated clicks
// Attach to window for inline attribute lookups
try { window.approveDirective = approveDirective; } catch {}
try { window.rejectDirective = rejectDirective; } catch {}

// Event delegation as a safety net in case inline onclick is blocked/sanitized
(function bindDirectiveDelegates() {
    if (window.__nbscribeDirectiveDelegateBound) return;
    window.__nbscribeDirectiveDelegateBound = true;
    document.addEventListener('click', (ev) => {
        const approveBtn = ev.target && ev.target.closest && ev.target.closest('button.approve-btn[data-directive]');
        if (approveBtn) {
            try {
                const data = JSON.parse(approveBtn.getAttribute('data-directive'));
                console.log('🟧 DELEGATE APPROVE CLICK:', data);
                approveDirective(data.id);
                ev.preventDefault();
                ev.stopPropagation();
            } catch (e) {
                console.warn('Delegate approve failed:', e);
            }
            return;
        }
        const rejectBtn = ev.target && ev.target.closest && ev.target.closest('button.reject-btn');
        if (rejectBtn) {
            const container = rejectBtn.closest('[data-directive-id]');
            const id = container ? container.getAttribute('data-directive-id') : null;
            console.log('🟧 DELEGATE REJECT CLICK:', { id });
            try { if (id) rejectDirective(id); } catch {}
            ev.preventDefault();
            ev.stopPropagation();
        }
    }, true);
})();

// Helper: Wait for JupyterLab app inside the iframe
async function waitForLabApp(win, timeoutMs = 10000) {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
        try {
            if (win && win.jupyterapp && win.jupyterapp.commands && typeof win.jupyterapp.commands.execute === 'function') {
                return true;
            }
        } catch {}
        await new Promise(r => setTimeout(r, 100));
    }
    return false;
}

// Helper: Extract notebook path from iframe URL
function getNotebookPathFromIframe() {
    const iframe = document.getElementById('notebook-iframe');
    if (!iframe) return null;
    const src = iframe.src || '';
    // Support classic Notebook and Lab routes
    let match = src.match(/\/jupyter\/notebooks\/([^?]+)/);
    if (!match) {
        match = src.match(/\/jupyter\/lab\/(?:workspaces\/[^/]+\/)?tree\/([^?]+)/);
    }
    return match ? decodeURIComponent(match[1]) : null;
}

// Perform an in-place refresh via JupyterLab's docmanager without navigation
async function refreshNotebookIframeLabSafe() {
    const iframe = document.getElementById('notebook-iframe');
    if (!iframe) { console.log('No notebook iframe found; skipping refresh'); return; }
    const win = iframe.contentWindow;
    if (!win) { console.log('No iframe contentWindow; skipping refresh'); return; }

    const iframeSrc = iframe.src || '';
    const path = getNotebookPathFromIframe();
    console.log('🔎 Lab refresh context:', { iframeSrc, extractedPath: path });

    if (!path) {
        console.warn('Lab refresh: could not extract notebook path from iframe src');
        return;
    }

    try {
        const ready = await waitForLabApp(win, 10000);
        if (!ready) {
            console.warn('JupyterLab app not ready within timeout; skipping in-place refresh');
            return;
        }

        // Command-based document refresh; avoids navigation and prompts
        await win.jupyterapp.commands.execute('docmanager:reload', { path });
        try { await win.jupyterapp.restored; } catch {}
        try { win.jupyterapp.shell?.collapseLeft?.(); } catch {}
        console.log('✅ Lab docmanager:reload executed');
    } catch (e) {
        console.warn('Lab refresh error:', e);
    }
}

async function rejectDirective(directiveId) {
    try {
        // Disable buttons
        disableDirectiveButtons(directiveId);
        
        // Replace buttons with rejection status (no backend call needed)
        replaceDirectiveButtons(directiveId, false, 'Rejected by user');
        
    } catch (error) {
        console.error('Error rejecting directive:', error);
        enableDirectiveButtons(directiveId);
    }
}

function disableDirectiveButtons(directiveId) {
    const container = document.querySelector(`[data-directive-id="${directiveId}"]`);
    if (container) {
        const buttons = container.querySelectorAll('button');
        buttons.forEach(btn => {
            btn.disabled = true;
            btn.style.opacity = '0.6';
        });
    }
}

function enableDirectiveButtons(directiveId) {
    const container = document.querySelector(`[data-directive-id="${directiveId}"]`);
    if (container) {
        const buttons = container.querySelectorAll('button');
        buttons.forEach(btn => {
            btn.disabled = false;
            btn.style.opacity = '1';
        });
    }
}

function replaceDirectiveButtons(directiveId, success, message) {
    const container = document.querySelector(`[data-directive-id="${directiveId}"]`);
    if (container) {
        const buttonsDiv = container.querySelector('.tool-directive-buttons');
        if (buttonsDiv) {
            const statusClass = success ? 'directive-applied' : 'directive-rejected';
            const statusIcon = success ? '✅' : '❌';
            const statusText = success ? 'Applied' : 'Rejected';
            
            buttonsDiv.innerHTML = `
                <div class="tool-directive-status ${statusClass}">
                    ${statusIcon} ${statusText}: ${message}
                </div>
            `;
        }
    }
} 