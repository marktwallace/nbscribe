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
    // Pattern to match code block followed by tool directive
    const pattern = /```(\w+)?\n([\s\S]*?)\n```\s*\n\s*```\s*\n([\s\S]*?)\n```/g;
    
    console.log('🔧 PROCESSING TOOL DIRECTIVES');
    console.log('🔧 Input text:', markdownText);
    console.log('🔧 Pattern:', pattern);
    
    let matchCount = 0;
    let seq = 0;
    const result = markdownText.replace(pattern, function(match, language, code, metadata) {
        matchCount++;
        console.log(`🔧 MATCH ${matchCount}:`, { language, code: code?.substring(0, 100), metadata });
        
        language = language || 'python';
        code = code.trim();
        metadata = metadata.trim();
        
        // Parse the metadata
        const directive = parseDirectiveMetadata(metadata, code, language);
        console.log('🔧 PARSED DIRECTIVE:', directive);
        
        if (directive && validateDirective(directive)) {
            // Attach ordering info for this assistant message group
            directive.group_id = groupId || 'group_default';
            directive.seq = seq++;
            console.log('🔧 DIRECTIVE VALID - Creating HTML');
            return createDirectiveHTML(directive, code, language);
        } else {
            // Malformed directive - show error
            console.error('🔧 Invalid tool directive:', metadata);
            return createErrorHTML(code, language, `❌ MALFORMED TOOL DIRECTIVE: ${metadata}`);
        }
    });
    
    console.log(`🔧 Found ${matchCount} matches`);
    console.log('🔧 Result:', result);
    return result;
}

function parseDirectiveMetadata(metadata, code, language) {
    try {
        const lines = metadata.split('\n').map(line => line.trim()).filter(line => line);
        
        let tool = null;
        let pos = null;
        let cell_id = null;
        let before = null;
        let after = null;
        
        for (const line of lines) {
            if (line.startsWith('TOOL:')) {
                tool = line.split(':', 2)[1].trim();
            } else if (line.startsWith('POS:')) {
                pos = parseInt(line.split(':', 2)[1].trim());
            } else if (line.startsWith('CELL_ID:')) {
                cell_id = line.split(':', 2)[1].trim();
            } else if (line.startsWith('BEFORE:')) {
                before = line.split(':', 2)[1].trim();
            } else if (line.startsWith('AFTER:')) {
                after = line.split(':', 2)[1].trim();
            }
        }
        
        if (!tool) return null;
        
        // Get session ID from meta tag
        const sessionId = document.querySelector('meta[name="conversation-id"]')?.content || null;
        
        return {
            tool: tool,
            pos: pos,
            cell_id: cell_id,
            before: before,
            after: after,
            code: code,
            language: language,
            id: generateDirectiveId(),
            session_id: sessionId
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
        <button class="approve-btn" 
                onclick="approveDirective('${directive.id}')" 
                data-directive='${directiveJson}'>
            ${approveText}
        </button>
        <button class="reject-btn" 
                onclick="rejectDirective('${directive.id}')">
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
        const button = document.querySelector(`button[onclick="approveDirective('${directiveId}')"]`);
        const directiveData = JSON.parse(button.getAttribute('data-directive'));
        
        // Disable buttons
        disableDirectiveButtons(directiveId);
        
        // Call backend API to apply the directive
        const response = await fetch('/api/directives/approve', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(directiveData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            // Replace buttons with success status
            replaceDirectiveButtons(directiveId, true, result.message || 'Applied successfully');

            // Try to refresh the embedded Jupyter notebook iframe so changes appear without manual reload
            try {
                const iframe = document.getElementById('notebook-iframe');
                if (iframe) {
                    // Give the backend a brief moment to finish any async follow-ups
                    setTimeout(async () => {
                        const win = iframe.contentWindow;
                        if (!win) {
                            console.log('No iframe contentWindow; skipping in-iframe refresh');
                            return;
                        }

                        try {
                            // Preferred: JupyterLab document reload (no navigation)
                            if (win.jupyterapp?.commands?.execute) {
                                try {
                                    await win.jupyterapp.commands.execute('docmanager:reload');
                                    console.log('🔄 JupyterLab docmanager:reload executed');
                                    return;
                                } catch {}
                                try {
                                    await win.jupyterapp.commands.execute('docmanager:revert');
                                    console.log('🔄 JupyterLab docmanager:revert executed');
                                    return;
                                } catch {}
                            }

                            // Classic Notebook fallback: save, clear dirty flag, avoid prompt, then soft reload
                            if (win.Jupyter?.notebook) {
                                try {
                                    // Best-effort save to clear dirty state
                                    if (typeof win.Jupyter.notebook.save_notebook === 'function') {
                                        try {
                                            // Try to await save completion via promise or event; fallback timeout
                                            await new Promise((resolve) => {
                                                let resolved = false;
                                                try {
                                                    const ev = win.Jupyter.notebook.events;
                                                    if (ev && typeof ev.one === 'function') {
                                                        ev.one('notebook_saved.Notebook', () => { resolved = true; resolve(); });
                                                    }
                                                } catch {}
                                                const maybe = win.Jupyter.notebook.save_notebook();
                                                if (maybe && typeof maybe.then === 'function') {
                                                    maybe.then(() => { if (!resolved) resolve(); }).catch(() => { if (!resolved) resolve(); });
                                                }
                                                // Fallback resolve
                                                setTimeout(() => { if (!resolved) resolve(); }, 1200);
                                            });
                                            console.log('💾 Classic Notebook saved before reload');
                                        } catch {}
                                    }
                                    if (typeof win.Jupyter.notebook.set_dirty === 'function') {
                                        win.Jupyter.notebook.set_dirty(false);
                                    }
                                    win.Jupyter.notebook.dirty = false;
                                } catch {}
                                // Temporarily suppress beforeunload prompts by blocking listeners
                                const stopper = (ev) => {
                                    try { ev.stopImmediatePropagation?.(); } catch {}
                                };
                                try { win.addEventListener('beforeunload', stopper, true); } catch {}
                                try { win.onbeforeunload = null; } catch {}
                                try {
                                    win.location.reload();
                                    console.log('🔄 Classic Notebook location.reload()');
                                } catch (e) {
                                    console.warn('Classic Notebook reload failed', e);
                                }
                                // Clean up stopper shortly after
                                setTimeout(() => { try { win.removeEventListener('beforeunload', stopper, true); } catch {} }, 2000);
                                return;
                            }

                            // Last resort: attempt a simple in-frame reload
                            try { win.onbeforeunload = null; } catch {}
                            try { win.location.reload(); } catch {}
                        } catch (e) {
                            console.warn('In-iframe refresh failed:', e);
                        }
                    }, 150);
                } else {
                    console.log('No notebook iframe found; skipping iframe refresh');
                }
            } catch (e) {
                console.warn('Notebook iframe refresh skipped due to error:', e);
            }
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