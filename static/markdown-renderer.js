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
    
    const rawContent = messageElement.textContent;
    if (typeof marked !== 'undefined' && rawContent.trim()) {
        try {
            // First, check for tool directives and parse them
            const processedContent = processToolDirectives(rawContent);
            
            // Then render the markdown
            const renderedHtml = marked.parse(processedContent);
            messageElement.innerHTML = '<div class="markdown-body">' + renderedHtml + '</div>';
        } catch (error) {
            console.warn('Markdown rendering failed:', error);
            // Keep original text content if rendering fails
        }
    }
}

function processToolDirectives(markdownText) {
    // Pattern to match code block followed by tool directive
    const pattern = /```(\w+)?\n([\s\S]*?)\n```\s*\n\s*```\s*\n([\s\S]*?)\n```/g;
    
    return markdownText.replace(pattern, function(match, language, code, metadata) {
        language = language || 'python';
        code = code.trim();
        metadata = metadata.trim();
        
        // Parse the metadata
        const directive = parseDirectiveMetadata(metadata, code, language);
        
        if (directive && validateDirective(directive)) {
            return createDirectiveHTML(directive, code, language);
        } else {
            // Malformed directive - show error
            console.error('Invalid tool directive:', metadata);
            return createErrorHTML(code, language, `❌ MALFORMED TOOL DIRECTIVE: ${metadata}`);
        }
    });
}

function parseDirectiveMetadata(metadata, code, language) {
    try {
        const lines = metadata.split('\n').map(line => line.trim()).filter(line => line);
        
        let tool = null;
        let pos = null;
        let cell_id = null;
        
        for (const line of lines) {
            if (line.startsWith('TOOL:')) {
                tool = line.split(':', 2)[1].trim();
            } else if (line.startsWith('POS:')) {
                pos = parseInt(line.split(':', 2)[1].trim());
            } else if (line.startsWith('CELL_ID:')) {
                cell_id = line.split(':', 2)[1].trim();
            }
        }
        
        if (!tool) return null;
        
        return {
            tool: tool,
            pos: pos,
            cell_id: cell_id,
            code: code,
            language: language,
            id: generateDirectiveId()
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
    
    if (directive.tool === 'insert_cell' && directive.pos === null) {
        return false;
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
        approveText = `✅ Insert Cell${directive.pos !== null ? ` (pos ${directive.pos})` : ''}`;
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
    const assistantMessages = document.querySelectorAll('chat-msg[role="assistant"]');
    assistantMessages.forEach(function(messageEl) {
        renderMarkdownInElement(messageEl);
    });
}

// Auto-initialize for archived logs
function initArchivedLog() {
    initMarkdownRenderer();
    renderAllMarkdown();
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