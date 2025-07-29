/**
 * nbscribe Chat Interface JavaScript
 * Interactive chat functionality with streaming support
 */

class ChatInterface {
    constructor() {
        this.chatForm = document.getElementById('chat-form');
        this.chatInput = document.getElementById('chat-input');
        this.chatSubmit = document.getElementById('chat-submit');
        this.chatHistory = document.getElementById('conversation');
        this.loadingIndicator = document.getElementById('loading-indicator');
        this.errorMessage = document.getElementById('error-message');
        
        // Extract session ID from page metadata
        this.sessionId = document.querySelector('meta[name="conversation-id"]').getAttribute('content');
        
        this.initMarkdown();
        this.initEventListeners();
        this.renderExistingMarkdown(); // Render any existing messages
        this.scrollToBottom(); // Scroll to bottom on page load
    }
    
    initMarkdown() {
        // Use shared markdown configuration
        initMarkdownRenderer();
    }
    
    renderExistingMarkdown() {
        // Use shared markdown rendering function
        renderAllMarkdown();
    }
    
    renderMessageMarkdown(messageElement) {
        // Use shared markdown rendering function
        renderMarkdownInElement(messageElement);
    }
    
    initEventListeners() {
        this.chatForm.addEventListener('submit', (e) => this.handleSubmit(e));
        
        // Auto-resize textarea
        this.chatInput.addEventListener('input', () => this.autoResize());
        
        // Remove Enter key submission - button only now
        // Users can use Enter for new lines without accidentally sending
    }
    
    autoResize() {
        this.chatInput.style.height = 'auto';
        this.chatInput.style.height = Math.max(this.chatInput.scrollHeight, 66) + 'px';
    }
    
    async handleSubmit(e) {
        e.preventDefault();
        
        const message = this.chatInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        this.addMessage('user', message);
        
        // Clear input and disable form
        this.chatInput.value = '';
        this.autoResize();
        this.setFormDisabled(true);
        this.showLoadingDots(); // Show dots for initial latency
        this.hideError();
        
        try {
            // Try streaming first, fallback to regular if it fails
            const success = await this.tryStreamingResponse(message);
            if (!success) {
                // Fallback to regular response
                this.hideLoadingDots(); // Hide dots before showing fallback response
                const response = await this.sendMessage(message);
                this.addMessage('assistant', response.response);
            }
        } catch (error) {
            this.showError(`Error: ${error.message}`);
            console.error('Chat error:', error);
        } finally {
            this.setFormDisabled(false);
            this.hideLoadingDots();
            this.chatInput.focus();
        }
    }
    
    async tryStreamingResponse(message) {
        // Try to get a streaming response, return true if successful
        try {
            const response = await fetch('/api/chat/stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    message: message,
                    session_id: this.sessionId
                })
            });
            
            if (!response.ok) {
                console.warn('Streaming failed, will use fallback');
                return false;
            }
            
            // Hide loading dots once streaming starts
            this.hideLoadingDots();
            
            // Create assistant message element for streaming
            const messageEl = document.createElement('chat-msg');
            messageEl.setAttribute('role', 'assistant');
            messageEl.setAttribute('timestamp', new Date().toISOString());
            messageEl.textContent = '';
            
            // Insert before loading indicator
            this.chatHistory.insertBefore(messageEl, this.loadingIndicator);
            this.scrollToBottom();
            
            // Process the stream
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); // Keep incomplete line in buffer
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            
                            if (data.type === 'chunk') {
                                // Append chunk to message
                                messageEl.textContent += data.content;
                                this.scrollToBottom();
                            } else if (data.type === 'complete') {
                                // Streaming complete - render markdown and file already saved on server
                                this.renderMessageMarkdown(messageEl);
                                return true;
                            } else if (data.type === 'error') {
                                // Remove partial message and let fallback handle it
                                messageEl.remove();
                                throw new Error(data.content);
                            }
                        } catch (parseError) {
                            console.warn('Failed to parse SSE data:', parseError);
                        }
                    }
                }
            }
            
            return true;
            
        } catch (error) {
            console.warn('Streaming failed:', error);
            return false;
        }
    }
    
    async sendMessage(message) {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                message: message,
                session_id: this.sessionId
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error || 'Unknown error occurred');
        }
        
        return data;
    }
    
    addMessage(role, content) {
        const messageEl = document.createElement('chat-msg');
        messageEl.setAttribute('role', role);
        messageEl.setAttribute('timestamp', new Date().toISOString());
        messageEl.textContent = content;
        
        // Insert before loading indicator
        this.chatHistory.insertBefore(messageEl, this.loadingIndicator);
        
        // Render markdown for assistant messages
        if (role === 'assistant') {
            this.renderMessageMarkdown(messageEl);
        }
        
        this.scrollToBottom();
    }
    
    setFormDisabled(disabled) {
        this.chatSubmit.disabled = disabled;
        this.chatInput.disabled = disabled;
        this.chatSubmit.textContent = disabled ? 'Sending...' : 'Send';
    }
    
    showLoadingDots() {
        this.loadingIndicator.style.display = 'block';
        this.scrollToBottom();
    }
    
    hideLoadingDots() {
        this.loadingIndicator.style.display = 'none';
    }
    
    // Legacy method for backward compatibility
    setLoading(loading) {
        this.setFormDisabled(loading);
        if (loading) {
            this.showLoadingDots();
        } else {
            this.hideLoadingDots();
        }
    }
    
    showError(message) {
        this.errorMessage.textContent = message;
        this.errorMessage.style.display = 'block';
        this.scrollToBottom();
    }
    
    hideError() {
        this.errorMessage.style.display = 'none';
    }
    
    scrollToBottom() {
        // Use smooth scrolling to bottom of document
        window.scrollTo({
            top: document.body.scrollHeight,
            behavior: 'smooth'
        });
    }
}

// Initialize chat interface when page loads
document.addEventListener('DOMContentLoaded', () => {
    new ChatInterface();
}); 