const modelSelect = document.getElementById('model-select');
const currentModelName = document.getElementById('current-model-name');
const chatMessages = document.getElementById('chat-messages');
const chatForm = document.getElementById('chat-form');
const messageInput = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const newChatBtn = document.getElementById('new-chat-btn');

let conversationHistory = [];
let isGenerating = false;

// Configure marked.js to use highlight.js
marked.setOptions({
    highlight: function(code, lang) {
        if (lang && hljs.getLanguage(lang)) {
            return hljs.highlight(code, { language: lang }).value;
        }
        return hljs.highlightAuto(code).value;
    }
});

// Auto-resize textarea
messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
    if(this.value.trim() === '') {
        this.style.height = 'auto';
    }
});

// Handle Enter key to send (Shift+Enter for new line)
messageInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event('submit'));
    }
});

// Fetch available models
async function loadModels() {
    try {
        const response = await fetch('/models');
        const models = await response.json();
        
        modelSelect.innerHTML = '';
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = model.name;
            modelSelect.appendChild(option);
        });
        
        if(models.length > 0) {
            updateModelHeader();
        }
    } catch (error) {
        console.error('Failed to load models:', error);
        currentModelName.textContent = 'Error loading models';
    }
}

function updateModelHeader() {
    const selectedOption = modelSelect.options[modelSelect.selectedIndex];
    if(selectedOption) {
        currentModelName.textContent = `Chatting with ${selectedOption.textContent}`;
    }
}

modelSelect.addEventListener('change', updateModelHeader);

function appendMessage(role, content) {
    const welcomeScreen = document.querySelector('.welcome-screen');
    if (welcomeScreen) {
        welcomeScreen.remove();
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = role === 'user' ? 'U' : 'AI';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    if (role === 'ai') {
        contentDiv.innerHTML = marked.parse(content);
    } else {
        contentDiv.textContent = content; // User text as plain text to prevent injection
    }
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return contentDiv;
}

function showTypingIndicator() {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ai typing-message`;
    messageDiv.id = 'typing-indicator';
    
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = 'AI';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = `
        <div class="typing-indicator">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        </div>
    `;
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) {
        indicator.remove();
    }
}

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    if (isGenerating) return;
    
    const message = messageInput.value.trim();
    if (!message) return;
    
    // Clear input
    messageInput.value = '';
    messageInput.style.height = 'auto';
    
    // Add user message to UI
    appendMessage('user', message);
    
    // Update history
    conversationHistory.push({ role: 'user', content: message });
    
    // Prepare for generation
    isGenerating = true;
    sendBtn.disabled = true;
    const selectedModel = modelSelect.value;
    
    showTypingIndicator();
    
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: selectedModel,
                messages: conversationHistory
            })
        });
        
        removeTypingIndicator();
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        // Handle streaming response
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        
        // Create an empty AI message div
        const welcomeScreen = document.querySelector('.welcome-screen');
        if (welcomeScreen) welcomeScreen.remove();
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ai`;
        
        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.textContent = 'AI';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        
        let aiResponse = "";
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value, { stream: true });
            aiResponse += chunk;
            contentDiv.innerHTML = marked.parse(aiResponse);
            
            // Auto scroll down as chunks arrive
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        // Add full AI response to history
        conversationHistory.push({ role: 'assistant', content: aiResponse });
        
    } catch (error) {
        removeTypingIndicator();
        appendMessage('ai', 'Sorry, I encountered an error. Make sure Ollama is running and the selected model is pulled.');
        console.error(error);
    } finally {
        isGenerating = false;
        sendBtn.disabled = false;
        messageInput.focus();
    }
});

newChatBtn.addEventListener('click', () => {
    conversationHistory = [];
    chatMessages.innerHTML = `
        <div class="welcome-screen">
            <div class="welcome-icon">👋</div>
            <h1>How can I help you code today?</h1>
            <p>Select a model from the sidebar and ask me anything about programming.</p>
        </div>
    `;
    messageInput.value = '';
    messageInput.style.height = 'auto';
    messageInput.focus();
});

// Initialize
loadModels();
