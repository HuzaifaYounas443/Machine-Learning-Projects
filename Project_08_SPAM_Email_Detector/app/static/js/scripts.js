// app/static/js/script.js

// Show loading spinner when form is submitted
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('spamForm');
    const spinner = document.getElementById('loadingSpinner');
    const checkBtn = document.getElementById('checkBtn');
    
    if (form) {
        form.addEventListener('submit', function() {
            spinner.style.display = 'block';
            checkBtn.disabled = true;
            checkBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
        });
    }
});

// Auto-resize textarea
document.addEventListener('DOMContentLoaded', function() {
    const textarea = document.querySelector('.message-input');
    if (textarea) {
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = this.scrollHeight + 'px';
        });
    }
});

// Example card click with smooth scroll
function fillExample(card) {
    const message = card.querySelector('p').textContent;
    const textarea = document.getElementById('message');
    if (textarea) {
        textarea.value = message;
        textarea.dispatchEvent(new Event('input'));
        textarea.focus();
        textarea.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

// Clear input
function clearInput() {
    const textarea = document.getElementById('message');
    if (textarea) {
        textarea.value = '';
        textarea.dispatchEvent(new Event('input'));
        textarea.focus();
    }
}

// Keyboard shortcut: Ctrl+Enter to submit
document.addEventListener('keydown', function(e) {
    if (e.ctrlKey && e.key === 'Enter') {
        const form = document.getElementById('spamForm');
        if (form) {
            form.submit();
        }
    }
});