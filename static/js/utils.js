function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    
    const container = document.getElementById('toast-container');
    container.appendChild(toast);
    
    setTimeout(() => toast.classList.add('show'), 10);
    
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function parseEquationToFunction(equation) {
    let cleaned = equation.replace(/\s+/g, '');

    cleaned = cleaned.replace(/sin/g, 'Math.sin');
    cleaned = cleaned.replace(/cos/g, 'Math.cos');
    cleaned = cleaned.replace(/tan/g, 'Math.tan');
    cleaned = cleaned.replace(/exp/g, 'Math.exp');
    cleaned = cleaned.replace(/log/g, 'Math.log');
    cleaned = cleaned.replace(/sqrt/g, 'Math.sqrt');
    cleaned = cleaned.replace(/\*\*/g, '**');

    return new Function('x', 'y', `return ${cleaned};`);
}

const methodDescriptions = {
    'euler': 'Numerical method - requires f(x,y) and step size',
    'direct_integration': 'For y\' = f(x) - function of x only',
    'separation': 'For y\' = g(x)·h(y) - separable equations',
    'integrating_factor': 'For y\' + P(x)y = Q(x) - linear first-order',
    'substitution': 'Uses variable substitution u for transformation'
};

function getMethodDescription(method) {
    return methodDescriptions[method] || 'Unknown method';
}
