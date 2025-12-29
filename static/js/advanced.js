let formState = null;
let lastSavedState = null;
let paramChartInstance = null;
let stepSizeChartInstance = null;

function generateColorGradient(count) {
    const colors = [];
    for (let i = 0; i < count; i++) {
        const hue = (i * 360) / count;
        colors.push(`hsl(${hue}, 70%, 55%)`);
    }
    return colors;
}

function saveFormState() {
    const form = document.getElementById('advanced-form');
    const formData = new FormData(form);
    const state = {};
    
    for (let [key, value] of formData.entries()) {
        state[key] = value;
    }
    
    state.equation = document.getElementById('equation').value;
    state.x0 = document.getElementById('x0').value;
    state.y0 = document.getElementById('y0').value;
    state.x_end = document.getElementById('x_end').value;
    state.param_name = document.getElementById('param_name').value;
    state.param_min = document.getElementById('param_min').value;
    state.param_max = document.getElementById('param_max').value;
    state.param_steps = document.getElementById('param_steps').value;
    state.step_sizes = document.getElementById('step_sizes').value;
    state.show_stability = document.getElementById('show_stability').checked;
    state.show_phase_portrait = document.getElementById('show_phase_portrait').checked;
    state.method = document.getElementById('method').value;
    
    return state;
}

function restoreFormState(state) {
    if (!state) return;
    
    document.getElementById('equation').value = state.equation || '';
    document.getElementById('x0').value = state.x0 || '0';
    document.getElementById('y0').value = state.y0 || '1';
    document.getElementById('x_end').value = state.x_end || '10';
    document.getElementById('param_name').value = state.param_name || '';
    document.getElementById('param_min').value = state.param_min || '0.5';
    document.getElementById('param_max').value = state.param_max || '2.0';
    document.getElementById('param_steps').value = state.param_steps || '5';
    document.getElementById('step_sizes').value = state.step_sizes || '0.1, 0.05, 0.01';
    document.getElementById('show_stability').checked = state.show_stability || false;
    document.getElementById('show_phase_portrait').checked = state.show_phase_portrait || false;
    document.getElementById('method').value = state.method || 'rk4';
}

function showModal() {
    const modal = document.getElementById('confirm-modal');
    modal.classList.add('show');
}

function hideModal() {
    const modal = document.getElementById('confirm-modal');
    modal.classList.remove('show');
}

function validateInitialConditions() {
    const x0 = parseFloat(document.getElementById('x0').value);
    const xEnd = parseFloat(document.getElementById('x_end').value);
    const x0Input = document.getElementById('x0');
    const xEndInput = document.getElementById('x_end');
    const xEndError = document.getElementById('xend-error');
    const summary = document.getElementById('initial-conditions-summary');
    
    let isValid = true;
    
    if (isNaN(xEnd) || xEnd <= x0) {
        xEndInput.classList.add('invalid');
        xEndInput.classList.remove('valid');
        xEndError.textContent = '⚠ x_end must be greater than x₀';
        isValid = false;
    } else {
        xEndInput.classList.add('valid');
        xEndInput.classList.remove('invalid');
        xEndError.textContent = '';
    }
    
    if (!isNaN(x0)) {
        x0Input.classList.add('valid');
        x0Input.classList.remove('invalid');
    }
    
    if (isValid && !isNaN(x0) && !isNaN(xEnd)) {
        const range = (xEnd - x0).toFixed(2);
        summary.innerHTML = `Valid range: [${x0}, ${xEnd}] (span: ${range})`;
        summary.className = 'validation-summary success';
    } else {
        summary.innerHTML = '';
    }
    
    return isValid;
}

function validateParameterName() {
    const paramName = document.getElementById('param_name').value;
    const paramInput = document.getElementById('param_name');
    const error = document.getElementById('param-name-error');
    
    if (paramName.length === 0) {
        paramInput.classList.remove('valid', 'invalid');
        error.textContent = '';
        return true;
    }
    
    const validPattern = /^[a-zA-Z]$/;
    if (!validPattern.test(paramName)) {
        paramInput.classList.add('invalid');
        paramInput.classList.remove('valid');
        error.textContent = '⚠ Must be a single letter (a-z, A-Z)';
        return false;
    } else {
        paramInput.classList.add('valid');
        paramInput.classList.remove('invalid');
        error.textContent = '';
        return true;
    }
}

function validateParameterRange() {
    const paramMin = parseFloat(document.getElementById('param_min').value);
    const paramMax = parseFloat(document.getElementById('param_max').value);
    const minInput = document.getElementById('param_min');
    const maxInput = document.getElementById('param_max');
    const minError = document.getElementById('param-min-error');
    const maxError = document.getElementById('param-max-error');
    const summary = document.getElementById('parameter-summary');
    
    let isValid = true;
    
    if (isNaN(paramMin)) {
        minInput.classList.add('invalid');
        minInput.classList.remove('valid');
        minError.textContent = '⚠ Invalid number';
        isValid = false;
    } else {
        minInput.classList.add('valid');
        minInput.classList.remove('invalid');
        minError.textContent = '';
    }
    
    if (isNaN(paramMax)) {
        maxInput.classList.add('invalid');
        maxInput.classList.remove('valid');
        maxError.textContent = '⚠ Invalid number';
        isValid = false;
    } else if (paramMax <= paramMin) {
        maxInput.classList.add('invalid');
        maxInput.classList.remove('valid');
        maxError.textContent = '⚠ Must be greater than minimum';
        isValid = false;
    } else {
        maxInput.classList.add('valid');
        maxInput.classList.remove('invalid');
        maxError.textContent = '';
    }
    
    if (isValid && !isNaN(paramMin) && !isNaN(paramMax)) {
        summary.innerHTML = `Parameter will vary from ${paramMin} to ${paramMax}`;
        summary.className = 'validation-summary success';
    } else {
        summary.innerHTML = '';
    }
    
    return isValid;
}

function validateParameterSteps() {
    const steps = parseInt(document.getElementById('param_steps').value);
    const stepsInput = document.getElementById('param_steps');
    const error = document.getElementById('param-steps-error');
    
    if (isNaN(steps) || steps < 2 || steps > 10) {
        stepsInput.classList.add('invalid');
        stepsInput.classList.remove('valid');
        error.textContent = '⚠ Must be between 2 and 10';
        return false;
    } else {
        stepsInput.classList.add('valid');
        stepsInput.classList.remove('invalid');
        error.textContent = '';
        return true;
    }
}

function validateStepSizes() {
    const stepSizesStr = document.getElementById('step_sizes').value;
    const input = document.getElementById('step_sizes');
    const error = document.getElementById('step-sizes-error');
    const preview = document.getElementById('step-sizes-preview');
    
    if (!stepSizesStr.trim()) {
        input.classList.remove('valid', 'invalid');
        error.textContent = '';
        preview.innerHTML = '';
        return true;
    }
    
    const values = stepSizesStr.split(',').map(s => s.trim());
    const numbers = values.map(v => parseFloat(v));
    
    const hasInvalid = numbers.some(n => isNaN(n) || n <= 0);
    
    if (hasInvalid) {
        input.classList.add('invalid');
        input.classList.remove('valid');
        error.textContent = '⚠ All values must be positive numbers';
        preview.innerHTML = '';
        return false;
    } else {
        input.classList.add('valid');
        input.classList.remove('invalid');
        error.textContent = '';
        
        const sortedNumbers = [...numbers].sort((a, b) => b - a);
        preview.innerHTML = `${numbers.length} step size(s): ${sortedNumbers.join(', ')}`;
        preview.className = 'step-size-preview success';
        
        return true;
    }
}

document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('advanced-form').addEventListener('submit', function(e) {
        e.preventDefault();
        
        lastSavedState = saveFormState();
        
        const formData = {
            equation: document.getElementById('equation').value,
            x0: document.getElementById('x0').value,
            y0: document.getElementById('y0').value,
            x_end: document.getElementById('x_end').value,
            method: document.getElementById('method').value,
            param_name: document.getElementById('param_name').value,
            param_min: document.getElementById('param_min').value,
            param_max: document.getElementById('param_max').value,
            param_steps: document.getElementById('param_steps').value,
            step_sizes: document.getElementById('step_sizes').value,
            show_stability: document.getElementById('show_stability').checked,
            show_phase_portrait: document.getElementById('show_phase_portrait').checked
        };
        
        const resultsSection = document.getElementById('results');
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
        
        showToast('Starting advanced analysis...', 'info');
        
        fetch('/advanced-analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.message || 'Server error');
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.status === 'success') {
                showToast('Analysis completed successfully!', 'success');
                
                if (data.parameter_variation) {
                    displayParameterVariation(data.parameter_variation, formData.equation);
                } else {
                    const card = document.getElementById('param-variation-card');
                    if (card) card.style.display = 'none';
                }
                
                if (data.step_size_comparison) {
                    displayStepSizeComparison(data.step_size_comparison, formData.equation);
                } else {
                    const card = document.getElementById('step-size-card');
                    if (card) card.style.display = 'none';
                }
                
                if (formData.show_stability && data.stability_analysis) {
                    displayStabilityAnalysis(data.stability_analysis, formData.equation);
                } else {
                    const card = document.getElementById('stability-card');
                    if (card) card.style.display = 'none';
                }
                
                if (formData.show_phase_portrait && data.phase_portrait) {
                    displayPhasePortrait(data.phase_portrait, formData.equation);
                } else {
                    const card = document.getElementById('phase-portrait-card');
                    if (card) card.style.display = 'none';
                }
            } else {
                showToast('Analysis failed: ' + data.message, 'error');
            }
        })
        .catch(error => {
            showToast('Error: ' + error.message, 'error');
            console.error('Analysis error:', error);
        });
    });

    document.getElementById('clear-form-btn').addEventListener('click', function() {
        formState = saveFormState();
        showModal();
    });

    document.getElementById('confirm-clear').addEventListener('click', function() {
        const form = document.getElementById('advanced-form');
        form.reset();
        
        const resultsSection = document.getElementById('results');
        resultsSection.style.display = 'none';
        
        hideModal();
        
        const restoreBtn = document.getElementById('restore-form-btn');
        restoreBtn.style.display = 'inline-block';
        
        showToast('Form cleared successfully', 'info');
    });

    document.getElementById('cancel-clear').addEventListener('click', function() {
        hideModal();
        formState = null;
        showToast('Action cancelled', 'warning');
    });

    document.getElementById('restore-form-btn').addEventListener('click', function() {
        if (formState) {
            restoreFormState(formState);
            showToast('Form restored successfully', 'success');
            
            const restoreBtn = document.getElementById('restore-form-btn');
            restoreBtn.style.display = 'none';
            formState = null;
        }
    });

    document.getElementById('confirm-modal').addEventListener('click', function(e) {
        if (e.target === this) {
            hideModal();
            showToast('Action cancelled', 'warning');
        }
    });

    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            const modal = document.getElementById('confirm-modal');
            if (modal.classList.contains('show')) {
                hideModal();
                showToast('Action cancelled', 'warning');
            }
        }
    });

    document.getElementById('equation-examples').addEventListener('change', function() {
        const selectedExample = this.value;
        if (selectedExample) {
            document.getElementById('equation').value = selectedExample;
            document.getElementById('equation').classList.add('auto-filled');
            showToast('Example equation loaded', 'success');
            
            setTimeout(() => {
                document.getElementById('equation').classList.remove('auto-filled');
            }, 1000);
        }
    });

    validateInitialConditions();
    validateStepSizes();
});

function displayParameterVariation(paramData, equation) {
    const card = document.getElementById('param-variation-card');
    const canvas = document.getElementById('param-variation-chart');
    const legend = document.getElementById('param-variation-legend');
    
    card.style.display = 'block';
    
    if (paramChartInstance) {
        paramChartInstance.destroy();
    }
    
    const datasets = [];
    const colors = generateColorGradient(paramData.datasets.length);
    
    paramData.datasets.forEach((dataset, index) => {
        const data = dataset.results.map(point => ({
            x: point.x,
            y: point.y
        }));
        
        const paramValue = parseFloat(dataset.param_value);
        const formattedValue = Math.abs(paramValue) >= 1000 || (Math.abs(paramValue) < 0.01 && paramValue !== 0) 
            ? paramValue.toExponential(2) 
            : paramValue.toString();
        
        datasets.push({
            label: `${paramData.param_name} = ${formattedValue}`,
            data: data,
            borderColor: colors[index],
            backgroundColor: colors[index] + '20',
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 4,
            tension: 0.4
        });
    });
    
    const ctx = canvas.getContext('2d');
    paramChartInstance = new Chart(ctx, {
        type: 'line',
        data: { datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 2,
            plugins: {
                title: {
                    display: true,
                    text: `dy/dx = ${equation}`,
                    color: '#FF4444',
                    font: { size: 16, weight: 'bold' }
                },
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#ffffff',
                        usePointStyle: true,
                        padding: 15
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(26, 26, 26, 0.9)',
                    titleColor: '#FF4444',
                    bodyColor: '#ffffff',
                    borderColor: '#FF4444',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'x',
                        color: '#ffffff',
                        font: { size: 14, weight: 'bold' }
                    },
                    ticks: { color: '#94a3b8' },
                    grid: { color: '#2a2a2a' }
                },
                y: {
                    title: {
                        display: true,
                        text: 'y',
                        color: '#ffffff',
                        font: { size: 14, weight: 'bold' }
                    },
                    ticks: { color: '#94a3b8' },
                    grid: { color: '#2a2a2a' }
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            }
        }
    });
    
    legend.innerHTML = `
        <div style="margin-top: 15px; padding: 15px; background: rgba(255, 68, 68, 0.1); border-radius: 8px; border: 1px solid rgba(255, 68, 68, 0.3);">
            <strong style="color: #FF4444;">Parameter Range:</strong> 
            ${paramData.param_name} ∈ [${paramData.param_min}, ${paramData.param_max}] 
            with ${paramData.param_steps} steps
        </div>
    `;
}

function displayStepSizeComparison(stepData, equation) {
    const card = document.getElementById('step-size-card');
    const canvas = document.getElementById('step-size-chart');
    const tableDiv = document.getElementById('step-size-table');
    
    card.style.display = 'block';
    
    if (stepSizeChartInstance) {
        stepSizeChartInstance.destroy();
    }
    
    const errorData = stepData.error_metrics.map(metric => ({
        x: metric.step_size,
        y: metric.mean_error
    }));
    
    const ctx = canvas.getContext('2d');
    stepSizeChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [{
                label: 'Mean Error vs Step Size',
                data: errorData,
                borderColor: '#FF4444',
                backgroundColor: 'rgba(255, 68, 68, 0.2)',
                borderWidth: 3,
                pointRadius: 6,
                pointHoverRadius: 8,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 2.5,
            plugins: {
                title: {
                    display: true,
                    text: `Error Analysis: dy/dx = ${equation}`,
                    color: '#FF4444',
                    font: { size: 16, weight: 'bold' }
                },
                legend: {
                    display: true,
                    labels: { color: '#ffffff' }
                },
                tooltip: {
                    backgroundColor: 'rgba(26, 26, 26, 0.9)',
                    titleColor: '#FF4444',
                    bodyColor: '#ffffff',
                    borderColor: '#FF4444',
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            return `Error: ${context.parsed.y.toExponential(4)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'Step Size (h)',
                        color: '#ffffff',
                        font: { size: 14, weight: 'bold' }
                    },
                    ticks: { color: '#94a3b8' },
                    grid: { color: '#2a2a2a' }
                },
                y: {
                    type: 'logarithmic',
                    title: {
                        display: true,
                        text: 'Mean Error (log scale)',
                        color: '#ffffff',
                        font: { size: 14, weight: 'bold' }
                    },
                    ticks: { color: '#94a3b8' },
                    grid: { color: '#2a2a2a' }
                }
            }
        }
    });
    
    let tableHTML = `
        <div style="margin-top: 20px;">
            <h4 style="color: #FF4444; margin-bottom: 15px;">Convergence Analysis</h4>
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="background: rgba(255, 68, 68, 0.2); border-bottom: 2px solid #FF4444;">
                        <th style="padding: 12px; text-align: left; color: #ffffff;">Step Size (h)</th>
                        <th style="padding: 12px; text-align: right; color: #ffffff;">Points</th>
                        <th style="padding: 12px; text-align: right; color: #ffffff;">Max Error</th>
                        <th style="padding: 12px; text-align: right; color: #ffffff;">Mean Error</th>
                        <th style="padding: 12px; text-align: right; color: #ffffff;">Convergence Rate</th>
                    </tr>
                </thead>
                <tbody>`;
    
    stepData.datasets.forEach((dataset, idx) => {
        const metric = stepData.error_metrics[idx];
        const bgColor = idx % 2 === 0 ? 'rgba(255, 68, 68, 0.05)' : 'transparent';
        tableHTML += `
                    <tr style="background: ${bgColor}; border-bottom: 1px solid #2a2a2a;">
                        <td style="padding: 10px; color: #ffffff;">${metric.step_size}</td>
                        <td style="padding: 10px; text-align: right; color: #94a3b8;">${dataset.num_points}</td>
                        <td style="padding: 10px; text-align: right; color: #94a3b8;">${metric.max_error.toExponential(4)}</td>
                        <td style="padding: 10px; text-align: right; color: #94a3b8;">${metric.mean_error.toExponential(4)}</td>
                        <td style="padding: 10px; text-align: right; color: ${metric.convergence_rate ? '#FF4444' : '#666'}; font-weight: ${metric.convergence_rate ? 'bold' : 'normal'};">
                            ${idx === 0 ? 'Reference' : idx === 1 ? '—' : (metric.convergence_rate !== null ? metric.convergence_rate.toFixed(2) : '—')}
                        </td>
                    </tr>`;
    });
    
    tableHTML += `
                </tbody>
            </table>
        </div>`;
    
    tableDiv.innerHTML = tableHTML;
}

function displayStabilityAnalysis(stabilityData, equation) {
    const card = document.getElementById('stability-card');
    const placeholder = document.getElementById('stability-placeholder');
    
    card.style.display = 'block';
    
    if (stabilityData.error) {
        placeholder.innerHTML = `
            <div style="padding: 20px; text-align: center; color: #FF4444;">
                <p><strong>Error:</strong> ${stabilityData.error}</p>
            </div>`;
        return;
    }
    
    const equilibriumPoints = stabilityData.equilibrium_points || [];
    
    if (equilibriumPoints.length === 0) {
        placeholder.innerHTML = `
            <div style="padding: 20px; text-align: center; color: #94a3b8;">
                <p>No equilibrium points found for dy/dx = ${equation}</p>
                <p style="font-size: 0.9em; margin-top: 10px;">Equilibrium points satisfy dy/dx = 0</p>
            </div>`;
        return;
    }
    
    let stabilityHTML = `
        <div style="padding: 20px;">
            <h4 style="color: #FF4444; margin-bottom: 15px;">Equilibrium Points Analysis</h4>
            <p style="color: #94a3b8; margin-bottom: 20px; font-size: 0.95em;">
                For equation: dy/dx = ${equation}
            </p>
            <div style="display: grid; gap: 15px;">`;
    
    equilibriumPoints.forEach((point, idx) => {
        const stabilityColor = {
            'stable': '#22c55e',
            'unstable': '#ef4444',
            'neutral': '#eab308',
            'unknown': '#94a3b8'
        }[point.stability] || '#94a3b8';
        
        const stabilityIcon = {
            'stable': 'Stable',
            'unstable': '✗',
            'neutral': '○',
            'unknown': '?'
        }[point.stability] || '?';
        
        stabilityHTML += `
            <div style="background: rgba(255, 68, 68, 0.05); border-left: 3px solid ${stabilityColor}; padding: 15px; border-radius: 5px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="color: #ffffff; font-weight: bold;">Equilibrium Point ${idx + 1}:</span>
                        <span style="color: #FF4444; font-size: 1.1em; margin-left: 10px;">y = ${point.y_value}</span>
                    </div>
                    <div style="background: ${stabilityColor}; color: #000; padding: 5px 15px; border-radius: 20px; font-weight: bold; font-size: 0.9em;">
                        ${stabilityIcon} ${point.stability.toUpperCase()}
                    </div>
                </div>
                <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #2a2a2a;">
                    <span style="color: #94a3b8;">Derivative at equilibrium:</span>
                    <span style="color: #ffffff; margin-left: 10px; font-family: monospace;">${point.derivative}</span>
                </div>
            </div>`;
    });
    
    stabilityHTML += `
            </div>
            <div style="margin-top: 20px; padding: 15px; background: rgba(26, 26, 26, 0.5); border-radius: 5px; font-size: 0.9em;">
                <p style="color: #94a3b8; margin-bottom: 5px;"><strong>Classification Guide:</strong></p>
                <p style="color: #22c55e;">Stable: f'(y) < 0 - Solutions converge to this point</p>
                <p style="color: #ef4444;">✗ Unstable: f'(y) > 0 - Solutions diverge from this point</p>
                <p style="color: #eab308;">○ Neutral: f'(y) = 0 - Further analysis required</p>
            </div>
        </div>`;
    
    placeholder.innerHTML = stabilityHTML;
}

let phasePortraitChartInstance = null;

function displayPhasePortrait(phaseData, equation) {
    const card = document.getElementById('phase-portrait-card');
    const canvas = document.getElementById('phase-portrait-chart');
    
    if (!card || !canvas) return;
    
    card.style.display = 'block';
    
    if (phasePortraitChartInstance) {
        phasePortraitChartInstance.destroy();
    }
    
    if (phaseData.error) {
        card.innerHTML = `
            <h3>Phase Portrait</h3>
            <div style="padding: 20px; text-align: center; color: #FF4444;">
                <p><strong>Error:</strong> ${phaseData.error}</p>
            </div>`;
        return;
    }
    
    const trajectories = phaseData.trajectories || [];
    const equilibriumPoints = phaseData.equilibrium_points || [];
    
    const datasets = [];
    const colors = generateColorGradient(trajectories.length);
    
    trajectories.forEach((traj, idx) => {
        datasets.push({
            label: `(${traj.x0}, ${traj.y0})`,
            data: traj.points.map(p => ({ x: p.x, y: p.y })),
            borderColor: colors[idx],
            backgroundColor: 'transparent',
            borderWidth: 2.5,
            pointRadius: 0,
            pointHoverRadius: 0,
            tension: 0.4
        });
    });
    
    if (equilibriumPoints.length > 0) {
        equilibriumPoints.forEach((eq, idx) => {
            const yValue = parseFloat(eq.y_value);
            if (!isNaN(yValue)) {
                const color = eq.stability === 'stable' ? '#22c55e' : eq.stability === 'unstable' ? '#ef4444' : '#eab308';
                datasets.push({
                    label: `y=${eq.y_value} (${eq.stability})`,
                    data: phaseData.x_range.map(x => ({ x: x, y: yValue })),
                    borderColor: color,
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    borderDash: [8, 4],
                    pointRadius: 0,
                    tension: 0
                });
            }
        });
    }
    
    const ctx = canvas.getContext('2d');
    phasePortraitChartInstance = new Chart(ctx, {
        type: 'line',
        data: { datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 2.2,
            plugins: {
                title: {
                    display: true,
                    text: `Phase Portrait: dy/dx = ${equation}`,
                    color: '#FF4444',
                    font: { size: 16, weight: 'bold' }
                },
                legend: {
                    display: true,
                    position: 'bottom',
                    labels: {
                        color: '#ffffff',
                        usePointStyle: true,
                        padding: 8,
                        font: { size: 10 },
                        boxWidth: 15,
                        boxHeight: 2
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(26, 26, 26, 0.9)',
                    titleColor: '#FF4444',
                    bodyColor: '#ffffff',
                    borderColor: '#FF4444',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'x',
                        color: '#ffffff',
                        font: { size: 14, weight: 'bold' }
                    },
                    ticks: { color: '#94a3b8' },
                    grid: { color: '#2a2a2a' }
                },
                y: {
                    title: {
                        display: true,
                        text: 'y',
                        color: '#ffffff',
                        font: { size: 14, weight: 'bold' }
                    },
                    ticks: { color: '#94a3b8' },
                    grid: { color: '#2a2a2a' }
                }
            }
        }
    });
}
