let chartInstance = null;
let currentResults = null;
let lastSolvePayload = null;
let baseY0 = null;
let baseXEnd = null;

let slopeFieldEnabled = false;
let slopeFieldDensity = 20;
let slopeFieldColor = '#FF6B6B';
let slopeFieldOpacity = 0.6;
let currentEquationForSlope = null;

function copyDataToClipboard() {
    if (!currentResults || currentResults.length === 0) {
        showToast('No data to copy', 'error');
        return;
    }

    let text = 'x\ty\n';
    currentResults.forEach(point => {
        text += `${point.x}\t${point.y}\n`;
    });

    navigator.clipboard.writeText(text).then(() => {
        showToast('✓ Data copied to clipboard!', 'success');
    }).catch(() => {
        showToast('Failed to copy data', 'error');
    });
}

function downloadGraphAsPNG() {
    if (!chartInstance) {
        showToast('No graph to download', 'error');
        return;
    }

    const link = document.createElement('a');
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
    link.download = `ode_graph_${timestamp}.png`;
    link.href = chartInstance.toBase64Image();
    link.click();

    showToast('✓ Graph downloaded!', 'success');
}

function calculateStatistics(results) {
    if (!results || results.length === 0) return;

    currentResults = results;
    const yValues = results.map(r => r.y);
    const xValues = results.map(r => r.x);

    const maxY = Math.max(...yValues);
    const minY = Math.min(...yValues);
    const maxIndex = yValues.indexOf(maxY);
    const minIndex = yValues.indexOf(minY);

    const finalPoint = results[results.length - 1];
    const initialY = results[0].y;
    const totalChange = finalPoint.y - initialY;

    document.getElementById('stat-max-y').textContent = maxY.toFixed(4);
    document.getElementById('stat-max-x').textContent = xValues[maxIndex].toFixed(4);
    document.getElementById('stat-min-y').textContent = minY.toFixed(4);
    document.getElementById('stat-min-x').textContent = xValues[minIndex].toFixed(4);
    document.getElementById('stat-final-y').textContent = finalPoint.y.toFixed(4);
    document.getElementById('stat-final-x').textContent = finalPoint.x.toFixed(4);
    document.getElementById('stat-points').textContent = results.length;
    document.getElementById('stat-change').textContent = (totalChange >= 0 ? '+' : '') + totalChange.toFixed(4);

    document.getElementById('stats-panel').style.display = 'block';
}

function exportToCSV() {
    if (!currentResults || currentResults.length === 0) {
        alert('No solution data to export');
        return;
    }

    let csvContent = 'x,y\n';
    currentResults.forEach(point => {
        csvContent += `${point.x},${point.y}\n`;
    });

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
    link.setAttribute('href', url);
    link.setAttribute('download', `ode_solution_${timestamp}.csv`);
    link.style.visibility = 'hidden';

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function plotSolution(results, method, equation) {
    if (typeof Chart === 'undefined') {
        console.error('Chart.js not loaded yet');
        setTimeout(() => plotSolution(results, method, equation), 100);
        return;
    }

    const graphContainer = document.getElementById('graph-container');
    const placeholder = document.querySelector('.visualization-placeholder');

    if (placeholder) {
        placeholder.style.display = 'none';
    }

    if (chartInstance) {
        chartInstance.destroy();
    }

    const xValues = results.map(r => r.x);
    const yValues = results.map(r => r.y);

    const ctx = document.getElementById('solutionChart').getContext('2d');
    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: xValues,
            datasets: [{
                label: `y(x) - ${method}`,
                data: yValues,
                borderColor: '#3498db',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                borderWidth: 2,
                pointRadius: 1.5,
                pointHoverRadius: 5,
                tension: 0.1,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 2,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                },
                title: {
                    display: true,
                    text: `dy/dx = ${equation}`,
                    font: {
                        size: 14
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function (context) {
                            return `y = ${context.parsed.y.toFixed(4)}`;
                        },
                        title: function (context) {
                            return `x = ${context[0].parsed.x.toFixed(4)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'x',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    },
                    ticks: {
                        maxTicksLimit: 10
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'y',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    }
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            }
        }
    });

    if (slopeFieldEnabled || document.getElementById('slope-field-toggle').checked) {
        currentEquationForSlope = equation;
        setTimeout(() => {
            updateSlopeField();
        }, 100);
    }
}

const slopeFieldPlugin = {
    id: 'slopeField',
    afterDraw: (chart) => {
        if (!slopeFieldEnabled || !currentEquationForSlope) return;

        const ctx = chart.ctx;
        const xAxis = chart.scales.x;
        const yAxis = chart.scales.y;

        if (!xAxis || !yAxis) return;

        try {
            const f = parseEquationToFunction(currentEquationForSlope);

            const xMin = xAxis.min;
            const xMax = xAxis.max;
            const yMin = yAxis.min;
            const yMax = yAxis.max;

            ctx.save();
            ctx.strokeStyle = slopeFieldColor;
            ctx.globalAlpha = slopeFieldOpacity;
            ctx.lineWidth = 1.5;

            const arrowLengthPixel = 12;

            for (let i = 0; i <= slopeFieldDensity; i++) {
                for (let j = 0; j <= slopeFieldDensity; j++) {
                    const x = xMin + (i * (xMax - xMin) / slopeFieldDensity);
                    const y = yMin + (j * (yMax - yMin) / slopeFieldDensity);

                    try {
                        const slope = f(x, y);
                        if (!isFinite(slope)) continue;

                        const xPixel = xAxis.getPixelForValue(x);
                        const yPixel = yAxis.getPixelForValue(y);

                        if (xPixel < chart.chartArea.left || xPixel > chart.chartArea.right ||
                            yPixel < chart.chartArea.top || yPixel > chart.chartArea.bottom) {
                            continue;
                        }

                        const deltaX = (xMax - xMin) * 0.001;
                        const deltaY = slope * deltaX;

                        const x1_pix = xAxis.getPixelForValue(x - deltaX);
                        const y1_pix = yAxis.getPixelForValue(y - deltaY);
                        const x2_pix = xAxis.getPixelForValue(x + deltaX);
                        const y2_pix = yAxis.getPixelForValue(y + deltaY);

                        const dx_pix = x2_pix - x1_pix;
                        const dy_pix = y2_pix - y1_pix;
                        const len = Math.sqrt(dx_pix * dx_pix + dy_pix * dy_pix);

                        if (len === 0) continue;

                        const scale = arrowLengthPixel / len;
                        const offX = dx_pix * scale * 0.5;
                        const offY = dy_pix * scale * 0.5;

                        ctx.beginPath();
                        ctx.moveTo(xPixel - offX, yPixel - offY);
                        ctx.lineTo(xPixel + offX, yPixel + offY);
                        ctx.stroke();

                    } catch (e) {
                        continue;
                    }
                }
            }
            ctx.restore();
        } catch (error) {
            console.error('Slope field error:', error);
        }
    }
};

Chart.register(slopeFieldPlugin);

function updateSlopeField() {
    if (chartInstance) chartInstance.update();
}

function toggleSlopeField() {
    slopeFieldEnabled = !slopeFieldEnabled;
    localStorage.setItem('slopeFieldEnabled', slopeFieldEnabled);
    updateSlopeField();
}

function updateSlopeFieldDensity(newDensity) {
    slopeFieldDensity = parseInt(newDensity);
    localStorage.setItem('slopeFieldDensity', slopeFieldDensity);
    updateSlopeField();
}

function updateSlopeFieldColor(newColor) {
    slopeFieldColor = newColor;
    localStorage.setItem('slopeFieldColor', slopeFieldColor);
    updateSlopeField();
}

function updateSlopeFieldOpacity(newOpacity) {
    slopeFieldOpacity = parseFloat(newOpacity);
    localStorage.setItem('slopeFieldOpacity', slopeFieldOpacity);
    updateSlopeField();
}

function toggleSlopeFieldPanel() {
    const panel = document.getElementById('slope-field-controls');
    const btn = document.getElementById('slope-field-btn');

    if (panel.style.display === 'none' || panel.style.display === '') {
        panel.style.display = 'block';
        btn.textContent = '📐 Hide Slope Field';
        btn.style.background = 'linear-gradient(135deg, #FF4444 0%, #CC0000 100%)';

        const checkbox = document.getElementById('slope-field-toggle');
        if (checkbox && !checkbox.checked) {
            checkbox.checked = true;
            slopeFieldEnabled = true;
            updateSlopeField();
        }
    } else {
        panel.style.display = 'none';
        btn.textContent = '📐 Slope Field Visualizer';
        btn.style.background = 'linear-gradient(135deg, #FF4444 0%, #CC0000 100%)';

        const checkbox = document.getElementById('slope-field-toggle');
        if (checkbox && checkbox.checked) {
            checkbox.checked = false;
            slopeFieldEnabled = false;
        }
    }
}

function loadSlopeFieldPreferences() {
    const savedEnabled = localStorage.getItem('slopeFieldEnabled');
    const savedDensity = localStorage.getItem('slopeFieldDensity');
    const savedColor = localStorage.getItem('slopeFieldColor');
    const savedOpacity = localStorage.getItem('slopeFieldOpacity');

    if (savedEnabled !== null) {
        slopeFieldEnabled = savedEnabled === 'true';
    }
    if (savedDensity !== null) {
        slopeFieldDensity = parseInt(savedDensity);
    }
    if (savedColor !== null) {
        slopeFieldColor = savedColor;
    }
    if (savedOpacity !== null) {
        slopeFieldOpacity = parseFloat(savedOpacity);
    }

    const checkbox = document.getElementById('slope-field-toggle');
    const densitySlider = document.getElementById('slope-density-slider');
    const colorPicker = document.getElementById('slope-color-picker');
    const opacitySlider = document.getElementById('slope-opacity-slider');

    if (checkbox) checkbox.checked = slopeFieldEnabled;
    if (densitySlider) densitySlider.value = slopeFieldDensity;
    if (colorPicker) colorPicker.value = slopeFieldColor;
    if (opacitySlider) opacitySlider.value = slopeFieldOpacity;
}

function enableY0Slider(initialY0) {
    baseY0 = parseFloat(initialY0);
    const sliderContainer = document.getElementById('y0-slider-container');
    const slider = document.getElementById('y0-slider');
    const valueDisplay = document.getElementById('y0-value');
    const minDisplay = document.getElementById('y0-min');
    const maxDisplay = document.getElementById('y0-max');

    const minY0 = baseY0 - Math.abs(baseY0) * 0.5;
    const maxY0 = baseY0 + Math.abs(baseY0) * 0.5;

    if (baseY0 === 0) {
        slider.min = -5;
        slider.max = 5;
    } else {
        slider.min = minY0;
        slider.max = maxY0;
    }

    slider.value = baseY0;
    slider.step = Math.abs(maxY0 - minY0) / 100;

    valueDisplay.textContent = baseY0.toFixed(3);
    minDisplay.textContent = slider.min;
    maxDisplay.textContent = slider.max;

    sliderContainer.style.display = 'block';
}

function enableXEndSlider(initialXEnd, x0) {
    baseXEnd = parseFloat(initialXEnd);
    const x0Val = parseFloat(x0);
    const sliderContainer = document.getElementById('xend-slider-container');
    const slider = document.getElementById('xend-slider');
    const valueDisplay = document.getElementById('xend-value');
    const minDisplay = document.getElementById('xend-min');
    const maxDisplay = document.getElementById('xend-max');

    const range = baseXEnd - x0Val;
    const minXEnd = x0Val + range * 0.5;
    const maxXEnd = x0Val + range * 2;

    slider.min = minXEnd;
    slider.max = maxXEnd;
    slider.value = baseXEnd;
    slider.step = (maxXEnd - minXEnd) / 100;

    valueDisplay.textContent = baseXEnd.toFixed(3);
    minDisplay.textContent = minXEnd.toFixed(1);
    maxDisplay.textContent = maxXEnd.toFixed(1);

    sliderContainer.style.display = 'block';
}

function updateSolutionWithNewY0(newY0) {
    if (!lastSolvePayload) return;

    const valueDisplay = document.getElementById('y0-value');
    valueDisplay.textContent = parseFloat(newY0).toFixed(3);

    const updatedPayload = { ...lastSolvePayload, y0: newY0 };

    fetch('/simulate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(updatedPayload)
    })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success' && data.results && data.results.length > 0) {
                const equationDisplay = data.equation || data.parsed_expression || 'Solution';
                plotSolution(data.results, data.method, equationDisplay);
                calculateStatistics(data.results);

                if (slopeFieldEnabled) {
                    setTimeout(() => updateSlopeField(), 100);
                }
            } else {
                console.error('Slider update failed:', data);
            }
        })
        .catch(error => {
            console.error('Slider update error:', error);
        });
}

function updateSolutionWithNewXEnd(newXEnd) {
    if (!lastSolvePayload) return;

    const valueDisplay = document.getElementById('xend-value');
    valueDisplay.textContent = parseFloat(newXEnd).toFixed(3);

    const updatedPayload = { ...lastSolvePayload, x_end: newXEnd };

    fetch('/simulate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(updatedPayload)
    })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success' && data.results && data.results.length > 0) {
                const equationDisplay = data.equation || data.parsed_expression || 'Solution';
                plotSolution(data.results, data.method, equationDisplay);
                calculateStatistics(data.results);

                if (slopeFieldEnabled) {
                    setTimeout(() => updateSlopeField(), 100);
                }
            } else {
                console.error('Slider update failed:', data);
            }
        })
        .catch(error => {
            console.error('Slider update error:', error);
        });
}

function handleMethodChange() {
    const method = document.getElementById('method').value;
    const eulerInputs = document.querySelector('.euler-inputs');
    const directInputs = document.querySelector('.direct-inputs');
    const separationInputs = document.querySelector('.separation-inputs');
    const integratingInputs = document.querySelector('.integrating-inputs');
    const substitutionInputs = document.querySelector('.substitution-inputs');
    const methodDesc = document.getElementById('method-desc');

    methodDesc.textContent = getMethodDescription(method);

    eulerInputs.style.display = 'none';
    directInputs.style.display = 'none';
    separationInputs.style.display = 'none';
    integratingInputs.style.display = 'none';
    substitutionInputs.style.display = 'none';

    document.getElementById('equation_euler').required = false;
    document.getElementById('step_size').required = false;
    document.getElementById('equation_direct').required = false;
    document.getElementById('g_x').required = false;
    document.getElementById('h_y').required = false;
    document.getElementById('p_x').required = false;
    document.getElementById('q_x').required = false;
    document.getElementById('equation_sub').required = false;
    document.getElementById('substitution_var').required = false;

    if (method === 'euler') {
        eulerInputs.style.display = 'block';
        document.getElementById('equation_euler').required = true;
        document.getElementById('step_size').required = true;
    } else if (method === 'direct_integration') {
        directInputs.style.display = 'block';
        document.getElementById('equation_direct').required = true;
    } else if (method === 'separation') {
        separationInputs.style.display = 'block';
        document.getElementById('g_x').required = true;
        document.getElementById('h_y').required = true;
    } else if (method === 'integrating_factor') {
        integratingInputs.style.display = 'block';
        document.getElementById('p_x').required = true;
        document.getElementById('q_x').required = true;
    } else if (method === 'substitution') {
        substitutionInputs.style.display = 'block';
        document.getElementById('equation_sub').required = true;
        document.getElementById('substitution_var').required = true;
    }
}

function handleFormSubmit(e) {
    e.preventDefault();

    const method = document.getElementById('method').value;
    const parameters = document.getElementById('parameters').value;
    const x0 = document.getElementById('x0').value;
    const y0 = document.getElementById('y0').value;
    const x_end = document.getElementById('x_end').value;
    const eval_points = document.getElementById('eval_points').value;

    let payload = {
        method: method,
        parameters: parameters,
        x0: x0,
        y0: y0,
        x_end: x_end,
        eval_points: eval_points
    };

    if (method === 'euler') {
        payload.equation = document.getElementById('equation_euler').value;
        payload.step_size = document.getElementById('step_size').value;
    } else if (method === 'direct_integration') {
        payload.equation = document.getElementById('equation_direct').value;
    } else if (method === 'separation') {
        payload.g_x = document.getElementById('g_x').value;
        payload.h_y = document.getElementById('h_y').value;
    } else if (method === 'integrating_factor') {
        payload.p_x = document.getElementById('p_x').value;
        payload.q_x = document.getElementById('q_x').value;
    } else if (method === 'substitution') {
        payload.equation = document.getElementById('equation_sub').value;
        payload.substitution = document.getElementById('substitution_var').value;
    }

    lastSolvePayload = payload;

    const resultDiv = document.getElementById('result');
    resultDiv.style.display = 'block';
    resultDiv.className = 'result-area processing';
    resultDiv.innerHTML = '<p>🔄 Processing...</p>';

    fetch('/simulate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
    })
        .then(response => {
            if (!response.ok) {
                return response.text().then(text => {
                    throw new Error(`Server returned ${response.status}: ${text.substring(0, 100)}...`);
                });
            }
            return response.json().then(data => ({
                status: response.status,
                data: data
            }));
        })
        .then(({ status, data }) => {
            if (status === 200 && data.status === 'success') {
                let html = `
                <div class="success-header">
                    <h3>✓ Configuration Successful</h3>
                </div>
                <div class="result-content">
                    <div class="result-row">
                        <strong>Equation:</strong> dy/dx = ${escapeHtml(data.equation)}
                    </div>`;

                if (data.parameters && Object.keys(data.parameters).length > 0) {
                    const paramStr = Object.entries(data.parameters)
                        .map(([k, v]) => `${k}=${v}`)
                        .join(', ');
                    html += `
                    <div class="result-row">
                        <strong>Parameters:</strong> ${escapeHtml(paramStr)}
                    </div>`;
                }

                html += `
                    <div class="result-row">
                        <strong>Parsed:</strong> dy/dx = ${escapeHtml(data.parsed_expression)}
                    </div>`;

                if (data.analytical_formula) {
                    html += `
                    <div class="result-row analytical-solution">
                        <strong>📐 Exact Solution:</strong> y(x) = ${escapeHtml(data.analytical_formula)}
                    </div>`;
                }

                html += `
                    <div class="result-row">
                        <strong>Method:</strong> ${data.method}
                    </div>
                    <div class="result-row">
                        <strong>Initial Condition:</strong> (${data.x0}, ${data.y0})
                    </div>
                    <div class="result-row">
                        <strong>Domain End:</strong> x_end = ${data.x_end}
                    </div>`;

                if (data.step_size !== undefined) {
                    html += `
                    <div class="result-row">
                        <strong>Step Size:</strong> h = ${data.step_size}
                    </div>`;
                }

                if (data.eval_points !== undefined) {
                    html += `
                    <div class="result-row">
                        <strong>Evaluation Points:</strong> ${data.eval_points}
                    </div>`;
                }

                html += `
                </div>
            `;
                resultDiv.className = 'result-area success';
                resultDiv.innerHTML = html;

                if (data.results && data.results.length > 0) {
                    document.getElementById('graph-container').style.display = 'block';

                    const equationDisplay = data.equation || data.parsed_expression || 'Solution';
                    plotSolution(data.results, data.method, equationDisplay);
                    calculateStatistics(data.results);

                    const currentMethodKey = document.getElementById('method').value;
                    const slopeBtn = document.getElementById('slope-field-btn');

                    if (currentMethodKey === 'euler' || currentMethodKey === 'substitution') {
                        currentEquationForSlope = equationDisplay;
                        slopeBtn.style.display = 'block';

                        const slopePanel = document.getElementById('slope-field-controls');
                        slopePanel.style.display = 'block';

                        slopeBtn.textContent = '📐 Hide Slope Field';
                        slopeBtn.style.background = 'linear-gradient(135deg, #FF4444 0%, #CC0000 100%)';

                    } else {
                        currentEquationForSlope = null;
                        slopeBtn.style.display = 'none';
                        document.getElementById('slope-field-controls').style.display = 'none';
                    }

                    enableY0Slider(data.y0);
                    enableXEndSlider(data.x_end, data.x0);

                    showToast('✓ Solution computed successfully!', 'success');
                }
            } else {
                const errorType = data.error_type || 'unknown';
                let errorIcon = '✗';

                if (errorType === 'validation' || errorType === 'parse') {
                    errorIcon = '⚠';
                }

                resultDiv.className = 'result-area error';
                resultDiv.innerHTML = `
                <div class="error-header">
                    <h3>${errorIcon} Error</h3>
                </div>
                <p>${escapeHtml(data.message)}</p>
            `;
            }
        })
        .catch(error => {
            resultDiv.className = 'result-area error';
            resultDiv.innerHTML = `
            <div class="error-header">
                <h3>✗ Connection Error</h3>
            </div>
            <p>Failed to connect to server. Details: ${escapeHtml(error.message)}</p>
        `;
            console.error('Error:', error);
        });
}

document.addEventListener('DOMContentLoaded', function () {
    const exportBtn = document.getElementById('export-csv-btn');
    if (exportBtn) {
        exportBtn.addEventListener('click', exportToCSV);
    }

    const copyBtn = document.getElementById('copy-data-btn');
    if (copyBtn) {
        copyBtn.addEventListener('click', copyDataToClipboard);
    }

    const downloadBtn = document.getElementById('download-graph-btn');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', downloadGraphAsPNG);
    }

    loadSlopeFieldPreferences();

    document.getElementById('method').addEventListener('change', handleMethodChange);
    document.getElementById('method').dispatchEvent(new Event('change'));

    document.getElementById('visualizer-form').addEventListener('submit', handleFormSubmit);

    document.addEventListener('keydown', function (e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            document.getElementById('visualizer-form').dispatchEvent(new Event('submit'));
            showToast('Solving equation...', 'info');
        }
    });
});

const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(400px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(400px); opacity: 0; }
    }
`;
document.head.appendChild(style);
