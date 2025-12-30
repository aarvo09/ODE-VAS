function handleMethodChange() {
    const method = document.getElementById('method').value;
    const eulerInputs = document.querySelectorAll('.euler-inputs');
    const directInputs = document.querySelectorAll('.direct-inputs');
    const separationInputs = document.querySelectorAll('.separation-inputs');
    const integratingInputs = document.querySelectorAll('.integrating-inputs');
    const substitutionInputs = document.querySelectorAll('.substitution-inputs');
    const methodDesc = document.getElementById('method-desc');

    methodDesc.textContent = getMethodDescription(method);


    eulerInputs.forEach(el => el.style.display = 'none');
    directInputs.forEach(el => el.style.display = 'none');
    separationInputs.forEach(el => el.style.display = 'none');
    integratingInputs.forEach(el => el.style.display = 'none');
    substitutionInputs.forEach(el => el.style.display = 'none');


    document.getElementById('equation_euler').required = false;
    document.getElementById('equation_euler').value = '';
    document.getElementById('step_size').required = false;
    document.getElementById('step_size').value = '';
    document.getElementById('equation_direct').required = false;
    document.getElementById('equation_direct').value = '';
    document.getElementById('g_x').required = false;
    document.getElementById('g_x').value = '';
    document.getElementById('h_y').required = false;
    document.getElementById('h_y').value = '';
    document.getElementById('p_x').required = false;
    document.getElementById('p_x').value = '';
    document.getElementById('q_x').required = false;
    document.getElementById('q_x').value = '';
    document.getElementById('equation_sub').required = false;
    document.getElementById('equation_sub').value = '';
    document.getElementById('substitution_var').required = false;
    document.getElementById('substitution_var').value = '';


    if (method === 'euler') {
        eulerInputs.forEach(el => el.style.display = 'block');
        document.getElementById('equation_euler').required = true;
        document.getElementById('step_size').required = true;
    } else if (method === 'direct_integration') {
        directInputs.forEach(el => el.style.display = 'block');
        document.getElementById('equation_direct').required = true;
    } else if (method === 'separation') {
        separationInputs.forEach(el => el.style.display = 'block');
        document.getElementById('g_x').required = true;
        document.getElementById('h_y').required = true;
    } else if (method === 'integrating_factor') {
        integratingInputs.forEach(el => el.style.display = 'block');
        document.getElementById('p_x').required = true;
        document.getElementById('q_x').required = true;
    } else if (method === 'substitution') {
        substitutionInputs.forEach(el => el.style.display = 'block');
        document.getElementById('equation_sub').required = true;
        document.getElementById('substitution_var').required = true;
    }
}

function handleFormSubmit(e) {
    e.preventDefault();

    console.log('Form submitted');

    const method = document.getElementById('method').value;
    console.log('Selected method:', method);

    const submitBtn = document.querySelector('.solve-btn');
    const originalBtnText = submitBtn.innerHTML;
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="material-symbols-outlined" style="font-size: 18px; vertical-align: middle; animation: spin 1s linear infinite;">progress_activity</span> Computing...';

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

    const resultDiv = document.getElementById('result');
    resultDiv.style.display = 'none';
    resultDiv.innerHTML = '';

    fetch('/simulate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
    })
        .then(response => {
            return response.json().then(data => ({
                status: response.status,
                data: data
            }));
        })
        .then(({ status, data }) => {
            if (status === 200 && data.status === 'success') {
                let html = `
                <div class="success-header">
                    <h3>Solution Complete</h3>
                </div>`;

                if (data.analytical_formula) {
                    html += `
                <div class="analytical-solution-box">
                    <h4>Exact Analytical Solution</h4>
                    <div class="formula-display">
                        y(x) = ${escapeHtml(data.analytical_formula)}
                    </div>
                    <small>Particular solution with initial conditions applied</small>
                </div>`;
                } else {
                    html += `
                <div class="numerical-note">
                    <p>ℹ️ Exact formula not available - numerical approximation provided</p>
                </div>`;
                }

                html += `
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
                    </div>
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

                if (data.g_x && data.h_y) {
                    html += `
                    <div class="result-row">
                        <strong>g(x):</strong> ${escapeHtml(data.g_x)}
                    </div>
                    <div class="result-row">
                        <strong>h(y):</strong> ${escapeHtml(data.h_y)}
                    </div>`;
                }

                if (data.p_x && data.q_x) {
                    html += `
                    <div class="result-row">
                        <strong>P(x):</strong> ${escapeHtml(data.p_x)}
                    </div>
                    <div class="result-row">
                        <strong>Q(x):</strong> ${escapeHtml(data.q_x)}
                    </div>`;
                }

                if (data.substitution) {
                    html += `
                    <div class="result-row">
                        <strong>Substitution:</strong> ${escapeHtml(data.substitution)}
                    </div>`;
                }

                html += `
                    <div class="result-row">
                        <strong>Data Points:</strong> ${data.num_points}
                    </div>
                </div>
                <div class="results-table-container">
                    <h4>${data.analytical_formula ? 'Data Table (from Analytical Formula)' : 'Numerical Solution'}</h4>
                    <table class="results-table">
                        <thead>
                            <tr>
                                <th>n</th>
                                <th>x</th>
                                <th>y</th>
                            </tr>
                        </thead>
                        <tbody>`;

                const results = data.results;
                const maxDisplay = 50;

                if (results.length <= maxDisplay) {
                    results.forEach((point, i) => {
                        html += `
                        <tr>
                            <td>${i}</td>
                            <td>${point.x.toFixed(6)}</td>
                            <td>${point.y.toFixed(6)}</td>
                        </tr>`;
                    });
                } else {
                    for (let i = 0; i < 20; i++) {
                        html += `
                        <tr>
                            <td>${i}</td>
                            <td>${results[i].x.toFixed(6)}</td>
                            <td>${results[i].y.toFixed(6)}</td>
                        </tr>`;
                    }

                    html += `
                    <tr class="ellipsis-row">
                        <td colspan="3">... (${results.length - 40} points omitted) ...</td>
                    </tr>`;

                    for (let i = results.length - 20; i < results.length; i++) {
                        html += `
                        <tr>
                            <td>${i}</td>
                            <td>${results[i].x.toFixed(6)}</td>
                            <td>${results[i].y.toFixed(6)}</td>
                        </tr>`;
                    }
                }

                html += `
                        </tbody>
                    </table>
                </div>
            `;

                resultDiv.className = 'result-area success';
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = html;
            } else {
                const errorType = data.error_type || 'unknown';
                let errorIcon = '<span class="material-symbols-outlined" style="vertical-align: middle; margin-right: 5px;">error</span>';

                if (errorType === 'validation' || errorType === 'parse') {
                    errorIcon = '<span class="material-symbols-outlined" style="vertical-align: middle; margin-right: 5px;">warning</span>';
                }

                resultDiv.className = 'result-area error';
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = `
                <div class="error-header">
                    <h3 style="display: flex; align-items: center;">${errorIcon} Error</h3>
                </div>
                <p>${escapeHtml(data.message)}</p>
            `;
            }
        })
        .catch(error => {
            resultDiv.className = 'result-area error';
            resultDiv.style.display = 'block';
            resultDiv.innerHTML = `
            <div class="error-header">
                <h3 style="display: flex; align-items: center;"><span class="material-symbols-outlined" style="vertical-align: middle; margin-right: 5px;">wifi_off</span> Connection Error</h3>
            </div>
            <p>Failed to connect to server. Please make sure the server is running.</p>
        `;
            console.error('Error:', error);
            console.error('Error:', error);
        })
        .finally(() => {
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalBtnText;
        });
}

document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('method').addEventListener('change', handleMethodChange);
    document.getElementById('method').dispatchEvent(new Event('change'));

    document.getElementById('solver-form').addEventListener('submit', handleFormSubmit);
});
