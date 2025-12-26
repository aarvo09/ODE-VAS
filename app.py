from flask import Flask, render_template, request, jsonify
from sympy import symbols, sympify, lambdify, sin, cos, exp, tan, log, sqrt
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import re

app = Flask(__name__)

SUPPORTED_FUNCTIONS = {'sin', 'cos', 'exp', 'tan', 'log', 'sqrt'}
SUPPORTED_SYMBOLS = {'x', 'y'}
ALLOWED_CHARS = set('xy0123456789+-*/^(). abcdefghijklmnopqrstuvwz')

def parse_parameters(param_str):
    if not param_str or param_str.strip() == '':
        return {}, None
    
    params = {}
    try:
        param_pairs = param_str.split(',')
        for pair in param_pairs:
            pair = pair.strip()
            if '=' not in pair:
                return None, f"Invalid parameter format: '{pair}'. Use format: name=value"
            
            name, value = pair.split('=', 1)
            name = name.strip()
            value = value.strip()
            
            if not re.match(r'^[a-z]$', name):
                return None, f"Parameter name must be a single lowercase letter (a-z, excluding x,y): '{name}'"
            
            if name in ['x', 'y']:
                return None, f"Cannot use '{name}' as parameter name (reserved for variables)"
            
            try:
                params[name] = float(value)
            except ValueError:
                return None, f"Parameter value must be a number: '{value}'"
        
        return params, None
    except Exception as e:
        return None, f"Error parsing parameters: {str(e)}"

def validate_equation_input(equation_str, param_names=None):
    if param_names is None:
        param_names = set()
    
    if not equation_str or equation_str.strip() == '':
        return False, "Equation cannot be empty"
    
    equation_clean = equation_str.lower()
    
    for func in SUPPORTED_FUNCTIONS:
        equation_clean = equation_clean.replace(func, '')
    
    for param in param_names:
        equation_clean = equation_clean.replace(param.lower(), '')
    
    for char in equation_clean:
        if char not in ALLOWED_CHARS:
            return False, f"Unsupported character: '{char}'"
    
    potential_functions = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', equation_str)
    for func in potential_functions:
        if func.lower() not in SUPPORTED_FUNCTIONS:
            return False, f"Unsupported function: '{func}'. Supported functions are: {', '.join(sorted(SUPPORTED_FUNCTIONS))}"
    
    return True, ""

def parse_equation(equation_str, parameters=None):
    if parameters is None:
        parameters = {}
    
    try:
        x, y = symbols('x y')
        
        param_symbols = {name: symbols(name) for name in parameters.keys()}
        
        transformations = standard_transformations + (implicit_multiplication_application,)
        
        local_dict = {
            'x': x, 'y': y, 
            'sin': sin, 'cos': cos, 'exp': exp, 
            'tan': tan, 'log': log, 'sqrt': sqrt
        }
        local_dict.update(param_symbols)
        
        expr = parse_expr(equation_str, 
                         local_dict=local_dict,
                         transformations=transformations)
        
        for name, value in parameters.items():
            expr = expr.subs(param_symbols[name], value)
        
        f = lambdify((x, y), expr, modules=['numpy'])
        
        try:
            test_result = f(1.0, 1.0)
            if test_result is None:
                raise ValueError("Function returned None")
        except Exception as runtime_error:
            return {
                'success': False,
                'error': f"Runtime error: {str(runtime_error)}. Check your equation syntax."
            }
        
        return {
            'success': True,
            'function': f,
            'expression': str(expr)
        }
    except SyntaxError as e:
        return {
            'success': False,
            'error': f"Syntax error: Invalid mathematical expression"
        }
    except Exception as e:
        error_msg = str(e)
        if "unexpected EOF" in error_msg.lower():
            return {'success': False, 'error': "Incomplete expression. Check for missing parentheses or operators."}
        elif "invalid syntax" in error_msg.lower():
            return {'success': False, 'error': "Invalid syntax. Check your equation format."}
        else:
            return {'success': False, 'error': f"Parse error: {error_msg}"}

def euler_method(f, x0, y0, x_end, h):
    x_values = [x0]
    y_values = [y0]
    
    x = x0
    y = y0
    
    while x < x_end:
        y = y + h * f(x, y)
        x = x + h
        
        x_values.append(x)
        y_values.append(y)
    
    return x_values, y_values

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/solver')
def solver():
    return render_template('solver.html')
@app.route('/quick-solver')
def quick_solver():
    return render_template('quick-solver.html')
@app.route('/simulate', methods=['POST'])
def simulate():
    data = request.get_json()
    
    equation_str = data.get('equation', '').strip()
    param_str = data.get('parameters', '').strip()
    x0 = data.get('x0', '')
    y0 = data.get('y0', '')
    x_start = data.get('x_start', '')
    x_end = data.get('x_end', '')
    step_size = data.get('step_size', '')
    
    parameters, param_error = parse_parameters(param_str)
    if param_error:
        return jsonify({
            'status': 'error',
            'message': param_error,
            'error_type': 'validation'
        }), 400
    

    is_valid, error_msg = validate_equation_input(equation_str, set(parameters.keys()))
    if not is_valid:
        return jsonify({
            'status': 'error',
            'message': error_msg,
            'error_type': 'validation'
        }), 400
    
    try:
        if x0 == '' or y0 == '':
            return jsonify({
                'status': 'error',
                'message': 'Initial conditions x₀ and y₀ are required',
                'error_type': 'validation'
            }), 400
        
        x0_val = float(x0)
        y0_val = float(y0)
    except ValueError:
        return jsonify({
            'status': 'error',
            'message': 'Initial conditions must be valid numbers',
            'error_type': 'validation'
        }), 400
    
    try:
        if x_start == '' or x_end == '':
            return jsonify({
                'status': 'error',
                'message': 'Domain range (start x and end x) is required',
                'error_type': 'validation'
            }), 400
        
        x_start_val = float(x_start)
        x_end_val = float(x_end)
        
        if x_start_val >= x_end_val:
            return jsonify({
                'status': 'error',
                'message': 'End x must be greater than Start x',
                'error_type': 'validation'
            }), 400
    except ValueError:
        return jsonify({
            'status': 'error',
            'message': 'Domain range values must be valid numbers',
            'error_type': 'validation'
        }), 400
    
    try:
        if step_size == '':
            return jsonify({
                'status': 'error',
                'message': 'Step size (h) is required',
                'error_type': 'validation'
            }), 400
        
        step_size_val = float(step_size)
        
        if step_size_val <= 0:
            return jsonify({
                'status': 'error',
                'message': 'Step size must be greater than 0',
                'error_type': 'validation'
            }), 400
        
        if step_size_val > (x_end_val - x_start_val):
            return jsonify({
                'status': 'error',
                'message': 'Step size is too large for the given domain',
                'error_type': 'validation'
            }), 400
    except ValueError:
        return jsonify({
            'status': 'error',
            'message': 'Step size must be a valid number',
            'error_type': 'validation'
        }), 400
    
    parse_result = parse_equation(equation_str, parameters)
    
    if not parse_result['success']:
        return jsonify({
            'status': 'error',
            'message': parse_result['error'],
            'error_type': 'parse'
        }), 400
    
    try:
        f = parse_result['function']
        x_values, y_values = euler_method(f, x0_val, y0_val, x_end_val, step_size_val)
        
        results = [
            {'x': float(x), 'y': float(y)} 
            for x, y in zip(x_values, y_values)
        ]
        
        response = {
            'status': 'success',
            'equation': equation_str,
            'parameters': parameters,
            'parsed_expression': parse_result['expression'],
            'x0': x0_val,
            'y0': y0_val,
            'x_start': x_start_val,
            'x_end': x_end_val,
            'step_size': step_size_val,
            'method': 'Euler',
            'results': results,
            'num_points': len(results),
            'message': 'Equation successfully solved using Euler method'
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Calculation error: {str(e)}',
            'error_type': 'calculation'
        }), 400

if __name__ == '__main__':
    app.run(debug=True)

