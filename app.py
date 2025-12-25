from flask import Flask, render_template, request, jsonify
from sympy import symbols, sympify, lambdify, sin, cos, exp, tan, log, sqrt
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import re

app = Flask(__name__)

SUPPORTED_FUNCTIONS = {'sin', 'cos', 'exp', 'tan', 'log', 'sqrt'}
SUPPORTED_SYMBOLS = {'x', 'y'}
ALLOWED_CHARS = set('xy0123456789+-*/^(). abcdefghijklmnopqrstuvwz')

def parse_parameters(param_str):
    """
    Parse parameter string like "k=0.5, a=2, b=1.5" into a dictionary.
    Returns (params_dict, error_message)
    """
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
    """
    Validate equation input before parsing.
    Returns (is_valid, error_message)
    param_names: set of allowed parameter names
    """
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
    """
    Parse user input equation string into a callable function.
    Supports: x, y, +, -, *, /, sin, cos, exp, tan, log, sqrt, and parameters
    
    Args:
        equation_str: The equation string
        parameters: Dictionary of parameter names and values to substitute
    """
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
            # Check if result is valid
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
        # Provide more user-friendly error messages
        if "unexpected EOF" in error_msg.lower():
            return {'success': False, 'error': "Incomplete expression. Check for missing parentheses or operators."}
        elif "invalid syntax" in error_msg.lower():
            return {'success': False, 'error': "Invalid syntax. Check your equation format."}
        else:
            return {'success': False, 'error': f"Parse error: {error_msg}"}


@app.route('/')
def index():
    return render_template('index.html')

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
    
    # Parse parameters first
    parameters, param_error = parse_parameters(param_str)
    if param_error:
        return jsonify({
            'status': 'error',
            'message': param_error,
            'error_type': 'validation'
        }), 400
    
    # Validate equation input with parameter names
    is_valid, error_msg = validate_equation_input(equation_str, set(parameters.keys()))
    if not is_valid:
        return jsonify({
            'status': 'error',
            'message': error_msg,
            'error_type': 'validation'
        }), 400
    
    # Validate initial conditions
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
    
    # Validate domain range
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
    
    # Validate step size
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
        'message': 'Equation successfully apply parsed and validated'
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
