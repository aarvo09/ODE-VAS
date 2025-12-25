from flask import Flask, render_template, request, jsonify
from sympy import symbols, sympify, lambdify, sin, cos, exp, tan, log, sqrt
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import re

app = Flask(__name__)

# Supported functions and symbols
SUPPORTED_FUNCTIONS = {'sin', 'cos', 'exp', 'tan', 'log', 'sqrt'}
SUPPORTED_SYMBOLS = {'x', 'y'}
ALLOWED_CHARS = set('xy0123456789+-*/^(). ')

def validate_equation_input(equation_str):
    """
    Validate equation input before parsing.
    Returns (is_valid, error_message)
    """
    # Check for empty input
    if not equation_str or equation_str.strip() == '':
        return False, "Equation cannot be empty"
    
    # Check for unsupported characters (basic check)
    equation_clean = equation_str.lower()
    
    # Remove supported function names for character check
    for func in SUPPORTED_FUNCTIONS:
        equation_clean = equation_clean.replace(func, '')
    
    # Check remaining characters
    for char in equation_clean:
        if char not in ALLOWED_CHARS:
            return False, f"Unsupported character: '{char}'"
    
    # Check for unsupported function names
    # Find all potential function calls (word followed by parenthesis)
    potential_functions = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', equation_str)
    for func in potential_functions:
        if func.lower() not in SUPPORTED_FUNCTIONS:
            return False, f"Unsupported function: '{func}'. Supported functions are: {', '.join(sorted(SUPPORTED_FUNCTIONS))}"
    
    # Check for invalid symbol names (letters that aren't x, y, or part of function names)
    # Remove function names and x, y
    temp = equation_str.lower()
    for func in SUPPORTED_FUNCTIONS:
        temp = temp.replace(func, '')
    temp = temp.replace('x', '').replace('y', '')
    
    # Find remaining alphabetic characters
    remaining_letters = re.findall(r'[a-zA-Z]+', temp)
    if remaining_letters:
        return False, f"Invalid variable: '{remaining_letters[0]}'. Only 'x' and 'y' are allowed"
    
    return True, ""

def parse_equation(equation_str):
    """
    Parse user input equation string into a callable function.
    Supports: x, y, +, -, *, /, sin, cos, exp, tan, log, sqrt
    """
    try:
        x, y = symbols('x y')
        
        transformations = standard_transformations + (implicit_multiplication_application,)
        
        expr = parse_expr(equation_str, 
                         local_dict={'x': x, 'y': y, 'sin': sin, 'cos': cos, 
                                   'exp': exp, 'tan': tan, 'log': log, 'sqrt': sqrt},
                         transformations=transformations)
        
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


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    data = request.get_json()
    
    equation_str = data.get('equation', '').strip()
    x0 = data.get('x0', '')
    y0 = data.get('y0', '')
    
    
    is_valid, error_msg = validate_equation_input(equation_str)
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
    
    parse_result = parse_equation(equation_str)
    
    if not parse_result['success']:
        return jsonify({
            'status': 'error',
            'message': parse_result['error'],
            'error_type': 'parse'
        }), 400
    
    response = {
        'status': 'success',
        'equation': equation_str,
        'parsed_expression': parse_result['expression'],
        'x0': x0_val,
        'y0': y0_val,
        'message': 'Equation successfully parsed and validated'
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)


