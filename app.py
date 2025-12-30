from flask import Flask, render_template, request, jsonify
from sympy import symbols, sympify, lambdify, sin, cos, exp, tan, log, sqrt, integrate, dsolve, Function, Eq, Derivative, nsolve, solve
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import numpy as np
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
            'expression': str(expr),
            'symbolic': expr
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
        try:
            val = f(x, y)
            if not np.isfinite(val):
                break
                
            y_new = y + h * val
            if not np.isfinite(y_new):
                break
                
            y = y_new
            x = x + h
            
            x_values.append(x)
            y_values.append(y)
        except:
            break
    
    results = [{'x': round(x_val, 6), 'y': round(y_val, 6)} 
               for x_val, y_val in zip(x_values, y_values)]
    return results

def improved_euler_method(f, x0, y0, x_end, h):
    x_values = [x0]
    y_values = [y0]
    
    x = x0
    y = y0
    
    while x < x_end:
        try:
            k1 = f(x, y)
            if not np.isfinite(k1):
                break
            
            y_predict = y + h * k1
            k2 = f(x + h, y_predict)
            if not np.isfinite(k2):
                break
            
            y_new = y + (h / 2) * (k1 + k2)
            if not np.isfinite(y_new):
                break
            
            y = y_new
            x = x + h
            
            x_values.append(x)
            y_values.append(y)
        except:
            break
    
    results = [{'x': round(x_val, 6), 'y': round(y_val, 6)} 
               for x_val, y_val in zip(x_values, y_values)]
    return results

def rk4_method(f, x0, y0, x_end, h):
    x_values = [x0]
    y_values = [y0]
    
    x = x0
    y = y0
    
    while x < x_end:
        try:
            k1 = f(x, y)
            if not np.isfinite(k1):
                break
            
            k2 = f(x + h/2, y + (h/2) * k1)
            if not np.isfinite(k2):
                break
            
            k3 = f(x + h/2, y + (h/2) * k2)
            if not np.isfinite(k3):
                break
            
            k4 = f(x + h, y + h * k3)
            if not np.isfinite(k4):
                break
            
            y_new = y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
            if not np.isfinite(y_new):
                break
            
            y = y_new
            x = x + h
            
            x_values.append(x)
            y_values.append(y)
        except:
            break
    
    results = [{'x': round(x_val, 6), 'y': round(y_val, 6)} 
               for x_val, y_val in zip(x_values, y_values)]
    return results

def direct_integration_method(f_x_str, x0, y0, x_end, num_points):
    x_values = np.linspace(x0, x_end, int(num_points))
    y_values = []
    solution_str = None
    
    try:
        x = symbols('x')
        f_expr = parse_expr(f_x_str, local_dict={'x': x, 'sin': sin, 'cos': cos, 'exp': exp, 'tan': tan, 'log': log, 'sqrt': sqrt})
        
        antiderivative = integrate(f_expr, x)
        C = y0 - float(antiderivative.subs(x, x0))
        solution = antiderivative + C
        solution_str = str(solution)
        
        solution_func = lambdify(x, solution, modules=['numpy'])
        y_values = [float(solution_func(xi)) for xi in x_values]
        
    except Exception as e:
        print(f"Direct integration error: {e}")
        y_values = [y0] * len(x_values)
    
    return x_values.tolist(), y_values, solution_str

def separation_of_variables_method(g_x_str, h_y_str, x0, y0, x_end, num_points):
    x_values = np.linspace(x0, x_end, int(num_points))
    y_values = []
    implicit_solution = None
    
    try:
        x, y = symbols('x y')
        g_expr = parse_expr(g_x_str, local_dict={'x': x, 'sin': sin, 'cos': cos, 'exp': exp, 'tan': tan, 'log': log, 'sqrt': sqrt})
        h_expr = parse_expr(h_y_str, local_dict={'y': y, 'sin': sin, 'cos': cos, 'exp': exp, 'tan': tan, 'log': log, 'sqrt': sqrt})
        
        g_int = integrate(g_expr, x)
        h_int = integrate(1/h_expr, y)
        
        C = float(h_int.subs(y, y0)) - float(g_int.subs(x, x0))
        
        implicit_solution = f"∫(1/{h_y_str})dy = ∫({g_x_str})dx + C, where {h_int} = {g_int} + {C:.6f}"
        
        current_y = y0
        for xi in x_values:
            try:
                implicit_eq = h_int - g_int.subs(x, xi) - C
                y_solution = nsolve(implicit_eq, current_y)
                y_values.append(float(y_solution))
                current_y = float(y_solution)
            except:
                y_values.append(y_values[-1] if y_values else y0)
        
    except Exception as e:
        print(f"Separation error: {e}")
        y_values = [y0] * len(x_values)
    
    return x_values.tolist(), y_values, implicit_solution

def integrating_factor_method(p_x_str, q_x_str, x0, y0, x_end, num_points):
    x_values = np.linspace(x0, x_end, int(num_points))
    y_values = []
    solution_str = None
    
    try:
        x, y = symbols('x y')
        P = parse_expr(p_x_str, local_dict={'x': x, 'sin': sin, 'cos': cos, 'exp': exp, 'tan': tan, 'log': log, 'sqrt': sqrt})
        Q = parse_expr(q_x_str, local_dict={'x': x, 'sin': sin, 'cos': cos, 'exp': exp, 'tan': tan, 'log': log, 'sqrt': sqrt})
        
        mu = exp(integrate(P, x))
        
        integral_term = integrate(mu * Q, x)
        C = (y0 * mu.subs(x, x0) - integral_term.subs(x, x0))
        
        solution = (integral_term + C) / mu
        solution_str = str(solution)
        solution_func = lambdify(x, solution, modules=['numpy'])
        
        y_values = [float(solution_func(xi)) for xi in x_values]
        
    except Exception as e:
        print(f"Integrating factor error: {e}")
        y_values = [y0] * len(x_values)
    
    return x_values.tolist(), y_values, solution_str

def substitution_method(parse_result, x0, y0, x_end, num_points):
    try:
        x, y = symbols('x y')
        Y = Function('y')

        ode_expr = parse_result.get('symbolic')
        if ode_expr is not None:
            ode = Eq(Derivative(Y(x), x), ode_expr)
            general_solution = dsolve(ode, ics={Y(x0): y0})

            solution_expr = None
            if hasattr(general_solution, 'rhs'):
                solution_expr = general_solution.rhs
            elif hasattr(general_solution, 'function'):
                solution_expr = general_solution.function

            if solution_expr is not None:
                solution_func = lambdify(x, solution_expr, modules=['numpy'])
                x_values = np.linspace(x0, x_end, int(num_points))
                y_values = [float(solution_func(xi)) for xi in x_values]
                return x_values.tolist(), y_values, str(solution_expr)
    except Exception as e:
        print(f"Substitution method symbolic solve failed: {e}")

    f = parse_result.get('function')
    f_lambda = lambda x_val, y_val: f(x_val, y_val) if callable(f) else 0
    x_vals, y_vals = euler_method(f_lambda, x0, y0, x_end, (x_end - x0) / num_points)
    return x_vals, y_vals, None


METHODS = {
    'euler': {'name': 'Euler Method', 'function': euler_method},
    'improved_euler': {'name': 'Improved Euler Method', 'function': improved_euler_method},
    'rk4': {'name': 'Runge-Kutta (RK4)', 'function': rk4_method},
    'direct_integration': {'name': 'Direct Integration', 'function': direct_integration_method},
    'separation': {'name': 'Separation of Variables', 'function': separation_of_variables_method},
    'integrating_factor': {'name': 'Integrating Factor', 'function': integrating_factor_method},
    'substitution': {'name': 'Substitution Method', 'function': substitution_method}
}

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/solver')
def solver():
    return render_template('solver.html')

@app.route('/quick-solver')
def quick_solver():
    return render_template('quick-solver.html')

@app.route('/advanced')
def advanced():
    return render_template('advanced.html')

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    data = request.get_json()
    
    method_key = data.get('method', 'euler').strip()
    param_str = data.get('parameters', '').strip()
    x0 = data.get('x0', '')
    y0 = data.get('y0', '')
    x_end = data.get('x_end', '')
    eval_points = data.get('eval_points', '')
    
    equation_str = data.get('equation', '').strip()
    step_size = data.get('step_size', '')
    g_x = data.get('g_x', '').strip()
    h_y = data.get('h_y', '').strip()
    p_x = data.get('p_x', '').strip()
    q_x = data.get('q_x', '').strip()
    substitution = data.get('substitution', '').strip()
    
    if method_key not in METHODS:
        return jsonify({
            'status': 'error',
            'message': f'Invalid method: {method_key}. Valid methods are: {", ".join(METHODS.keys())}',
            'error_type': 'validation'
        }), 400
    
    parameters, param_error = parse_parameters(param_str)
    if param_error:
        return jsonify({
            'status': 'error',
            'message': param_error,
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
        if x_end == '':
            return jsonify({
                'status': 'error',
                'message': 'Domain end (x_end) is required',
                'error_type': 'validation'
            }), 400
        
        x_end_val = float(x_end)
        
        if x_end_val <= x0_val:
            return jsonify({
                'status': 'error',
                'message': f'Domain end must be greater than x₀. Got x_end={x_end_val}, x0={x0_val}',
                'error_type': 'validation'
            }), 400
    except ValueError:
        return jsonify({
            'status': 'error',
            'message': 'Domain end must be a valid number',
            'error_type': 'validation'
        }), 400
    
    try:
        if eval_points == '':
            return jsonify({
                'status': 'error',
                'message': 'Number of evaluation points is required',
                'error_type': 'validation'
            }), 400
        
        eval_points_val = int(eval_points)
        
        if eval_points_val < 10:
            return jsonify({
                'status': 'error',
                'message': 'Number of evaluation points must be at least 10',
                'error_type': 'validation'
            }), 400
    except ValueError:
        return jsonify({
            'status': 'error',
            'message': 'Number of evaluation points must be a valid integer',
            'error_type': 'validation'
        }), 400
    
    if method_key == 'euler':
        if equation_str == '':
            return jsonify({
                'status': 'error',
                'message': 'Equation f(x, y) is required for Euler method',
                'error_type': 'validation'
            }), 400
        
        try:
            if step_size == '':
                return jsonify({
                    'status': 'error',
                    'message': 'Step size (h) is required for Euler method',
                    'error_type': 'validation'
                }), 400
            
            step_size_val = float(step_size)
            
            if step_size_val <= 0:
                return jsonify({
                    'status': 'error',
                    'message': 'Step size must be greater than 0',
                    'error_type': 'validation'
                }), 400
            
            if step_size_val > (x_end_val - x0_val):
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
    elif method_key == 'direct_integration':
        if equation_str == '':
            return jsonify({
                'status': 'error',
                'message': 'Equation f(x) is required for direct integration',
                'error_type': 'validation'
            }), 400
    elif method_key == 'separation':
        if g_x == '' or h_y == '':
            return jsonify({
                'status': 'error',
                'message': 'Both g(x) and h(y) are required for separation of variables',
                'error_type': 'validation'
            }), 400
    elif method_key == 'integrating_factor':
        if p_x == '' or q_x == '':
            return jsonify({
                'status': 'error',
                'message': 'Both P(x) and Q(x) are required for integrating factor method',
                'error_type': 'validation'
            }), 400
    elif method_key == 'substitution':
        if equation_str == '' or substitution == '':
            return jsonify({
                'status': 'error',
                'message': 'Both equation and substitution variable are required',
                'error_type': 'validation'
            }), 400
    
    if equation_str and method_key == 'euler':
        is_valid, error_msg = validate_equation_input(equation_str, set(parameters.keys()))
        if not is_valid:
            return jsonify({
                'status': 'error',
                'message': error_msg,
                'error_type': 'validation'
            }), 400
    
    try:
        method_info = METHODS[method_key]
        method_function = method_info['function']
        method_name = method_info['name']
        
        analytical_formula = None
        
        if method_key == 'euler':
            parse_result = parse_equation(equation_str, parameters)
            if not parse_result['success']:
                return jsonify({
                    'status': 'error',
                    'message': parse_result['error'],
                    'error_type': 'parse'
                }), 400
            
            f = parse_result['function']
            results = method_function(f, x0_val, y0_val, x_end_val, step_size_val)
            x_values = [r['x'] for r in results]
            y_values = [r['y'] for r in results]
            parsed_expr = parse_result['expression']
            
        elif method_key == 'direct_integration':
            x_values, y_values, analytical_formula = method_function(equation_str, x0_val, y0_val, x_end_val, eval_points_val)
            parsed_expr = equation_str
            
        elif method_key == 'separation':
            x_values, y_values, analytical_formula = method_function(g_x, h_y, x0_val, y0_val, x_end_val, eval_points_val)
            parsed_expr = f"g(x)·h(y) = ({g_x})·({h_y})"
            
        elif method_key == 'integrating_factor':
            x_values, y_values, analytical_formula = method_function(p_x, q_x, x0_val, y0_val, x_end_val, eval_points_val)
            parsed_expr = f"y' + ({p_x})y = ({q_x})"
            
        elif method_key == 'substitution':
            parse_result = parse_equation(equation_str, parameters)
            if not parse_result['success']:
                return jsonify({
                    'status': 'error',
                    'message': parse_result['error'],
                    'error_type': 'parse'
                }), 400
            
            x_values, y_values, analytical_formula = method_function(parse_result, x0_val, y0_val, x_end_val, eval_points_val)
            parsed_expr = parse_result['expression']
        
        response = {
            'status': 'success',
            'equation': equation_str if equation_str else f'Method: {method_name}',
            'parameters': parameters,
            'parsed_expression': parsed_expr,
            'x0': x0_val,
            'y0': y0_val,
            'x_end': x_end_val,
            'method': method_name,
            'results': [{'x': float(x), 'y': float(y)} for x, y in zip(x_values, y_values)],
            'num_points': len(x_values),
            'message': f'Equation successfully solved using {method_name}',
            'analytical_formula': analytical_formula
        }
        
        if method_key == 'euler':
            response['step_size'] = step_size_val
        else:
            response['eval_points'] = eval_points_val
        
        if method_key == 'separation':
            response['g_x'] = g_x
            response['h_y'] = h_y
        elif method_key == 'integrating_factor':
            response['p_x'] = p_x
            response['q_x'] = q_x
        elif method_key == 'substitution':
            response['substitution'] = substitution
        
        return jsonify(response)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Calculation error: {str(e)}',
            'error_type': 'calculation'
        }), 400

@app.route('/advanced-analyze', methods=['POST'])
def advanced_analyze():
    data = request.get_json()
    
    equation_str = data.get('equation', '').strip()
    x0_str = data.get('x0', '')
    y0_str = data.get('y0', '')
    x_end_str = data.get('x_end', '')
    method_key = data.get('method', 'rk4').strip()
    
    param_name = data.get('param_name', '').strip()
    param_min_str = data.get('param_min', '')
    param_max_str = data.get('param_max', '')
    param_steps_str = data.get('param_steps', '')
    
    try:
        x0 = float(x0_str)
        y0 = float(y0_str)
        x_end = float(x_end_str)
    except (ValueError, TypeError):
        return jsonify({
            'status': 'error',
            'message': 'Invalid initial conditions. x0, y0, and x_end must be numbers.',
            'error_type': 'validation'
        }), 400
    
    if x_end <= x0:
        return jsonify({
            'status': 'error',
            'message': f'Domain end must be greater than x₀. Got x_end={x_end}, x0={x0}',
            'error_type': 'validation'
        }), 400
    
    if not equation_str:
        return jsonify({
            'status': 'error',
            'message': 'Equation is required',
            'error_type': 'validation'
        }), 400
    
    if method_key not in METHODS:
        return jsonify({
            'status': 'error',
            'message': f'Invalid method: {method_key}',
            'error_type': 'validation'
        }), 400
    
    results = {
        'status': 'success',
        'equation': equation_str,
        'x0': x0,
        'y0': y0,
        'x_end': x_end,
        'method': METHODS[method_key]['name']
    }
    
    if param_name and param_min_str and param_max_str and param_steps_str:
        try:
            param_min = float(param_min_str)
            param_max = float(param_max_str)
            param_steps = int(param_steps_str)
            
            if param_steps < 2 or param_steps > 10:
                return jsonify({
                    'status': 'error',
                    'message': 'Parameter steps must be between 2 and 10',
                    'error_type': 'validation'
                }), 400
            
            if param_max <= param_min:
                return jsonify({
                    'status': 'error',
                    'message': 'Parameter maximum must be greater than minimum',
                    'error_type': 'validation'
                }), 400
            
            if not re.match(r'^[a-zA-Z]$', param_name):
                return jsonify({
                    'status': 'error',
                    'message': 'Parameter name must be a single letter',
                    'error_type': 'validation'
                }), 400
            
            param_values = np.linspace(param_min, param_max, param_steps)
            param_datasets = []
            
            for param_value in param_values:
                params = {param_name: param_value}
                
                is_valid, validation_msg = validate_equation_input(equation_str, param_names=set(params.keys()))
                if not is_valid:
                    return jsonify({
                        'status': 'error',
                        'message': validation_msg,
                        'error_type': 'validation'
                    }), 400
                
                try:
                    parse_result = parse_equation(equation_str, parameters=params)
                    if not parse_result.get('success'):
                        return jsonify({
                            'status': 'error',
                            'message': parse_result.get('error', 'Failed to parse equation'),
                            'error_type': 'parsing'
                        }), 400
                    f = parse_result['function']
                except Exception as e:
                    return jsonify({
                        'status': 'error',
                        'message': f'Failed to parse equation: {str(e)}',
                        'error_type': 'parsing'
                    }), 400
                
                try:
                    if method_key == 'euler':
                        step_size = (x_end - x0) / 100
                        solution = euler_method(f, x0, y0, x_end, step_size)
                    elif method_key == 'improved_euler':
                        step_size = (x_end - x0) / 100
                        solution = improved_euler_method(f, x0, y0, x_end, step_size)
                    elif method_key == 'rk4':
                        step_size = (x_end - x0) / 100
                        solution = rk4_method(f, x0, y0, x_end, step_size)
                    else:
                        solution = METHODS[method_key]['function'](f, x0, y0, x_end, None)
                    
                    param_datasets.append({
                        'param_value': round(param_value, 4),
                        'results': solution
                    })
                except Exception as e:
                    return jsonify({
                        'status': 'error',
                        'message': f'Calculation failed for {param_name}={param_value}: {str(e)}',
                        'error_type': 'calculation'
                    }), 400
            
            results['parameter_variation'] = {
                'param_name': param_name,
                'param_min': param_min,
                'param_max': param_max,
                'param_steps': param_steps,
                'datasets': param_datasets
            }
        except ValueError as e:
            return jsonify({
                'status': 'error',
                'message': f'Invalid parameter values: {str(e)}',
                'error_type': 'validation'
            }), 400
    
    step_sizes_str = data.get('step_sizes', '').strip()
    if step_sizes_str:
        try:
            step_size_values = [float(s.strip()) for s in step_sizes_str.split(',')]
            
            if not all(h > 0 for h in step_size_values):
                return jsonify({
                    'status': 'error',
                    'message': 'All step sizes must be positive',
                    'error_type': 'validation'
                }), 400
            
            if len(step_size_values) < 2:
                return jsonify({
                    'status': 'error',
                    'message': 'At least 2 step sizes required for comparison',
                    'error_type': 'validation'
                }), 400
            
            step_size_values_sorted = sorted(step_size_values)
            
            is_valid, validation_msg = validate_equation_input(equation_str)
            if not is_valid:
                return jsonify({
                    'status': 'error',
                    'message': validation_msg,
                    'error_type': 'validation'
                }), 400
            
            try:
                parse_result = parse_equation(equation_str, parameters={})
                if not parse_result.get('success'):
                    return jsonify({
                        'status': 'error',
                        'message': parse_result.get('error', 'Failed to parse equation'),
                        'error_type': 'parsing'
                    }), 400
                f = parse_result['function']
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Failed to parse equation: {str(e)}',
                    'error_type': 'parsing'
                }), 400
            
            step_size_datasets = []
            all_solutions = []
            
            for idx, h in enumerate(step_size_values_sorted):
                try:
                    if method_key == 'euler':
                        solution = euler_method(f, x0, y0, x_end, h)
                    elif method_key == 'improved_euler':
                        solution = improved_euler_method(f, x0, y0, x_end, h)
                    elif method_key == 'rk4':
                        solution = rk4_method(f, x0, y0, x_end, h)
                    else:
                        return jsonify({
                            'status': 'error',
                            'message': 'Step-size comparison only supports numerical methods',
                            'error_type': 'validation'
                        }), 400
                    
                    all_solutions.append(solution)
                    
                    step_size_datasets.append({
                        'step_size': h,
                        'num_points': len(solution),
                        'results': solution
                    })
                except Exception as e:
                    return jsonify({
                        'status': 'error',
                        'message': f'Calculation failed for step size h={h}: {str(e)}',
                        'error_type': 'calculation'
                    }), 400
            
            reference_solution = all_solutions[0]
            
            error_metrics = []
            
            for idx, dataset in enumerate(step_size_datasets):
                if idx == 0:
                    error_metrics.append({
                        'step_size': dataset['step_size'],
                        'max_error': 0.0,
                        'mean_error': 0.0,
                        'convergence_rate': None
                    })
                    continue
                
                current_solution = dataset['results']
                errors = []
                
                for point in current_solution:
                    ref_point = min(reference_solution, 
                                   key=lambda p: abs(p['x'] - point['x']))
                    error = abs(point['y'] - ref_point['y'])
                    errors.append(error)
                
                max_error = max(errors) if errors else 0.0
                mean_error = sum(errors) / len(errors) if errors else 0.0
                
                convergence_rate = None
                if idx >= 2:
                    prev_mean_error = error_metrics[idx - 1]['mean_error']
                    prev_step_size = error_metrics[idx - 1]['step_size']
                    h_ratio = prev_step_size / dataset['step_size']
                    if prev_mean_error > 0 and mean_error > 0:
                        convergence_rate = np.log(prev_mean_error / mean_error) / np.log(h_ratio)
                
                error_metrics.append({
                    'step_size': dataset['step_size'],
                    'max_error': round(max_error, 8),
                    'mean_error': round(mean_error, 8),
                    'convergence_rate': round(convergence_rate, 2) if convergence_rate else None
                })
            
            results['step_size_comparison'] = {
                'step_sizes': step_size_values_sorted,
                'datasets': step_size_datasets,
                'error_metrics': error_metrics
            }
        except ValueError as e:
            return jsonify({
                'status': 'error',
                'message': f'Invalid parameter values: {str(e)}',
                'error_type': 'validation'
            }), 400
    
    equilibrium_points = []
    show_stability = data.get('show_stability', False)
    if show_stability:
        try:
            x, y = symbols('x y')
            
            transformations = standard_transformations + (implicit_multiplication_application,)
            local_dict = {
                'x': x, 'y': y,
                'sin': sin, 'cos': cos, 'exp': exp,
                'tan': tan, 'log': log, 'sqrt': sqrt
            }
            
            expr = parse_expr(equation_str, local_dict=local_dict, transformations=transformations)
            
            equilibrium_points = []
            try:
                eq_solutions = solve(expr, y)
                for sol in eq_solutions:
                    if sol.is_real or sol.has(x):
                        try:
                            derivative = expr.diff(y)
                            stability_type = 'unknown'
                            
                            deriv_at_eq = derivative.subs(y, sol)
                            
                            if deriv_at_eq.is_number:
                                if deriv_at_eq < 0:
                                    stability_type = 'stable'
                                elif deriv_at_eq > 0:
                                    stability_type = 'unstable'
                                else:
                                    stability_type = 'neutral'
                            
                            equilibrium_points.append({
                                'y_value': str(sol),
                                'derivative': str(deriv_at_eq),
                                'stability': stability_type
                            })
                        except Exception:
                            equilibrium_points.append({
                                'y_value': str(sol),
                                'derivative': 'N/A',
                                'stability': 'unknown'
                            })
            except Exception:
                pass
            
            results['stability_analysis'] = {
                'equilibrium_points': equilibrium_points,
                'equation_derivative': str(expr.diff(y)) if expr else None
            }
        except Exception as e:
            results['stability_analysis'] = {
                'error': f'Stability analysis failed: {str(e)}',
                'equilibrium_points': []
            }
    
    show_phase_portrait = data.get('show_phase_portrait', False)
    if show_phase_portrait and method_key in ['euler', 'improved_euler', 'rk4']:
        try:
            parse_result = parse_equation(equation_str, parameters={})
            if parse_result.get('success'):
                f = parse_result['function']
                
                y0_base = float(data.get('y0', 1))
                x_mid = (x0 + x_end) / 2
                x_quarter = x0 + (x_end - x0) / 4
                
                y_range = max(abs(y0_base) * 3, 5)
                y_min = -y_range if y0_base < 0 else y0_base - y_range
                y_max = y_range if y0_base < 0 else y0_base + y_range
                
                starting_points = [
                    (x0, y0_base * 0.3),
                    (x0, y0_base * 0.7),
                    (x0, y0_base),
                    (x0, y0_base * 1.3),
                    (x0, y0_base * 1.7),
                    (x_quarter, y_min + (y_max - y_min) * 0.3),
                    (x_quarter, y_min + (y_max - y_min) * 0.7),
                ]
                
                trajectories = []
                for idx, (x_start, y_start) in enumerate(starting_points):
                    if method_key == 'euler':
                        traj = euler_method(f, x_start, y_start, x_end, 0.01)
                    elif method_key == 'improved_euler':
                        traj = improved_euler_method(f, x_start, y_start, x_end, 0.01)
                    elif method_key == 'rk4':
                        traj = rk4_method(f, x_start, y_start, x_end, 0.01)
                    
                    if len(traj) > 1:
                        trajectories.append({
                            'x0': round(x_start, 2),
                            'y0': round(y_start, 3),
                            'points': traj
                        })
                
                results['phase_portrait'] = {
                    'trajectories': trajectories,
                    'x_range': [x0, x_end],
                    'equilibrium_points': equilibrium_points if show_stability else []
                }
        except Exception as e:
            results['phase_portrait'] = {
                'error': f'Phase portrait generation failed: {str(e)}',
                'trajectories': []
            }
    
    return jsonify(results), 200

if __name__ == '__main__':
    app.run(debug=True)


