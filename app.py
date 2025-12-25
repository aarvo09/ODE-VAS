from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    data = request.get_json()
    
    equation = data.get('equation', '')
    x0 = data.get('x0', '')
    y0 = data.get('y0', '')

    response = {
        'status': 'received',
        'equation': equation,
        'x0': x0,
        'y0': y0
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
s