from flask import Flask, request, jsonify, send_from_directory

import test_model  # Replace with your Python model file

app = Flask(__name__)

# Example route for serving the frontend
@app.route('/')
def index():
    return send_from_directory('', 'index.html')

# Example API route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    stats = [data['testYear']] + data['stats']

    prediction = test_model.get_model(stats)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)


