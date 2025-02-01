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
    
    print(data)

    stats = [data['testYear']] + data['stats']

    prediction, image_base64 = test_model.main(stats)
    return jsonify({'prediction': prediction, 'image' : image_base64})

if __name__ == '__main__':
    app.run(debug=True)


