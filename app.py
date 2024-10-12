from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the model and scaler
model = pickle.load(open('startup_model.pkl3', 'rb'))
scaler = pickle.load(open('scaler.pkl3', 'rb'))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.json['data']
    
    # Convert the data to numpy array and reshape
    data = np.array(data).reshape(1, -1)
    
    # Scale the data using the same scaler
    data = scaler.transform(data)
    
    # Make a prediction
    prediction = model.predict(data)
    
    # Send the result back
    return jsonify({'profit': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
    
