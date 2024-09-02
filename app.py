from flask import Flask, request, jsonify, make_response
import pickle

app = Flask(__name__)

# Load the models
with open("diabetese_trained_model.pkl", 'rb') as f:
    dia_model = pickle.load(f)

with open("heart_trained_model.pkl", 'rb') as f:
    heart_model = pickle.load(f)

def add_cors_headers(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

# Diabetes Prediction API
@app.route('/predict_diabetes', methods=['POST', 'OPTIONS'])
def predict_diabetes():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response())
    
    data = request.json
    features = [
        data['Pregnancies'], data['Glucose'], data['BloodPressure'],
        data['SkinThickness'], data['Insulin'], data['BMI'],
        data['DiabetesPedigreeFunction'], data['Age']
    ]
    
    prediction = dia_model.predict([features])
    result = 'yes' if prediction[0] == 1 else 'no'
    
    response = jsonify({"diabetes_prediction": result})
    return add_cors_headers(response)

# Heart Disease Prediction API
@app.route('/predict_heart', methods=['POST', 'OPTIONS'])
def predict_heart():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response())
    
    data = request.json
    features = [
        data['age'], data['sex'], data['cp'], data['trestbps'],
        data['chol'], data['fbs'], data['restecg'], data['thalach'],
        data['exang'], data['oldpeak'], data['slope'], data['ca'], data['thal']
    ]
    
    prediction = heart_model.predict([features])
    result = 'yes' if prediction[0] == 1 else 'no'
    
    response = jsonify({"heart_disease_prediction": result})
    return add_cors_headers(response)

if __name__ == '__main__':
    app.run(debug=True)
