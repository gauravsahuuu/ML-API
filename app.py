from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the models
with open("diabetese_trained_model.pkl", 'rb') as f:
    dia_model = pickle.load(f)

with open("heart_trained_model.pkl", 'rb') as f:
    heart_model = pickle.load(f)

# Diabetes Prediction API
@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    data = request.json
    features = [
        data['Pregnancies'], data['Glucose'], data['BloodPressure'],
        data['SkinThickness'], data['Insulin'], data['BMI'],
        data['DiabetesPedigreeFunction'], data['Age']
    ]
    
    prediction = dia_model.predict([features])
    result = 'yes' if prediction[0] == 1 else 'no'
    
    return jsonify({"diabetes_prediction": result})

# Heart Disease Prediction API
@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    data = request.json
    features = [
        data['age'], data['sex'], data['cp'], data['trestbps'],
        data['chol'], data['fbs'], data['restecg'], data['thalach'],
        data['exang'], data['oldpeak'], data['slope'], data['ca'], data['thal']
    ]
    
    prediction = heart_model.predict([features])
    result = 'yes' if prediction[0] == 1 else 'no'
    
    return jsonify({"heart_disease_prediction": result})

if __name__ == '__main__':
    app.run(debug=True)
