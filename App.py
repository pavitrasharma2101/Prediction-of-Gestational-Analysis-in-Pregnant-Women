from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name)

# Load your trained machine learning model (change the filename accordingly).
with open('gestational_diabetes_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        pregnancy = float(request.form['pregnancy'])
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        bmi = float(request.form['bmi'])
        heredity = float(request.form['heredity'])

        # Prepare data for prediction
        input_data = pd.DataFrame({
            'Age': [age],
            'Pregnancy No': [pregnancy],
            'Weight': [weight],
            'Height': [height],
            'BMI': [bmi],
            'Heredity': [heredity]
        })

        # Perform prediction using the loaded model
        prediction = model.predict(input_data)

        result_message = "High Risk" if prediction[0] == 1 else "Low Risk"

        return render_template('index.html', prediction_result=result_message)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
