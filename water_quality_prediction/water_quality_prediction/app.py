from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the machine learning model
data = pd.read_csv(r"C:\Users\bandi\OneDrive\Documents\water_quality_prediction\water_potability.csv")
data.dropna(inplace=True)
selected_features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                     'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
x = data[selected_features]
y = data["Potability"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Define the selected features for water quality prediction
selected_features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                     'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

@app.route('/')
def index():
    return render_template('mainpage.html')

@app.route('/prediction')
def pred():
     return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        ph = float(request.form['ph'])
        hardness = float(request.form['hardness'])
        solids = float(request.form['solids'])
        chloramines = float(request.form['chloramines'])
        sulfate = float(request.form['sulfate'])
        conductivity = float(request.form['conductivity'])
        organic_carbon = float(request.form['organic_carbon'])
        trihalomethanes = float(request.form['trihalomethanes'])
        turbidity = float(request.form['turbidity'])

        # Create a DataFrame with input data
        input_data = pd.DataFrame([[ph, hardness, solids, chloramines, sulfate, conductivity, 
                                    organic_carbon, trihalomethanes, turbidity]], 
                                  columns=selected_features)

        # Predict water safety
        prediction = model.predict(input_data)
        return redirect(url_for('result', prediction=prediction))
        
@app.route('/result/<prediction>')
def result(prediction):
    print(prediction)
    return render_template('pred.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
