import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score
data=pd.read_csv(r"C:\Users\bandi\OneDrive\Documents\water_quality_prediction\water_potability.csv")
data.dropna(inplace=True)
selected_features=['ph',	'Hardness',	'Solids',	'Chloramines',	'Sulfate',	'Conductivity',	'Organic_carbon',	'Trihalomethanes',	'Turbidity']
x=data[selected_features]
y=data["Potability"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5)
from sklearn.impute import SimpleImputer
import numpy as np
# Create a SimpleImputer to replace missing values with the mean
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# Fit the imputer on the training data
imputer.fit(x_train)
# Transform the training and test data
x_train = imputer.transform(x_train)
x_test = imputer.transform(x_test)
model=RandomForestClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

import pandas as pd

# Define the selected features for water quality prediction
selected_features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                     'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

def predict_water_safety(ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, 
                         trihalomethanes, turbidity):
    # Create a DataFrame with input data
    input_data = pd.DataFrame([[ph, hardness, solids, chloramines, sulfate, conductivity, 
                                organic_carbon, trihalomethanes, turbidity]], 
                              columns=selected_features)

    # Assuming 'model' is already defined elsewhere
    # Predict water safety
    prediction = model.predict(input_data)
    return prediction[0]

# Take user input for water quality parameters
ph = float(input("Enter pH: "))
hardness = float(input("Enter water hardness (mg/L): "))
solids = float(input("Enter total dissolved solids (ppm): "))
chloramines = float(input("Enter chloramines concentration (ppm): "))
sulfate = float(input("Enter sulfate concentration (ppm): "))
conductivity = float(input("Enter water conductivity (µS/cm): "))
organic_carbon = float(input("Enter organic carbon concentration (ppm): "))
trihalomethanes = float(input("Enter trihalomethanes concentration (µg/L): "))
turbidity = float(input("Enter water turbidity (NTU): "))

# Predict water safety
predicted_water_safety = predict_water_safety(ph, hardness, solids, chloramines, sulfate, conductivity, 
                                               organic_carbon, trihalomethanes, turbidity)
print("Predicted water safety:", predicted_water_safety)
