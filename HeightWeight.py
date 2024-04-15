import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import joblib

rawdata=pd.read_csv("D:\MLP_WEIGHT_PREDICTION_PROJECT_FILES\Height Data.csv")
X=rawdata.drop(columns=['Weight'])
y=rawdata['Weight']

model=DecisionTreeRegressor()
model.fit(X,y)

joblib.dump(model, 'D:\MLP_WEIGHT_PREDICTION_PROJECT_FILES\HeightWeightMLModel.joblib')