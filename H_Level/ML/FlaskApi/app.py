import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import os 
import mlflow
from flask import Flask, request,render_template, json
import joblib


from ML_Utils import *

app = Flask(__name__)

@app.route('/',methods=["Get","POST"])
def home():
    return render_template("index.html")

@app.route('/predict',methods=["Get","POST"])
def predict():
    try:
        uploaded_file = request.files['file']
        df = pd.read_csv(uploaded_file)
        df_tranformed = data_validation(df)
        os.chdir("/data1")
    
        
        mlflow.set_tracking_uri("...")
#       I have removed URI for privacy concerns.
    
        experiment_name = "Book_Impact_Prediction_1"
    
        runs = mlflow.search_runs(experiment_ids=mlflow.get_experiment_by_name(experiment_name).experiment_id)
    
        runs = runs.sort_values(['metrics.Mean Absolute Percentage Error'], ascending = [False]).reset_index().drop(['index'], axis = 1)
    
        best_run = runs.loc[runs["metrics.Mean Absolute Percentage Error"].idxmax()]
    
        best_model = mlflow.pyfunc.load_model(best_run.artifact_uri + "/model")
    
        predicted = best_model.predict(df_tranformed)
    
        with open('/data1/notebooks/H_Level/ML/Scaler Pickle/scaler.pkl', 'rb') as f:
            loaded_scaler = pickle.load(f)
    
        # Use the loaded scaler to transform data
        predicted = loaded_scaler.inverse_transform(pd.DataFrame(predicted))
    
        df['Book_Impact'] = pd.DataFrame(predicted)
        
        return df.to_json(orient="split")
        
    except Exception as e:
        print(e)
        print("Opps, seems like a error, Please upload the file again.")

def data_validation(df):
    
    # Pre Processing of Data

    data_preprocessed = general_preprocessing(df)
    data_preprocessed.removesquarebracket(columns = ['authors', 'categories'])
    data_preprocessed.date_processing(columns = ['publishedDate'])
    data_preprocessed.dropna()

    df_tranformed = TextVectorizer(64)
    df_tranformed = df_tranformed.fit_transform(df, text_columns = ['Title', 'description', 'authors', 'publisher', 'categories'], numerical_column = ['publishedDate'])

    return df_tranformed

if __name__ == '__main__':
     app.run(debug=True, host = '....', port=5003)


# I have removed host name for privacy concerns.

 
