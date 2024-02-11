import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import pickle

class DataOverview:
    def __init__(self, data):
        self.data = data

    def get_data_head(self, n=20):
        """Prints the first n rows of the data."""
        print(self.data.head(n))
        print('\n')

    def get_column_names(self):
        """Prints the column names of the data."""
        print('column names')
        print(self.data.columns.values)
        print('\n')

    def get_data_types(self):
        """Prints the data types of each column."""
        print('data type of column')
        print(self.data.dtypes)
        print('\n')

    def describe_numerical(self):
        """Prints summary statistics for numerical columns."""
        print('describe numerical')
        print(self.data.describe())
        print('\n')

    def describe_categorical(self):
        """Prints summary statistics for categorical columns."""
        print('describe categorical')
        print(self.data.describe(include=['O']))
        print('\n')

    def get_data_info(self):
        """Prints information about the data."""
        print('info')
        print(self.data.info())
        print('\n')

    def get_null_count(self):
        """Prints the count of null values for each column."""
        print('null_count')
        print(self.data.isnull().sum()*100/len(self.data))
        print('\n')
        
#--------------------Pre Processing of Data----------------------------------# 

class general_preprocessing:
    
    def __init__(self, data):
        self.data = data
        
    def removesquarebracket(self, columns):
        """remove square brackets"""
        for col in columns:
            self.data[col] = self.data[col].str.replace("[", "").str.replace("]", "").str.replace("'", "")
    
    def date_processing(self, columns):
        def is_integer(value):
            return bool(re.match(r'^\d+$', str(value)))
        for col in columns:
            self.data[col] = self.data[col].str[:4]
            self.data[col] = self.data[col].apply(lambda x: x if is_integer(x) else np.nan)
            self.data.dropna(inplace=True)
            self.data[col] = self.data[col].astype(int)
            
    def dropna(self):
        self.data.dropna(inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        
    def scaled(self, columns_to_scale):
        scaler = MinMaxScaler()
        for col in columns_to_scale:
            df_scaled = pd.DataFrame(scaler.fit_transform(self.data[columns_to_scale]), columns=columns_to_scale)
            # Replace the original columns in the DataFrame with the scaled versions
            self.data[columns_to_scale] = df_scaled
            with open('/data1/notebooks/H_Level/ML/Scaler Pickle/scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            
            
class TextVectorizer:
    def __init__(self, max_features=64):
        self.max_features = max_features
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features)

    def fit_transform(self, data, text_columns, numerical_column):
        tfidf_matrices = []
        for col in text_columns:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(data[col])
            tfidf_matrices.append(tfidf_matrix)
        
        X_numerical = csr_matrix(data[numerical_column]).reshape(-1, 1)

        X_combined = hstack(tfidf_matrices + [X_numerical])
        return X_combined

            