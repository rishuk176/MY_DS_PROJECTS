import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

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
        
    def plot_box_plot_numerical(self, numerical_cols):
        """Plots box plots for numerical columns."""
        for col in numerical_cols:
            plt.figure(figsize=(6, 4))
            self.data.boxplot(column=col)
            plt.title(f'Box Plot of {col}')
            plt.show()

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
            self.data[col] = self.data[col].apply(lambda x: x if is_integer(x) else None)
            
    def dropna(self):
        self.data.dropna(inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        
    def scaled(self, columns_to_scale):
        scaler = MinMaxScaler()
        for col in columns_to_scale:
            df_scaled = pd.DataFrame(scaler.fit_transform(self.data[columns_to_scale]), columns=columns_to_scale)
            # Replace the original columns in the DataFrame with the scaled versions
            self.data[columns_to_scale] = df_scaled

class Preprocessing_text_to_seq_padd:
    def __init__(self, data):
        self.data = data
        
    def text_to_seq_padd(self, columns):
        tokenizer = Tokenizer()
        sequence = {}
        padded = {}
        for col in columns:
            tokenizer.fit_on_texts(self.data[col])
            sequence[col] = tokenizer.texts_to_sequences(self.data[col])
            padded[col] = pad_sequences(sequence[col])
        return padded
    
class InputFeatures:
    def __init__(self, data, padded):
        self.X_title = padded['Title']
        self.X_description = padded['description']
        self.X_author = padded['authors']
        self.X_publisher = padded['publisher']
        self.X_categories = padded['categories']
        self.X_year = padded['publishedDate']
        
        self.y = data['Impact'].values
        
        self.split_data()
        
    def split_data(self):
        self.X_train_title, self.X_test_title, \
        self.X_train_description, self.X_test_description, \
        self.X_train_author, self.X_test_author, \
        self.X_train_publisher, self.X_test_publisher, \
        self.X_train_categories, self.X_test_categories, \
        self.X_train_year, self.X_test_year, \
        self.y_train, self.y_test = \
        train_test_split(self.X_title, self.X_description, 
                         self.X_author, self.X_publisher, 
                         self.X_categories, self.X_year, 
                         self.y, test_size=0.2, random_state=42)
        
class DeepLearningModel:
    
    def __init__(self, embedding_dim=64):
        self.embedding_dim = embedding_dim

    def build_model(self, X_title, X_description, X_author, X_publisher, X_categories, X_year, X_train_title, X_train_description, X_train_author, X_train_publisher, X_train_categories, X_train_year, X_test_title, X_test_description, X_test_author, X_test_publisher, X_test_categories, X_test_year, y_train, y_test):
        self._define_input_layers(X_title, X_description, X_author, X_publisher, X_categories, X_year)
        self._define_embedding_layers(X_title, X_description, X_author, X_publisher, X_categories)
        self._flatten_embedding_layers()
        self._concatenate_layers()
        self._dense_layers()
        self._compile_model()
        self._train_model(X_train_title, X_train_description, X_train_author, X_train_publisher, X_train_categories, X_train_year, y_train)
        self._evaluate_model(X_test_title, X_test_description, X_test_author, X_test_publisher, X_test_categories, X_test_year, y_test)

    def _define_input_layers(self, X_title, X_description, X_author, X_publisher, X_categories, X_year):
        self.input_title = Input(shape=(X_title.shape[1],), name='Title_Input')
        self.input_description = Input(shape=(X_description.shape[1],), name='Description_Input')
        self.input_author = Input(shape=(X_author.shape[1],), name='Author_Input')
        self.input_publisher = Input(shape=(X_publisher.shape[1],), name='Publisher_Input')
        self.input_categories = Input(shape=(X_categories.shape[1],), name='Categories_Input')
        self.input_year = Input(shape=(X_year.shape[1],), name='Year_Input')

    def _define_embedding_layers(self, X_title, X_description, X_author, X_publisher, X_categories):
        tokenizer = Tokenizer()
        self.embedding_title = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=self.embedding_dim, input_length=X_title.shape[1])(self.input_title)
        self.embedding_description = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=self.embedding_dim, input_length=X_description.shape[1])(self.input_description)
        self.embedding_author = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=self.embedding_dim, input_length=X_author.shape[1])(self.input_author)
        self.embedding_publisher = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=self.embedding_dim, input_length=X_publisher.shape[1])(self.input_publisher)
        self.embedding_categories = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=self.embedding_dim, input_length=X_categories.shape[1])(self.input_categories)

    def _flatten_embedding_layers(self):
        self.flatten_title = Flatten()(self.embedding_title)
        self.flatten_description = Flatten()(self.embedding_description)
        self.flatten_author = Flatten()(self.embedding_author)
        self.flatten_publisher = Flatten()(self.embedding_publisher)
        self.flatten_categories = Flatten()(self.embedding_categories)

    def _concatenate_layers(self):
        self.concatenated = concatenate([self.flatten_title, self.flatten_description, self.flatten_author, self.flatten_publisher, self.flatten_categories, self.input_year])

    def _dense_layers(self):
        dense1 = Dense(128, activation='relu')(self.concatenated)
        dropout1 = Dropout(0.2)(dense1)
        dense2 = Dense(64, activation='relu')(dropout1)
        dropout2 = Dropout(0.2)(dense2)
        self.output = Dense(1, activation='linear')(dropout2)

    def _compile_model(self):
        self.model = Model(inputs=[self.input_title, self.input_description, self.input_author, self.input_publisher, self.input_categories, self.input_year], outputs=self.output)
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def _train_model(self, X_train_title, X_train_description, X_train_author, X_train_publisher, X_train_categories, X_train_year, y_train):
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        self.model.fit([X_train_title, X_train_description, X_train_author, X_train_publisher, X_train_categories, X_train_year], y_train, epochs=3, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    def _evaluate_model(self, X_test_title, X_test_description, X_test_author, X_test_publisher, X_test_categories, X_test_year, y_test):
        mse = self.model.evaluate([X_test_title, X_test_description, X_test_author, X_test_publisher, X_test_categories, X_test_year], y_test)
        print("Mean Squared Error:", mse)

        predicted_values = self.model.predict([X_test_title, X_test_description, X_test_author, X_test_publisher, X_test_categories, X_test_year])
        mape = mean_absolute_percentage_error(y_test, predicted_values)
        print("Mean Absolute Percentage Error:", mape)
