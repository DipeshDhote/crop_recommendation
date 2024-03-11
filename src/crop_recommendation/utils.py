import os 
import sys
from src.crop_recommendation.logger import logging
from src.crop_recommendation.exception import CustomException
import pandas as pd
import pymysql 
from dotenv import load_dotenv
import pickle
import numpy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pymysql
import numpy as np
from dotenv import load_dotenv


load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password= os.getenv("password")
db = os.getenv('db')


def read_sql_data():
    logging.info("reading mySQL database started")
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("connection established",mydb)
        df=pd.read_sql_query('select * from crop',mydb)
        print(df.head())

        return df
        

    except Exception as ex:
        raise CustomException(ex,sys)
    

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train,y_train_pred)

            test_model_score = accuracy_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)

