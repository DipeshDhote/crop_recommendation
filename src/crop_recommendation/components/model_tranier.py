import sys
from src.crop_recommendation.logger import logging
from src.crop_recommendation.exception import CustomException
from src.crop_recommendation.utils import save_object,evaluate_models
import mlflow
import os
from dataclasses import dataclass
from urllib.parse import urlparse
import numpy as np


from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostClassifier

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self,actual,pred):
          ac = accuracy_score(actual,pred)
          
          return ac


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splite training and test input data")

            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                "Decision Tree Classifier": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(),
                "Logistic Regression": LogisticRegression(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier()
                
            }
            logging.info("Reading all models")

            params={
                "Decision Tree Classifier": {
                    'criterion':['gini','entropy', 'log_loss'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest Classifier":{
                    'n_estimators': [8,16,32,75,100]
                    #'criterion':['gini','entropy', 'log_loss'],
                    #'max_features':['sqrt','log2',None],
                    
                },
                "Logistic Regression":{},
                "K-Neighbors Classifier":{},
                
                "AdaBoost Classifier":{}
            }     

            logging.info("completed hyperparametertuning")
            model_report:dict=evaluate_models(X_train,y_train,X_test,y_test,models,params)

            # to get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # to get the best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print("This is the best model:",best_model_name)

            model_names = list(params.keys())
            
            actual_model =""
            
            for model in model_names:
                if best_model_name == model:
                    actual_model = actual_model + model

            best_params = params[actual_model]

            mlflow.set_registry_uri("https://dagshub.com/DipeshDhote/crop_recommendation.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


            # mlflow

            with mlflow.start_run():

                predicted_qualities = best_model.predict(X_test)

                (ac) = self.eval_metrics(y_test, predicted_qualities)

                mlflow.log_params(best_params)

                mlflow.log_metric("Accuracy Score", ac)
                


                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=actual_model)
                else:
                    mlflow.sklearn.log_model(best_model, "model")

            

            if  best_model_score <0.6:
                raise CustomException("No best model found")
            logging.info("best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted=best_model.predict(X_test)

            ac_score = accuracy_score(y_test, predicted)
            return ac_score

        except Exception as e:
            raise CustomException(e,sys) 

        